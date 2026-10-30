"""
metrics.py
Scores TakePredictions against gold labels.

Headline: note-level error F1. A note is a positive if it has any note error
(wrong_pitch or missed); a hit needs gold and prediction to agree on the same
note. Also computed: per-type P/R/F1, extras (matched within +/-1 note of the
same hand), flags, a combined F1 over everything, a confusion matrix,
false alarms on perfect takes and recording-level pass/fail accuracy.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field

from eval.contract import ERROR_TYPES, EXTRA, FLAGS, NOTE_ERRORS, NOTE_LABELS, TakePrediction
from eval.labels import TakeLabels
from eval.pieces import PieceSpec

COUNT_KEYS = ("error",) + ERROR_TYPES + ("combined",)


@dataclass
class TakeEval:
    take_id: str
    piece: str
    perfect: bool
    failed: bool
    counts: Counter = field(default_factory=Counter)        # "<key>.tp|fp|fn"
    confusion: Counter = field(default_factory=Counter)     # (gold, pred)
    played_hits: int = 0
    played_total: int = 0
    pred_errors: int = 0
    gold_errors: int = 0
    unsupported: frozenset[str] = frozenset()
    rows: list[dict] = field(default_factory=list)


def _position(after: str | None, spec_ids: dict) -> tuple[str | None, int]:
    if after is None:
        return (None, 0)
    n = spec_ids[after]
    return (n.hand, n.index)


def _match_extras(gold: list, pred: list, spec_ids: dict) -> int:
    """Greedy one-to-one matching; a pair matches if same hand (or START) and |index diff| <= 1."""
    gpos = [_position(g.after, spec_ids) for g in gold]
    ppos = [_position(p.after_note_id, spec_ids) for p in pred]
    used, hits = set(), 0
    for gh, gi in gpos:
        best = None
        for j, (ph, pi) in enumerate(ppos):
            if j in used or abs(gi - pi) > 1:
                continue
            if gh is not None and ph is not None and gh != ph:
                continue
            if best is None or abs(gi - pi) < abs(gi - ppos[best][1]):
                best = j
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def evaluate_take(take_id: str, spec: PieceSpec, gold: TakeLabels,
                  pred: TakePrediction | None) -> TakeEval:
    """pred=None means the model crashed: every note counts as predicted correct."""
    ev = TakeEval(take_id, spec.name, gold.perfect, failed=pred is None)
    c = ev.counts
    spec_ids = spec.note_by_id()
    gold_labels = gold.note_labels(spec)
    gold_played = gold.played()
    gold_flags = gold.flags()
    pred_by_id = {n.note_id: n for n in pred.notes} if pred else {}
    caps = pred.capabilities if pred else frozenset(ERROR_TYPES)
    ev.unsupported = frozenset(t for t in ERROR_TYPES if t not in caps)

    def tally(key: str, g: bool, p: bool) -> None:
        if g and p:
            c[f"{key}.tp"] += 1
        elif p:
            c[f"{key}.fp"] += 1
        elif g:
            c[f"{key}.fn"] += 1

    for n in spec.notes():
        g = gold_labels[n.id]
        pn = pred_by_id.get(n.id)
        p = pn.label if pn else "correct"
        pflags = set(pn.flags) if pn else set()
        gflags = gold_flags.get(n.id, set())
        ev.confusion[(g, p)] += 1
        tally("error", g != "correct", p != "correct")
        tally("combined", g != "correct", p != "correct")
        for t in NOTE_ERRORS:
            tally(t, g == t, p == t)
        for fl in FLAGS:
            tally(fl, fl in gflags, fl in pflags)
            tally("combined", fl in gflags, fl in pflags)
        if g == "wrong_pitch" and p == "wrong_pitch" and n.id in gold_played:
            ev.played_total += 1
            ev.played_hits += int(pn.played_pitch == gold_played[n.id])
        ev.gold_errors += int(g != "correct") + len(gflags)
        ev.pred_errors += int(p != "correct") + len(pflags)
        ev.rows.append({
            "take": take_id, "piece": spec.name, "note_id": n.id, "pitch": n.name,
            "measure": n.measure, "beat": n.beat_in_measure,
            "gold": g, "pred": p,
            "gold_played": gold_played.get(n.id, ""),
            "pred_played": pn.played_pitch if pn and pn.played_pitch is not None else "",
            "gold_flags": "|".join(sorted(gflags)), "pred_flags": "|".join(sorted(pflags)),
            "onset_s": round(pn.onset_time_s, 3) if pn and pn.onset_time_s is not None else "",
            "label_ok": g == p,
        })

    gold_x, pred_x = gold.extras(), (pred.extras if pred else [])
    hits = _match_extras(gold_x, pred_x, spec_ids)
    for key in (EXTRA, "combined"):
        c[f"{key}.tp"] += hits
        c[f"{key}.fp"] += len(pred_x) - hits
        c[f"{key}.fn"] += len(gold_x) - hits
    ev.gold_errors += len(gold_x)
    ev.pred_errors += len(pred_x)
    return ev


# ---------------------------------------------------------------- aggregation
def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def aggregate(evals: list[TakeEval]) -> dict:
    total = Counter()
    conf = Counter()
    for ev in evals:
        total.update(ev.counts)
        conf.update(ev.confusion)
    unsupported = sorted(set().union(*(ev.unsupported for ev in evals))) if evals else []
    block = {k: prf(total[f"{k}.tp"], total[f"{k}.fp"], total[f"{k}.fn"]) for k in COUNT_KEYS}
    for t in unsupported:
        block[t]["unsupported"] = True
    perfect = [ev for ev in evals if ev.perfect]
    played_total = sum(ev.played_total for ev in evals)
    block.update({
        "n_takes": len(evals),
        "n_notes": sum(len(ev.rows) for ev in evals),
        "failed_takes": [ev.take_id for ev in evals if ev.failed],
        "unsupported_types": unsupported,
        "confusion": {g: {p: conf[(g, p)] for p in NOTE_LABELS} for g in NOTE_LABELS},
        "played_pitch_accuracy": (sum(ev.played_hits for ev in evals) / played_total) if played_total else None,
        "played_pitch_n": played_total,
        "perfect_takes": {
            "n": len(perfect),
            "clean": sum(1 for ev in perfect if ev.pred_errors == 0),
            "false_alarms_per_take": (sum(ev.pred_errors for ev in perfect) / len(perfect)) if perfect else None,
        },
        "pass_fail_accuracy": (sum(1 for ev in evals if (ev.pred_errors > 0) == (ev.gold_errors > 0))
                               / len(evals)) if evals else None,
    })
    return block


def _from_block(take_id: str, piece: str, b: dict) -> TakeEval:
    """Rebuild a TakeEval's counters from its saved single-take block (metrics.json by_take)."""
    counts = Counter({f"{k}.{x}": b[k][x] for k in COUNT_KEYS for x in ("tp", "fp", "fn")})
    conf = Counter({(g, p): n for g, row in b["confusion"].items() for p, n in row.items()})
    played_total = b["played_pitch_n"]
    return TakeEval(
        take_id, piece, perfect=b["perfect_takes"]["n"] == 1, failed=bool(b["failed_takes"]),
        counts=counts, confusion=conf,
        played_hits=round((b["played_pitch_accuracy"] or 0) * played_total), played_total=played_total,
        pred_errors=b["combined"]["tp"] + b["combined"]["fp"],
        gold_errors=b["combined"]["tp"] + b["combined"]["fn"],
        unsupported=frozenset(b["unsupported_types"]),
        rows=[{}] * b["n_notes"])


def subset(metrics: dict, rows: list[dict], pieces: set[str]) -> dict:
    """Re-summarize a saved run restricted to some pieces (same shape as summarize())."""
    piece_of = {r["take"]: r["piece"] for r in rows}
    evals = [_from_block(t, piece_of[t], b) for t, b in metrics["by_take"].items()
             if piece_of.get(t) in pieces]
    return summarize(evals)


def summarize(evals: list[TakeEval]) -> dict:
    pieces = sorted({ev.piece for ev in evals})
    return {
        "overall": aggregate(evals),
        "by_piece": {p: aggregate([ev for ev in evals if ev.piece == p]) for p in pieces},
        "by_take": {ev.take_id: aggregate([ev]) for ev in evals},
    }
