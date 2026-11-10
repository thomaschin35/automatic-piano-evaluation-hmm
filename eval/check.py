"""
check.py  --  one scoreboard for "is the model closer to the goal?"

  real takes   strict per-class scores on every labelled recording (all / monophonic / two-hand /
               monophonic by player), clean takes passed
  synthetic    eval/synth.py cases built from the clean monophonic takes
  onset-only   clean monophonic takes decoded without the pitch term (is the alignment timing-led?)

`python -m eval.cli check --model v3` prints it and the change against the accepted scoreboard
(eval/scoreboards/<model>.json, or --against another model's); `--accept` stores it.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import time

from eval import EVAL_DIR
from eval.cache import load_bundle
from eval.labels import load_manifest
from eval.metrics import aggregate, evaluate_take
from eval.pieces import all_pieces
from eval import synth

BOARD_DIR = EVAL_DIR / "scoreboards"
CLASSES = ("error", "strict", "goal", "wrong_pitch", "missed", "extra", "held_long", "late")
# metrics that must not drop for a change to be accepted
GUARDED = ("real.mono.goal.f1", "real.mono.strict.f1", "real.mono.clean", "real.all.strict.f1", "real.all.clean")


def _block(evals) -> dict:
    if not evals:
        return {"n_takes": 0}
    a = aggregate(evals)
    out = {"n_takes": a["n_takes"], "n_notes": a["n_notes"],
           "clean": a["perfect_takes"]["clean"], "n_perfect": a["perfect_takes"]["n"],
           "played_pitch_accuracy": a["played_pitch_accuracy"], "failed": a["failed_takes"]}
    for k in CLASSES:
        out[k] = {x: a[k][x] for x in ("tp", "fp", "fn", "precision", "recall", "f1")}
    return out


def _ready_takes(pieces):
    from eval.cli import check_take, model_audio
    for t in load_manifest():
        status, spec, labels = check_take(t, pieces)
        if status == "ok":
            yield t, spec, labels, model_audio(t.audio)


def _predict(adapter, bundle, spec, overrides=None):
    try:
        pred = adapter.predict_bundle(bundle, spec, overrides)
        pred.validate([n.id for n in spec.notes()])
        return pred
    except Exception:
        return None


def build(adapter, kinds=synth.KINDS, with_synth: bool = True) -> dict:
    pieces = all_pieces()
    t0 = time.time()
    real, clean_mono = [], []
    for t, spec, labels, audio in _ready_takes(pieces):
        bundle = load_bundle(audio, t.piece)
        ev = evaluate_take(t.id, spec, labels, _predict(adapter, bundle, spec))
        real.append(ev)
        if labels.perfect and t.piece in synth.MONO_PIECES:
            clean_mono.append((t, spec, bundle))
    mono = [e for e in real if e.piece in synth.MONO_PIECES]
    board = {"model": adapter.name, "real": {
        "all": _block(real),
        "mono": _block(mono),
        "two_hand": _block([e for e in real if e.piece not in synth.MONO_PIECES]),
        "mono_p1": _block([e for e in mono if e.take_id.startswith("1")]),
        "mono_p2": _block([e for e in mono if e.take_id.startswith("2")]),
    }, "real_errors": [
        f"{r['take']} {r['note_id']} {r['pitch']} gold={r['gold']}{'+' + r['gold_flags'] if r['gold_flags'] else ''}"
        f" pred={r['pred']}{'+' + r['pred_flags'] if r['pred_flags'] else ''}"
        for e in real for r in e.rows if not r["label_ok"] or r["gold_flags"] != r["pred_flags"]]
        + [f"{e.take_id} FAILED" for e in real if e.failed]
        + [f"{e.take_id} extras tp/fp/fn={e.counts['extra.tp']}/{e.counts['extra.fp']}/{e.counts['extra.fn']}"
           for e in real if e.counts["extra.fp"] or e.counts["extra.fn"]]}

    # onset-only: clean monophonic takes, pitch term switched off
    ok = 0
    for t, spec, bundle in clean_mono:
        ev = evaluate_take(t.id, spec, type("L", (), {
            "perfect": True, "note_labels": lambda self, s: {n.id: "correct" for n in s.notes()},
            "played": lambda self: {}, "flags": lambda self: {}, "extras": lambda self: []})(),
            _predict(adapter, bundle, spec, adapter.onset_only))
        ok += int(not ev.failed and ev.pred_errors == 0)
    board["onset_only"] = {"clean": ok, "n": len(clean_mono)}

    board["synth"] = {}
    if with_synth:
        per_kind: dict[str, list] = {k: [] for k in kinds}
        exact: Counter = Counter()
        located: Counter = Counter()
        for t, spec, bundle in clean_mono:
            for case in synth.generate(t.id, bundle, spec, kinds):
                pred = _predict(adapter, case.bundle, case.spec)
                ev = evaluate_take(f"{t.id}:{case.kind}", case.spec, case.gold, pred)
                per_kind[case.kind].append(ev)
                c = ev.counts
                exact[case.kind] += int(not ev.failed and sum(
                    c[f"{k}.{x}"] for k in ("strict", "extra", "held_long", "late") for x in ("fp", "fn")) == 0)
                if case.kind == "restart" and pred is not None:
                    located[case.kind] += int(case.meta["restart"] in [tuple(r) for r in pred.restarts])
        for k, evs in per_kind.items():
            if not evs:
                continue
            b = _block(evs)
            b.update(n_cases=len(evs), exact=exact[k] / len(evs),
                     false_flags=sum(b[k]["fp"] for k in ("strict", "extra", "held_long", "late")) / len(evs))
            if k == "restart":
                b["located"] = located[k] / len(evs)
            board["synth"][k] = b
    board["seconds"] = round(time.time() - t0, 1)
    return board


# ---------------------------------------------------------------- flatten / print / diff
def flatten(board: dict) -> dict[str, float]:
    out = {}
    for g, b in board["real"].items():
        if not b.get("n_takes"):
            continue
        out[f"real.{g}.clean"] = b["clean"]
        for k in CLASSES:
            if b[k]["f1"] is not None:
                out[f"real.{g}.{k}.f1"] = b[k]["f1"]
    out["onset_only.clean"] = board["onset_only"]["clean"]
    for k, b in board["synth"].items():
        out[f"synth.{k}.exact"] = b["exact"]
        for c in ("strict", "extra", "held_long", "late"):
            if b[c]["recall"] is not None:
                out[f"synth.{k}.{c}.recall"] = b[c]["recall"]
        out[f"synth.{k}.false_flags"] = b["false_flags"]
        if "located" in b:
            out[f"synth.{k}.located"] = b["located"]
    return out


def _f(x) -> str:
    return "  -  " if x is None else f"{x:5.2f}"


def render(board: dict) -> str:
    L = [f"SCOREBOARD  model={board['model']}  ({board['seconds']}s)", "",
         "REAL TAKES        takes notes | goal F1 | strict P     R    F1 | wrong missed extra  held  late | clean"]
    for g, b in board["real"].items():
        if not b.get("n_takes"):
            continue
        s = b["strict"]
        L.append(f"  {g:<15} {b['n_takes']:>5} {b['n_notes']:>5} |  {_f(b['goal']['f1'])}  |      {_f(s['precision'])} {_f(s['recall'])} {_f(s['f1'])} |"
                 f" {_f(b['wrong_pitch']['f1'])} {_f(b['missed']['f1'])} {_f(b['extra']['f1'])} {_f(b['held_long']['f1'])} {_f(b['late']['f1'])} |"
                 f" {b['clean']}/{b['n_perfect']}")
    m = board["real"]["mono"]
    if m.get("n_takes"):
        L.append(f"  goal counts (mono): strict tp/fp/fn={m['strict']['tp']}/{m['strict']['fp']}/{m['strict']['fn']}  "
                 f"held_long tp/fp/fn={m['held_long']['tp']}/{m['held_long']['fp']}/{m['held_long']['fn']}  "
                 f"late={m['late']['tp']}/{m['late']['fp']}/{m['late']['fn']}  extra={m['extra']['tp']}/{m['extra']['fp']}/{m['extra']['fn']}")
    o = board["onset_only"]
    L += ["", f"ONSET-ONLY (pitch term off): {o['clean']}/{o['n']} clean monophonic takes still flag nothing"]
    if board["synth"]:
        L += ["", "SYNTHETIC (clean mono takes)  cases | exact | strict P     R    F1 | extra R  held R  late R | false flags/case"]
        for k, b in board["synth"].items():
            s = b["strict"]
            loc = f" | restart located {_f(b['located'])}" if "located" in b else ""
            L.append(f"  {k:<26} {b['n_cases']:>6} | {_f(b['exact'])} |      {_f(s['precision'])} {_f(s['recall'])} {_f(s['f1'])} |"
                     f"  {_f(b['extra']['recall'])}   {_f(b['held_long']['recall'])}   {_f(b['late']['recall'])} | {b['false_flags']:.2f}{loc}")
    if board["real_errors"]:
        L += ["", f"REAL-TAKE DISAGREEMENTS ({len(board['real_errors'])})"] + [f"  {e}" for e in board["real_errors"]]
    return "\n".join(L)


def diff(new: dict, old: dict, tol: float = 1e-9) -> tuple[list[str], list[str]]:
    """(all changes, regressions on guarded metrics)."""
    a, b = flatten(old), flatten(new)
    changes, regressions = [], []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x is None or y is None or abs(x - y) <= tol:
            continue
        lower_is_better = k.endswith("false_flags")
        worse = (y > x) if lower_is_better else (y < x)
        changes.append(f"  {'WORSE ' if worse else 'better'} {k:<40} {x:.3f} -> {y:.3f}")
        if worse and k in GUARDED:
            regressions.append(k)
    return changes, regressions


def board_path(model: str) -> Path:
    return BOARD_DIR / f"{model}.json"


def load(model: str) -> dict | None:
    p = board_path(model)
    return json.loads(p.read_text()) if p.exists() else None


def save(board: dict) -> Path:
    BOARD_DIR.mkdir(exist_ok=True)
    p = board_path(board["model"])
    p.write_text(json.dumps(board, indent=1, default=str))
    return p
