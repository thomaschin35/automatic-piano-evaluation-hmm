"""
tempo.py
Tempo over time for one take, from the onset the model matched to each score note
(per_note.csv `onset_s`, original-file seconds).

  beat_tempo(rows, spec)     one point per gap between consecutive played score onsets:
                             BPM = 60 * (beats between the two notes) / (seconds between them)
  measure_tempo(rows, spec)  one point per measure: beats from its first played onset to the next
                             measure's first played onset, over the seconds between them

Both hands striking together count as one onset (the earliest of the two). Notes without an onset
(missed) are skipped, so a gap spans them. A gap whose onsets run backwards (alignment confusion)
is dropped. Note: these are the *model's* onsets, so a badly aligned take gives a wrong curve, and a
restart shows up as one very slow gap (the time spent going back).
"""
from __future__ import annotations
from dataclasses import dataclass

from eval.pieces import PieceSpec


@dataclass(frozen=True)
class TempoPoint:
    t_start: float      # seconds
    t_end: float
    beat_start: float   # score beats from the start of the piece
    beat_end: float
    bpm: float
    label: str          # "RH3→RH4" or "m2"

    @property
    def t_mid(self) -> float:
        return (self.t_start + self.t_end) / 2


def _onsets(rows: list[dict], spec: PieceSpec) -> list[tuple[float, float, str]]:
    """(score beat, seconds, note ids) per distinct score onset that the model placed, beat order."""
    notes = spec.note_by_id()
    by_beat: dict[float, tuple[float, list[str]]] = {}
    for r in rows:
        s = r.get("onset_s")
        if s in (None, "") or r["note_id"] not in notes:
            continue
        b, s = round(notes[r["note_id"]].start_beat, 4), float(s)
        t, ids = by_beat.get(b, (s, []))
        by_beat[b] = (min(t, s), ids + [r["note_id"]])
    return [(b, t, "+".join(ids)) for b, (t, ids) in sorted(by_beat.items())]


def beat_tempo(rows: list[dict], spec: PieceSpec) -> list[TempoPoint]:
    on = _onsets(rows, spec)
    out = []
    for (b0, t0, i0), (b1, t1, i1) in zip(on, on[1:]):
        if t1 > t0:
            out.append(TempoPoint(t0, t1, b0, b1, 60 * (b1 - b0) / (t1 - t0), f"{i0}→{i1}"))
    return out


def measure_tempo(rows: list[dict], spec: PieceSpec) -> list[TempoPoint]:
    on = _onsets(rows, spec)
    bpm_ = spec.beats_per_measure
    measure = lambda b: int((b - spec.pickup_beats) // bpm_) + 1
    firsts: dict[int, tuple[float, float]] = {}
    lasts: dict[int, tuple[float, float]] = {}
    for b, t, _ in on:
        firsts.setdefault(measure(b), (b, t))
        lasts[measure(b)] = (b, t)
    ms = sorted(firsts)
    out = []
    for m in ms:
        (b0, t0) = firsts[m]
        (b1, t1) = firsts[m + 1] if m + 1 in firsts else lasts[m]   # last measure: its own notes only
        if t1 > t0 and b1 > b0:
            out.append(TempoPoint(t0, t1, b0, b1, 60 * (b1 - b0) / (t1 - t0), f"m{m}"))
    return out


def steadiness(points: list[TempoPoint]) -> dict:
    """Mean BPM (time-weighted) and spread for a tempo curve."""
    if not points:
        return {"mean_bpm": None, "cv_pct": None, "min_bpm": None, "max_bpm": None}
    w = [p.t_end - p.t_start for p in points]
    mean = sum(p.bpm * x for p, x in zip(points, w)) / sum(w)
    var = sum(x * (p.bpm - mean) ** 2 for p, x in zip(points, w)) / sum(w)
    return {"mean_bpm": mean, "cv_pct": 100 * var ** 0.5 / mean,
            "min_bpm": min(p.bpm for p in points), "max_bpm": max(p.bpm for p in points)}
