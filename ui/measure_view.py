"""
ui/measure_view.py
Everything the algorithm saw in one bar of a graded take, for the page's bar inspector.

For a bar it returns, over that bar's stretch of the recording (plus a little either side):
  * the waveform (min/max envelope) and the frame-level loudness and broadband onset
  * per key, three maps: the raw spectrum level at the key's fundamental (dB, before any
    processing), the key's presence score (after overtone removal: what v3 reads as "sounding")
    and its onset score (how sharply it was struck)
  * every strike v3 detected there: which keys it kept, which it removed and why (overtone /
    leakage of another key, phantom), keys seen only as a jump in the raw spectrum, and which
    written note the strike was matched to
  * the written notes with the verdicts, and where each beat falls in time
"""
from __future__ import annotations
import numpy as np

from score_models import midi_to_note_name as name
from stft_features import STFTConfig, frequency_to_stft_bin, midi_to_frequency

WAVE_POINTS = 1400
PAD_S = 0.3


def _r(a, nd: int = 3) -> list[float]:
    return [round(float(x), nd) for x in a]


class Clock:
    """Score beat -> seconds in the uploaded file, from the notes the alignment matched
    (those of one attempt, when the player went back)."""

    def __init__(self, an, notes: list[dict]):
        pts: dict[float, list[float]] = {}
        for x in notes:
            if x["time_s"] is not None and x["label"] != "missed":
                pts.setdefault(x["start_beat"], []).append(x["time_s"])
        beats, times = [], []
        for b in sorted(pts):
            t = float(np.median(pts[b]))
            if not times or t > times[-1]:                  # keep it monotonic (restarts, outliers)
                beats.append(b); times.append(t)
        self.b, self.t = np.array(beats), np.array(times)
        spb = float(np.median(an.seconds_per_beat)) if len(an.seconds_per_beat) else 0.5
        self.spb = spb if spb > 0 else 0.5

    def __call__(self, beat: float) -> float:
        if len(self.b) == 0:
            return beat * self.spb
        if len(self.b) == 1 or beat < self.b[0]:
            return float(self.t[0] + (beat - self.b[0]) * self.spb)
        if beat > self.b[-1]:
            return float(self.t[-1] + (beat - self.b[-1]) * self.spb)
        return float(np.interp(beat, self.b, self.t))


def measure_detail(bundle, an, rep, spec, result: dict, measure: int, attempt: int | None = None) -> dict:
    """attempt: show the bar as played in that attempt (its notes and timing); None = the final grade."""
    audio = bundle.audio
    spf, off, sr = an.seconds_per_frame, audio.trim_offset_seconds, audio.sample_rate
    att = next((a for a in result.get("attempts", []) if a["n"] == attempt), None) if attempt else None
    graded = att["notes"] if att else result["notes"]
    clock = Clock(an, graded)
    bpm, pick = spec.beats_per_measure, spec.pickup_beats
    measures = sorted({n["measure"] for n in result["notes"]})
    if measure not in measures:
        raise ValueError(f"Bar {measure} is not in the piece (bars {measures[0]}-{measures[-1]})")
    start_beat = max(0.0, pick + (measure - 1) * bpm) if measure > 0 else 0.0
    end_beat = pick + measure * bpm if measure > 0 else pick
    duration = len(audio.waveform) / sr + off
    t0 = max(off, clock(start_beat) - PAD_S)
    t1 = min(duration, max(clock(end_beat) + PAD_S, t0 + 0.5))
    T = audio.num_frames
    f0 = max(0, int(np.floor((t0 - off) / spf)))
    f1 = min(T, int(np.ceil((t1 - off) / spf)) + 1)
    frames = np.arange(f0, f1)

    # waveform envelope (the trimmed, normalised signal the features were computed from)
    s0, s1 = max(0, int((t0 - off) * sr)), min(len(audio.waveform), int((t1 - off) * sr))
    w = audio.waveform[s0:s1]
    n = max(1, min(WAVE_POINTS, len(w)))
    edges = np.linspace(0, len(w), n + 1).astype(int)
    wmin = [float(w[a:b].min()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    wmax = [float(w[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]

    notes = [x for x in graded if x["measure"] == measure]
    m_of = {x["id"]: x for x in result["notes"]}
    events = [(i, e) for i, e in enumerate(an.events) if f0 <= e.frame < f1]
    keys = {x["midi"] for x in notes}
    keys |= {_midi(x["played"]) for x in notes if x["played"]}
    for _, e in events:
        keys |= {p for p, _ in e.primary} | set(e.explained_by) | set(e.phantom)
    keys = {k for k in keys if k}
    lo = max(min(bundle.candidate_scores), min(keys) - 2)
    hi = min(max(bundle.candidate_scores), max(keys) + 12)          # an octave above: overtones show there
    rows = [k for k in range(lo, hi + 1) if k in bundle.candidate_scores]

    stft = STFTConfig()
    level = []
    for k in rows:
        b = int(round(frequency_to_stft_bin(midi_to_frequency(k), sr, stft.n_fft)))
        level.append(_r(audio.stft_db[min(b, audio.stft_db.shape[0] - 1), f0:f1], 1))

    matched = {m.event: m for m in an.matches}
    by_place = {(round(n.start_beat, 4), n.pitch): n for n in reversed(spec.notes())}   # right hand wins
    by_beat: dict[float, list[str]] = {}
    for x in result["notes"]:
        by_beat.setdefault(round(x["start_beat"], 4), []).append(x["id"])
    strikes = []
    for i, e in events:
        m = matched.get(i)
        ids = by_beat.get(round(an.positions[m.position].beat, 4), []) if m else []
        strikes.append({
            "time_s": round(e.frame * spf + off, 3),
            "kept": [[name(p), round(s, 3)] for p, s in e.primary],
            "candidates": [[name(p), round(s, 3)] for p, s in sorted(e.strikes.items(), key=lambda x: -x[1])[:8]],
            "removed": [[name(p), name(by)] for p, by in e.explained_by.items()],
            "phantom": [[name(p), name(by)] for p, by in e.phantom.items()],
            "raw_only": [name(p) for p in sorted(e.raw_only)],
            "rise_db": {name(p): round(float(v), 1) for p, v in e.rise.items()},
            "matched": [{"id": j, "pitch": m_of[j]["pitch"]} for j in ids],
            "match_kind": m.kind if m else None,
            "attempt": rep.attempt_of.get(i) if m else None,
            # the verdicts of this visit (not the note's final one: a note played twice has two)
            "verdicts": [{"id": by_place[(round(an.positions[m.position].beat, 4), v.pitch)].id, "pitch": name(v.pitch),
                          "label": v.label, "played": name(v.played) if v.label == "wrong_pitch" and v.played else None,
                          "late": "late" in v.flags} for v in rep.by_event.get(i, [])] if m else [],
        })

    beats = []
    b = start_beat
    while b <= end_beat + 1e-9:
        beats.append({"beat": round(b - start_beat + 1, 3), "time_s": round(clock(b), 3)})
        b += 1.0
    return {
        "measure": measure, "measures": measures, "attempt": attempt, "t0": round(t0, 3), "t1": round(t1, 3),
        "wave": {"t0": round(s0 / sr + off, 4), "t1": round(s1 / sr + off, 4), "min": _r(wmin), "max": _r(wmax)},
        "frames": {"t0": round(f0 * spf + off, 4), "dt": spf, "n": len(frames)},
        "energy": _r(bundle.frame_energy[f0:f1]), "onset": _r(bundle.broadband_onset[f0:f1]),
        "keys": [{"midi": k, "name": name(k)} for k in rows],
        "level_db": level,
        "presence": [_r(bundle.candidate_scores[k][f0:f1]) for k in rows],
        "onset_score": [_r(bundle.candidate_onset_scores[k][f0:f1]) for k in rows],
        "notes": [{**x, "expected_s": round(clock(x["start_beat"]), 3)} for x in notes],
        "strikes": strikes, "beats": beats,
    }


def _midi(note_name: str) -> int:
    from eval.pieces import parse_pitch
    return parse_pitch(note_name)
