"""
grade.py
One entry point for the whole algorithm: a score + a recording in, a graded performance out.

    .venv/bin/python grade.py --audio take.wav --piece ode_melody        # a piece from eval/pieces/
    .venv/bin/python grade.py --audio take.m4a --score sheet.pdf         # sheet music, read automatically
    .venv/bin/python grade.py --audio take.wav --score sheet.musicxml --json

From Python:

    from grade import grade
    from score_reader import read_score
    spec, warnings = read_score("sheet.pdf")
    result = grade("take.wav", spec)          # plain dict, ready for json.dumps

The grading itself is stft_hmm_v3 (strikes -> note-level HMM alignment -> verdicts), unchanged.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

import numpy as np

import score_models
import stft_features
import stft_hmm_v3 as v3
from score_models import midi_to_note_name
from eval.pieces import PieceSpec, load_piece

NATIVE_AUDIO = {".wav", ".flac", ".ogg", ".mp3"}       # read directly; anything else goes through ffmpeg


def _readable_audio(path: Path, workdir: Path) -> Path:
    if path.suffix.lower() in NATIVE_AUDIO:
        return path
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"ffmpeg is needed to read {path.suffix} recordings (or upload a .wav / .mp3)")
    out = workdir / "audio.wav"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                        "-ac", "1", "-ar", "22050", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Could not read the recording: {r.stderr.strip()[-300:]}")
    return out


def run(audio_path: str | Path, spec: PieceSpec, cfg: v3.V3Config | None = None):
    """Run the algorithm. Returns (bundle, analysis, report) exactly as stft_hmm_v3 produces them."""
    cfg = cfg or v3.V3Config()
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Recording not found: {audio_path}")
    key = f"_grade_{uuid.uuid4().hex}"                  # stage 1 looks the score up by name
    score_models.SCORE_MODELS[key] = spec.to_score_model
    try:
        with tempfile.TemporaryDirectory(prefix="grade_") as tmp:
            bundle = stft_features.run_stage1_and_stage2(_readable_audio(audio_path, Path(tmp)), key)
    finally:
        del score_models.SCORE_MODELS[key]
    an = v3.analyze(bundle, cfg, spec.beats_per_measure, spec.pickup_beats)
    return bundle, an, v3.report(an, cfg, bundle)


def grade(audio_path: str | Path, spec: PieceSpec, cfg: v3.V3Config | None = None) -> dict:
    """Grade a recording against a score. Returns a JSON-ready dict (see `to_result`)."""
    bundle, an, rep = run(audio_path, spec, cfg)
    return to_result(spec, bundle, an, rep)


def to_result(spec: PieceSpec, bundle, an, rep) -> dict:
    """Verdicts per written note, extras, restarts and a summary. Times are seconds in the uploaded file."""
    spf, offset = an.seconds_per_frame, bundle.audio.trim_offset_seconds

    def at(frame) -> float | None:
        return None if frame is None else round(frame * spf + offset, 3)

    by_place: dict[tuple[float, int], object] = {}       # (start beat, pitch) -> ScoreNote, right hand first
    for n in spec.notes():
        by_place.setdefault((round(n.start_beat, 4), n.pitch), n)

    def place(pos_index: int | None) -> dict | None:
        """Where a score position is: the id, bar and beat of its (first) note."""
        if pos_index is None:
            return None
        pos = an.positions[pos_index]
        n = by_place[(round(pos.beat, 4), pos.pitches[0])]
        return {"id": n.id, "pitch": n.name, "measure": n.measure, "beat": round(n.beat_in_measure, 3)}

    def row(pos, v) -> dict:
        n = by_place[(round(pos.beat, 4), v.pitch)]
        late = "late" in v.flags
        return {
                "id": n.id, "hand": n.hand, "pitch": n.name, "midi": n.pitch,
                "measure": n.measure, "beat": round(n.beat_in_measure, 3),
                "start_beat": n.start_beat, "beats": round(n.end_beat - n.start_beat, 4),
                "label": v.label,                                    # correct | wrong_pitch | missed
                "played": midi_to_note_name(v.played) if v.label == "wrong_pitch" and v.played else None,
                "corrected": bool(v.corrected),
                "time_s": at(v.frame),
                "late": late, "late_beats": round(v.late_beats, 2) if late else 0.0,
                "late_reason": v.late_reason if late else "",       # held | paused | hands not together
                "held_long": "held_long" in v.flags,                 # key still down under the next note
                "times_played": len(rep.visits.get(pos.index, [])),
            }

    notes = [row(pos, v) for pos in an.positions for v in rep.verdicts[pos.index]]

    attempts, judged_by = _attempts(an, rep, row, place, at)
    for x in notes:
        x["attempt"] = judged_by.get(x["id"], len(attempts) or 1)

    def attempt_at(frame) -> int:
        """The attempt being played at this frame: the last one that started before it."""
        k = 1
        for a in attempts:
            if a["first_frame"] <= frame:
                k = a["n"]
        return k

    extras = [{"pitch": midi_to_note_name(p), "time_s": at(f), "after": place(a), "attempt": attempt_at(f)}
              for a, p, f in rep.extras]
    for a in attempts:
        a["summary"]["extra"] = sum(1 for x in extras if x["attempt"] == a["n"])

    back_frames = [an.events[m.event].frame for m in an.matches if m.kind == "back"]
    restarts = []
    for k, (a, b) in enumerate(rep.restarts, 1):
        mistakes = []
        for att, pi, v in rep.earlier:
            if att == k:
                n = by_place[(round(an.positions[pi].beat, 4), v.pitch)]
                mistakes.append({"id": n.id, "pitch": n.name, "measure": n.measure,
                                 "beat": round(n.beat_in_measure, 3), "label": v.label,
                                 "played": midi_to_note_name(v.played) if v.label == "wrong_pitch" and v.played else None})
        restarts.append({"from": place(a), "to": place(b),
                         "time_s": at(back_frames[k - 1]) if k - 1 < len(back_frames) else None,
                         "mistakes": mistakes})

    count = lambda pred: sum(1 for n in notes if pred(n))
    clean = count(lambda n: n["label"] == "correct" and not n["late"])
    spb = float(np.median(an.seconds_per_beat)) if len(an.seconds_per_beat) else 0.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v3.print_report(an, rep)
    return {
        "piece": {"title": spec.title, "time_signature": list(spec.time_signature),
                  "beats_per_measure": spec.beats_per_measure, "pickup_beats": spec.pickup_beats,
                  "two_hands": len(spec.playing_hands) == 2, "hands": spec.playing_hands,
                  "chords": spec.has_chords},
        "summary": {
            "notes": len(notes), "clean": clean,
            "score_pct": round(100.0 * clean / len(notes), 1) if notes else 0.0,
            "correct": count(lambda n: n["label"] == "correct"),
            "wrong_pitch": count(lambda n: n["label"] == "wrong_pitch"),
            "missed": count(lambda n: n["label"] == "missed"),
            "late": count(lambda n: n["late"]),
            "held_long": count(lambda n: n["held_long"]),
            "extra": len(extras), "restarts": len(restarts),
            "tempo_bpm": round(60.0 / spb, 1) if spb > 0 else None,
            "duration_s": round(len(bundle.audio.waveform) / bundle.audio.sample_rate + offset, 2),
        },
        "notes": notes, "extras": extras, "restarts": restarts, "attempts": attempts,
        "report_text": buf.getvalue(),
    }


def _attempts(an, rep, row, place, at) -> tuple[list[dict], dict[str, int]]:
    """Every attempt graded on its own. The take starts in attempt 1 and each restart starts the next.
    An attempt covers the notes from where it started to where it got to; a note it jumped over is
    missed in it, and the notes it never reached are not part of it. Returns the attempts and, per
    note id, the attempt that gives its final verdict (the last one that reached it)."""
    out: list[dict] = []
    last = None
    for m in sorted(an.matches, key=lambda m: m.event):
        k = rep.attempt_of.get(m.event, 1)
        if not out or out[-1]["n"] != k:
            out.append({"n": k, "visits": {}, "skipped": [], "frames": []})
            last = None if k == 1 else m.position
            if k == 1:
                out[-1]["skipped"] += list(range(0, m.position))      # the take began past these notes
        cur = out[-1]
        if last is not None and m.position > last + 1:
            cur["skipped"] += list(range(last + 1, m.position))
        cur["visits"][m.position] = rep.by_event[m.event]
        cur["frames"].append(an.events[m.event].frame)
        last = m.position
    judged_by: dict[str, int] = {}
    result = []
    for i, a in enumerate(out):
        reached = sorted(set(a["visits"]) | set(a["skipped"]))
        notes = []
        for q in reached:
            pos = an.positions[q]
            vs = a["visits"].get(q) or [v3.PitchVerdict(p, "missed") for p in pos.pitches]
            for v in vs:
                r = row(pos, v)
                r["times_played"] = 1 if q in a["visits"] else 0
                notes.append(r)
                judged_by[r["id"]] = a["n"]
        clean = sum(1 for x in notes if x["label"] == "correct" and not x["late"])
        final = i == len(out) - 1
        back = rep.restarts[i] if not final and i < len(rep.restarts) else None
        result.append({
            "n": a["n"], "final": final,
            "first_frame": min(a["frames"]), "start_s": at(min(a["frames"])), "end_s": at(max(a["frames"])),
            "from": place(reached[0]) if reached else None, "to": place(reached[-1]) if reached else None,
            "went_back_to": place(back[1]) if back else None,
            "notes": notes,
            "summary": {"notes": len(notes), "clean": clean,
                        "score_pct": round(100.0 * clean / len(notes), 1) if notes else 0.0,
                        "wrong_pitch": sum(1 for x in notes if x["label"] == "wrong_pitch"),
                        "missed": sum(1 for x in notes if x["label"] == "missed"),
                        "late": sum(1 for x in notes if x["late"])},
        })
    return result, judged_by


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade a piano recording against a score with stft_hmm_v3.")
    ap.add_argument("--audio", required=True, help="recording (.wav .mp3 .flac .ogg; .m4a and others via ffmpeg)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--piece", help="name of a piece in eval/pieces/, e.g. ode_melody")
    src.add_argument("--score", help="sheet music: .pdf .png .jpg (read automatically), .musicxml .mxl, or a piece .toml")
    ap.add_argument("--json", action="store_true", help="print the result as JSON instead of the text report")
    a = ap.parse_args()

    if a.piece:
        spec, warnings = load_piece(a.piece), []
    else:
        from score_reader import read_score
        spec, warnings = read_score(a.score)
    result = grade(a.audio, spec)
    if a.json:
        print(json.dumps({**result, "warnings": warnings}, indent=2))
        return
    for w in warnings:
        print(f"note: {w}")
    s = result["summary"]
    print(result["report_text"], end="")
    print(f"{spec.title}: {s['clean']}/{s['notes']} notes right and on time ({s['score_pct']}%)  |  "
          f"wrong {s['wrong_pitch']}, missed {s['missed']}, late {s['late']}, held too long {s['held_long']}, extra {s['extra']}, "
          f"went back {s['restarts']}x  |  about {s['tempo_bpm']} bpm")


if __name__ == "__main__":
    main()
