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

    notes = []
    for pos in an.positions:
        for v in rep.verdicts[pos.index]:
            n = by_place[(round(pos.beat, 4), v.pitch)]
            late = "late" in v.flags
            notes.append({
                "id": n.id, "hand": n.hand, "pitch": n.name,
                "measure": n.measure, "beat": round(n.beat_in_measure, 3),
                "start_beat": n.start_beat, "beats": round(n.end_beat - n.start_beat, 4),
                "label": v.label,                                    # correct | wrong_pitch | missed
                "played": midi_to_note_name(v.played) if v.label == "wrong_pitch" and v.played else None,
                "corrected": bool(v.corrected),
                "time_s": at(v.frame),
                "late": late, "late_beats": round(v.late_beats, 2) if late else 0.0,
                "late_reason": v.late_reason if late else "",       # held | paused
                "times_played": len(rep.visits.get(pos.index, [])),
            })

    extras = [{"pitch": midi_to_note_name(p), "time_s": at(f), "after": place(a)} for a, p, f in rep.extras]

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
                  "two_hands": not spec.is_mono},
        "summary": {
            "notes": len(notes), "clean": clean,
            "score_pct": round(100.0 * clean / len(notes), 1) if notes else 0.0,
            "correct": count(lambda n: n["label"] == "correct"),
            "wrong_pitch": count(lambda n: n["label"] == "wrong_pitch"),
            "missed": count(lambda n: n["label"] == "missed"),
            "late": count(lambda n: n["late"]),
            "extra": len(extras), "restarts": len(restarts),
            "tempo_bpm": round(60.0 / spb, 1) if spb > 0 else None,
            "duration_s": round(len(bundle.audio.waveform) / bundle.audio.sample_rate + offset, 2),
        },
        "notes": notes, "extras": extras, "restarts": restarts,
        "report_text": buf.getvalue(),
    }


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
          f"wrong {s['wrong_pitch']}, missed {s['missed']}, late {s['late']}, extra {s['extra']}, "
          f"went back {s['restarts']}x  |  about {s['tempo_bpm']} bpm")


if __name__ == "__main__":
    main()
