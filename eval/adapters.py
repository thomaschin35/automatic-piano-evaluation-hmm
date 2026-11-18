"""
adapters.py
Model registry. Each adapter turns one recording into a TakePrediction
(see eval/contract.py). The "v2" adapter wraps stft_hmm_v2, which grades
time slices, and maps each score note to the slice where it starts.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import contextlib
import io

import matplotlib
matplotlib.use("Agg")

import numpy as np

from eval.contract import ExtraNotePrediction, NotePrediction, TakePrediction
from eval.pieces import HANDS, PieceSpec


def _segment_for_note(segments, note):
    return next(s for s in segments if abs(s.start_beat - note.start_beat) < 1e-6)


def _sounding_end(scores: np.ndarray, start: int, threshold: float) -> int:
    """Last frame of the run that begins at `start` and stays above `threshold`."""
    end = start
    above = False
    for i in range(max(0, start), len(scores)):
        if scores[i] >= threshold:
            end = i
            above = True
        elif above:
            break
    return end


def duration_flags(results, segments, spec: PieceSpec, candidate_scores, interior_fpb: float,
                   cfg) -> dict[str, frozenset[str]]:
    """held_long when one note's IOI is long and its neighbors are not.

    A whole phrase played slowly has a longer interior beat, so every ratio stays
    near 1. A ritardando, where the neighbors are long too, is left as tempo.
    """
    by_index = {r.segment_index: r for r in results}
    onset: dict[str, int | None] = {}
    for n in spec.notes():
        r = by_index.get(_segment_for_note(segments, n).index)
        onset[n.id] = None if r is None or r.start_frame < 0 else r.start_frame

    ratios: dict[str, float] = {}
    for hand in HANDS:
        hand_notes = [n for n in spec.notes() if n.hand == hand]
        for i, n in enumerate(hand_notes):
            if onset[n.id] is None:
                continue
            nxt = next((m for m in hand_notes[i + 1:] if onset[m.id] is not None), None)
            if nxt is not None:
                frames = onset[nxt.id] - onset[n.id]
            else:
                series = None if candidate_scores is None else candidate_scores.get(n.pitch)
                if series is None:
                    continue
                end = _sounding_end(series, onset[n.id], cfg.present_threshold)
                frames = end - onset[n.id] + 1
            expected = max(n.end_beat - n.start_beat, 1e-6) * interior_fpb
            ratios[n.id] = frames / expected

    flags: dict[str, frozenset[str]] = {}
    for hand in HANDS:
        hand_notes = [n for n in spec.notes() if n.hand == hand]
        for i, n in enumerate(hand_notes):
            if ratios.get(n.id, 0.0) <= cfg.held_ratio:
                continue
            prev_r = next((ratios[m.id] for m in reversed(hand_notes[:i]) if m.id in ratios), None)
            next_r = next((ratios[m.id] for m in hand_notes[i + 1:] if m.id in ratios), None)
            if (prev_r is not None and prev_r > cfg.neighbor_ratio) or (
                    next_r is not None and next_r > cfg.neighbor_ratio):
                continue
            flags[n.id] = frozenset({"held_long"})
    return flags


def _anchor_note_id(spec: PieceSpec, segments, after_segment: int) -> str | None:
    seg = next((s for s in segments if s.index == after_segment), None)
    if seg is None:
        return None
    starters = [n for n in spec.notes() if abs(n.start_beat - seg.start_beat) < 1e-6]
    if starters:
        return starters[-1].id
    earlier = [n for n in spec.notes() if n.start_beat <= seg.start_beat + 1e-6]
    return earlier[-1].id if earlier else None


def _extra_pitch(onset_scores, start: int, end: int, exclude: set[int]) -> int | None:
    if not onset_scores:
        return None
    best_p, best_v = None, -1.0
    sl = slice(start, end + 1)
    for pitch, series in onset_scores.items():
        if pitch in exclude or pitch <= 0 or series.size == 0:
            continue
        strength = float(np.max(series[sl]))
        if strength > best_v:
            best_v, best_p = strength, pitch
    return best_p


def extras_from_insertions(insertions, results, segments, spec: PieceSpec,
                           seconds_per_frame: float, offset_s: float,
                           onset_scores) -> list[ExtraNotePrediction]:
    by_index = {r.segment_index: r for r in results}
    extras = []
    ordered = list(segments)
    for after_segment, start, end in insertions or []:
        exclude: set[int] = set()
        here = by_index.get(after_segment)
        if here is not None:
            exclude.update(here.expected_pitches)
        nxt = next((s for s in ordered if s.index > after_segment), None)
        if nxt is not None:
            exclude.update(nxt.expected_pitches)
        extras.append(ExtraNotePrediction(
            after_note_id=_anchor_note_id(spec, ordered, after_segment),
            pitch=_extra_pitch(onset_scores, start, end, exclude),
            time_s=start * seconds_per_frame + offset_s,
        ))
    return extras


def segments_to_note_predictions(results, segments, spec: PieceSpec,
                                 seconds_per_frame: float, offset_s: float,
                                 candidate_scores=None, interior_fpb: float | None = None,
                                 cfg=None) -> list[NotePrediction]:
    """Map v2 SegmentResults onto score notes via each note's onset segment."""
    by_index = {r.segment_index: r for r in results}
    flags = {}
    if interior_fpb is not None and cfg is not None:
        flags = duration_flags(results, segments, spec, candidate_scores, interior_fpb, cfg)
    preds = []
    notes = spec.notes()
    # group notes by the segment they start in, so a two-note chord can share out wrong pitches
    groups: dict[int, list] = {}
    for n in notes:
        seg = _segment_for_note(segments, n)
        groups.setdefault(seg.index, []).append(n)
    for seg_idx, group in groups.items():
        r = by_index[seg_idx]
        onset = None if r.start_frame < 0 else r.start_frame * seconds_per_frame + offset_s
        substituted = dict(r.substitutes)          # expected pitch -> pitch played instead
        for n in group:
            note_flags = flags.get(n.id, frozenset())
            if n.pitch in r.detected_pitches:
                preds.append(NotePrediction(n.id, "correct", n.pitch, onset, note_flags))
            elif n.pitch in substituted:
                preds.append(NotePrediction(n.id, "wrong_pitch", substituted[n.pitch], onset, note_flags))
            else:
                preds.append(NotePrediction(n.id, "missed", None, onset))
    return preds


class V2Adapter:
    name = "v2"
    capabilities = frozenset({"wrong_pitch", "missed", "held_long", "extra"})

    def config(self) -> dict:
        from stft_hmm_v2 import TimingConfig
        return asdict(TimingConfig())

    def predict(self, audio_path: str, piece: PieceSpec, out_dir: Path) -> TakePrediction:
        import stft_hmm_v2
        out_dir.mkdir(parents=True, exist_ok=True)
        plot = out_dir / "alignment.png"
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            bundle, results, insertions = stft_hmm_v2.run_pipeline(
                str(audio_path), piece.name, str(plot))
        (out_dir / "model_stdout.txt").write_text(log.getvalue())
        cfg = stft_hmm_v2.TimingConfig()
        fpb, _, _ = stft_hmm_v2.recording_tempo(bundle, cfg)
        spf = bundle.audio.seconds_per_frame
        offset = bundle.audio.trim_offset_seconds
        notes = segments_to_note_predictions(
            results, bundle.segments, piece, spf, offset,
            candidate_scores=bundle.candidate_scores, interior_fpb=fpb, cfg=cfg)
        extras = extras_from_insertions(
            insertions, results, bundle.segments, piece, spf, offset,
            bundle.candidate_onset_scores)
        return TakePrediction(notes=notes, extras=extras, capabilities=self.capabilities,
                              artifacts={"plot": str(plot), "stdout": str(out_dir / "model_stdout.txt")})


ADAPTERS = {
    "v2": V2Adapter,
}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(ADAPTERS)}")
    return ADAPTERS[name]()
