"""Decoder tests for the monophonic HMM.

The current chain, before skip / repeat / duration grading, did this on the
same synthetic bundles:

  tied repeated pitch     both notes correct
  skipped scale step      wrong_pitch (heard the next note)
  staccato quarter        missed
  legato substitution     missed
  one long note           pitches correct, no held_long

These tests fail if a change walks any of that back, and they keep a clean
scale and a uniformly slow scale correct with no duration flags.
"""
from __future__ import annotations
import unittest
from types import SimpleNamespace

import numpy as np

from score_models import ScoreSegment, slice_score_to_segments
from stft_hmm import AlignedSegment, SegmentResult, name_substitutes
from stft_hmm_v2 import (
    TimingConfig, _drop_final_hold, _fit_durations, align_bundle, build_graph,
    log_emissions, path_to_intervals, recording_tempo,
)
from eval.adapters import (
    duration_flags, extras_from_insertions, get_adapter, segments_to_note_predictions,
)
from eval.labels import GoldError, TakeLabels
from eval.metrics import MONOPHONIC_PIECES, evaluate_take, monophonic_summary, summarize
from eval.pieces import PieceSpec, load_piece
from eval.tests.test_eval import all_correct
from eval.contract import NotePrediction


SPF = 0.023


def _segment(index: int, pitch: int, beats: float = 1.0, start: float = 0.0) -> ScoreSegment:
    return ScoreSegment(index, start, start + beats, pitch, 0, (pitch,) if pitch else ())


def _aligned(index: int, pitch: int, frames: int = 12) -> AlignedSegment:
    return AlignedSegment(_segment(index, pitch, start=float(index)), index * frames, (index + 1) * frames - 1)


def _bundle(pitches, onset_at, regions, T, energy=0.7):
    used = set(pitches) | set(onset_at) | set(regions)
    scores = {p: np.zeros(T) for p in used}
    onsets = {p: np.zeros(T) for p in used}
    for pitch, (a, b) in regions.items():
        scores[pitch][a:b] = 0.9
    for pitch, frames in onset_at.items():
        for frame in frames:
            onsets.setdefault(pitch, np.zeros(T))
            onsets[pitch][frame] = 1.0
    segments = [_segment(i, p, start=float(i)) for i, p in enumerate(pitches)]
    audio = SimpleNamespace(num_frames=T, seconds_per_frame=SPF, trim_offset_seconds=0.0)
    return SimpleNamespace(
        audio=audio,
        score=SimpleNamespace(total_beats=float(len(pitches))),
        segments=segments,
        candidate_scores=scores,
        candidate_onset_scores=onsets,
        note_scores=scores,
        note_onset_scores=onsets,
        frame_energy=np.full(T, energy),
    )


def _spec(pitches) -> PieceSpec:
    return PieceSpec(
        name="probe", title="probe", time_signature=(4, 4), bpm=120,
        pickup_beats=0.0, hands={"RH": [(p, 1.0) for p in pitches]},
    )


class TestGraph(unittest.TestCase):
    def test_skip_insert_and_tail_leave_the_last_note(self):
        alignment = [_aligned(0, 60), _aligned(1, 60), _aligned(2, 64)]
        cfg = TimingConfig()
        states, A = build_graph(alignment, cfg)
        self.assertEqual(states[0].kind, "lead")
        self.assertEqual(states[-1].kind, "tail")
        self.assertTrue(np.allclose(A.sum(axis=1), 1.0))

        skip0 = next(i for i, s in enumerate(states) if s.kind == "skip")
        att0 = next(i for i, s in enumerate(states) if s.kind == "att")
        self.assertAlmostEqual(A[0, skip0], cfg.skip_prob)
        self.assertAlmostEqual(A[0, att0], 1.0 - cfg.lead_in_self_loop - cfg.skip_prob)

        frees = [i for i, s in enumerate(states) if s.kind == "free"]
        self.assertLess(A[frees[-1], frees[-1]], 1.0)
        self.assertGreater(A[frees[-1], len(states) - 1], 0.0)
        self.assertGreater(A[frees[0], next(i for i, s in enumerate(states) if s.kind == "ins")], 0.0)

        gated = [s for s in states if s.repeat_gate]
        self.assertEqual([s.seg for s in gated], [1])

    def test_repeated_attack_is_penalized_without_an_onset(self):
        alignment = [_aligned(0, 64), _aligned(1, 64)]
        cfg = TimingConfig()
        states, _ = build_graph(alignment, cfg)
        bundle = _bundle([64, 64], {}, {64: (0, 24)}, T=24)
        emissions = log_emissions(states, bundle, cfg, alignment)
        gate = next(i for i, s in enumerate(states) if s.repeat_gate)
        other = next(i for i, s in enumerate(states) if s.kind == "att" and not s.repeat_gate)
        self.assertAlmostEqual(emissions[other, 10] - emissions[gate, 10], cfg.repeat_missing_penalty)

    def test_skip_is_empty_and_lead_is_not_note_one(self):
        alignment = [_aligned(i, p) for i, p in enumerate((60, 62, 64))]
        states, _ = build_graph(alignment, TimingConfig())
        T = 20
        path = np.zeros(T, dtype=int)
        lead = 0
        free0 = next(i for i, s in enumerate(states) if s.seg == 0 and s.kind == "free")
        skip1 = next(i for i, s in enumerate(states) if s.seg == 1 and s.kind == "skip")
        free2 = next(i for i, s in enumerate(states) if s.seg == 2 and s.kind == "free")
        ins0 = next(i for i, s in enumerate(states) if s.seg == 0 and s.kind == "ins")
        tail = len(states) - 1
        path[0] = lead
        path[1:6] = free0
        path[6:9] = ins0
        path[9] = skip1
        path[10:16] = free2
        path[16:] = tail
        intervals, insertions = path_to_intervals(path, states, 3, T)
        self.assertEqual(intervals[0], (1, 5))
        self.assertIsNone(intervals[1])
        self.assertEqual(intervals[2], (10, 15))
        self.assertEqual(insertions, [(0, 6, 8)])

    def test_duration_prior_does_not_eat_the_ending(self):
        fitted = _fit_durations([9, 9, 9, 9], 14)
        self.assertEqual(sum(fitted), 14)
        self.assertGreaterEqual(fitted[-1], fitted[0])

    def test_final_hold_does_not_slow_the_file(self):
        raw = np.array([10.0, 10.0, 10.0, 40.0])
        self.assertEqual(list(_drop_final_hold(raw)), [10.0, 10.0, 10.0])
        T = 90
        pitches = [60, 62, 64, 65]
        onsets = {}
        regions = {}
        for i, pitch in enumerate(pitches):
            frame = 5 + i * 10
            onsets[pitch] = [frame]
            regions[pitch] = (frame, frame + 8)
        # The last peak is 40 frames later than an even beat.
        onsets[65] = [5 + 3 * 10 + 30]
        bundle = _bundle(pitches, onsets, regions, T)
        fpb, curve, info = recording_tempo(bundle, TimingConfig())
        self.assertEqual(info["source"], "pitch_gated_ioi")
        self.assertEqual(info["dropped_final_hold"], 1)
        self.assertLess(float(curve[-1]), 16.0)
        self.assertLess(fpb, 16.0)


class TestGrading(unittest.TestCase):
    def test_staccato_is_present_where_the_old_fraction_missed_it(self):
        T = 30
        bundle = _bundle([60], {60: [2]}, {60: (2, 7)}, T, energy=0.15)
        bundle.frame_energy[2:8] = 0.8
        results, insertions, _, _ = align_bundle(bundle)
        self.assertEqual(results[0].detected_pitches, (60,))
        self.assertEqual(results[0].status, "correct")
        self.assertEqual(insertions, [])
        # The rule this replaced: 40% of the frames after the attack, across the whole gap.
        body = bundle.candidate_scores[60][2:]
        old_fraction = float(np.mean(body >= 0.40))
        self.assertLess(old_fraction, 0.40)

    def test_legato_substitution_and_octave_partial(self):
        T = 24
        segment = _segment(0, 67)
        audio = SimpleNamespace(num_frames=T)
        scores = {67: np.zeros(T), 69: np.full(T, 0.85), 60: np.full(T, 0.9), 48: np.full(T, 0.95)}
        onsets = {p: np.zeros(T) for p in scores}
        bundle = SimpleNamespace(audio=audio, candidate_scores=scores, candidate_onset_scores=onsets)
        body = slice(3, 8)
        named = name_substitutes(
            bundle, 0, T - 1, body, segment, (), (),
            onset_threshold=0.5, attack_fraction=0.4, presence_threshold=0.4,
            previous_detected=())
        self.assertEqual(named, ((67, 69),))
        # Without the new argument the ringing pitch is not a substitute (v1).
        self.assertEqual(name_substitutes(
            bundle, 0, T - 1, body, segment, (), (),
            onset_threshold=0.5, attack_fraction=0.4, presence_threshold=0.4,
        ), ())
        # C4 sitting on a louder C3 is a partial, not a substitute for D4.
        partial = name_substitutes(
            bundle, 0, T - 1, body, _segment(0, 62), (48,), (),
            onset_threshold=0.5, attack_fraction=0.4, presence_threshold=0.4,
            previous_detected=())
        self.assertNotIn(60, {q for _, q in partial})

    def test_tied_repeat_is_a_miss_and_a_long_note(self):
        T = 40
        bundle = _bundle([64, 64], {64: [2]}, {64: (2, 38)}, T, energy=0.05)
        bundle.frame_energy[2:38] = 0.8
        results, _, info, _ = align_bundle(bundle)
        self.assertEqual(results[0].status, "correct", info)
        self.assertEqual(results[1].status, "skipped")
        self.assertEqual(results[1].substitutes, ())
        spec = _spec([64, 64])
        flags = duration_flags(
            results, bundle.segments, spec, bundle.candidate_scores, info.get("median_ioi_frames", T / 2),
            TimingConfig())
        # Interior tempo may fall back to a file stretch; the held note is still the long one.
        self.assertIn("RH1", flags)
        self.assertNotIn("RH2", flags)

    def test_wrong_pitch_on_the_beat_stays_a_substitution(self):
        T = 36
        bundle = _bundle([60, 62, 64], {60: [1], 65: [13], 64: [25]},
                         {60: (1, 12), 65: (13, 24), 64: (25, 35)}, T, energy=0.05)
        bundle.frame_energy[1:35] = 0.8
        results, insertions, _, _ = align_bundle(bundle)
        self.assertEqual([r.status for r in results], ["correct", "substitution", "correct"])
        self.assertEqual(results[1].substitutes, ((62, 65),))
        self.assertEqual(insertions, [])

    def test_legato_wrong_note_is_named_by_the_decoder(self):
        T = 24
        bundle = _bundle([60], {60: [2]}, {}, T, energy=0.05)
        bundle.frame_energy[2:22] = 0.8
        bundle.candidate_scores[60][:] = 0.05
        bundle.candidate_scores[62] = np.full(T, 0.85)
        bundle.candidate_onset_scores[62] = np.zeros(T)
        results, _, _, _ = align_bundle(bundle)
        self.assertEqual(results[0].status, "substitution")
        self.assertEqual(results[0].substitutes, ((60, 62),))

    def test_skipped_step_is_a_miss_not_a_wrong_pitch(self):
        T = 36
        bundle = _bundle([60, 62, 64], {60: [2], 64: [8]}, {60: (2, 8), 64: (8, 34)}, T, energy=0.05)
        bundle.frame_energy[2:34] = 0.8
        results, _, _, _ = align_bundle(bundle)
        self.assertEqual(results[0].status, "correct")
        self.assertEqual(results[1].status, "skipped")
        self.assertEqual(results[1].substitutes, ())
        self.assertEqual(results[2].detected_pitches, (64,))

    def test_clean_and_slow_scales_stay_clean(self):
        for hop in (12, 20):
            pitches = [60, 62, 64, 65]
            onsets, regions = {}, {}
            for i, pitch in enumerate(pitches):
                start = i * hop + 1
                onsets[pitch] = [start]
                regions[pitch] = (start, start + hop - 2)
            bundle = _bundle(pitches, onsets, regions, hop * len(pitches))
            results, insertions, info, _ = align_bundle(bundle)
            self.assertEqual([r.status for r in results], ["correct"] * 4, hop)
            self.assertEqual(insertions, [])
            fpb = info["median_ioi_frames"] if info["source"] == "pitch_gated_ioi" else bundle.audio.num_frames / 4
            flags = duration_flags(
                results, bundle.segments, _spec(pitches), bundle.candidate_scores, fpb, TimingConfig())
            self.assertEqual(flags, {}, hop)

    def test_one_long_note_among_short_neighbors(self):
        pitches = [60, 62, 64, 65]
        spec = _spec(pitches)
        segments = slice_score_to_segments(spec.to_score_model())
        results = [
            SegmentResult(i, (p,), start, start + 8, 0, 0, (p,), False)
            for i, (p, start) in enumerate(zip(pitches, (0, 10, 28, 36)))
        ]
        scores = {p: np.zeros(48) for p in pitches}
        scores[65][36:46] = 0.9
        flags = duration_flags(results, segments, spec, scores, interior_fpb=10.0, cfg=TimingConfig())
        self.assertEqual(flags, {"RH2": frozenset({"held_long"})})

        # Every note stretched the same amount is a tempo change, not four long notes.
        even = [
            SegmentResult(i, (p,), i * 20, i * 20 + 18, 0, 0, (p,), False)
            for i, p in enumerate(pitches)
        ]
        self.assertEqual(duration_flags(even, segments, spec, scores, 10.0, TimingConfig()), {})

    def test_skip_maps_to_missed_and_insertion_maps_to_extra(self):
        pitches = [60, 62]
        spec = _spec(pitches)
        segments = slice_score_to_segments(spec.to_score_model())
        results = [
            SegmentResult(0, (60,), 0, 10, 0, 0, (60,), False),
            SegmentResult(1, (62,), -1, -1, 0, 0, (), False),
        ]
        preds = {p.note_id: p for p in segments_to_note_predictions(results, segments, spec, 0.1, 0.0)}
        self.assertEqual(preds["RH1"].label, "correct")
        self.assertEqual(preds["RH2"].label, "missed")
        self.assertIsNone(preds["RH2"].onset_time_s)
        onsets = {66: np.zeros(20)}
        onsets[66][12:16] = 1.0
        extras = extras_from_insertions([(0, 12, 15)], results, segments, spec, 0.1, 0.5, onsets)
        self.assertEqual(extras[0].after_note_id, "RH1")
        self.assertEqual(extras[0].pitch, 66)
        self.assertAlmostEqual(extras[0].time_s, 1.7)


class TestMonophonicReport(unittest.TestCase):
    def test_mean_uses_only_the_monophonic_pieces(self):
        spec = load_piece("c_major_rh")
        canon = load_piece("canon_lh")
        gold = TakeLabels("c_major_rh", False, [GoldError("missed", "RH2")])
        pred = all_correct(spec, RH2=NotePrediction("RH2", "missed"))
        mono = evaluate_take("m", spec, gold, pred)
        other = evaluate_take("c", canon, TakeLabels("canon_lh", True), None)
        full = summarize([mono, other])
        report = monophonic_summary(full, mono.rows + other.rows)
        self.assertEqual(tuple(report["pieces"]), MONOPHONIC_PIECES)
        self.assertEqual(set(report["by_piece"]), {"c_major_rh"})
        self.assertEqual(report["class_f1"]["missed"], 1.0)
        self.assertEqual(report["mean_class_f1"], 1.0)
        self.assertEqual(get_adapter("v2").capabilities, frozenset({"wrong_pitch", "missed", "held_long", "extra"}))


if __name__ == "__main__":
    unittest.main()
