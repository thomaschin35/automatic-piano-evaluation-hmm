"""Unit tests for the feature cache helpers and synthetic-case generators (no audio needed)."""
from __future__ import annotations
import unittest

import numpy as np

import stft_features as sf
from eval import synth
from eval.cache import rescore, splice
from eval.contract import NotePrediction, TakePrediction
from eval.metrics import evaluate_take
from eval.labels import GoldError, TakeLabels
from eval.pieces import load_piece


def fake_bundle(spec, frames_per_note=10):
    """A perfect performance: each note's pitch sounds for `frames_per_note` frames, onset at its start."""
    notes = spec.notes()
    T = frames_per_note * len(notes)
    pres = {p: np.zeros(T) for p in range(28, 97)}
    ons = {p: np.zeros(T) for p in range(28, 97)}
    for i, n in enumerate(notes):
        pres[n.pitch][i * frames_per_note:(i + 1) * frames_per_note] = 0.9
        ons[n.pitch][i * frames_per_note + synth.PRE] = 1.0
    audio = sf.AudioData(waveform=np.zeros(0), sample_rate=22050, hop_length=512, trim_start_sample=0,
                         trim_end_sample=0, stft_magnitude=np.zeros((0, 0)), stft_db=np.zeros((4, T)),
                         onset_stft_db=np.zeros((0, 0)))
    b = sf.Stage1FeatureBundle(audio=audio, score=None, segments=[], candidate_scores=pres,
                               candidate_onset_scores=ons, note_scores={}, note_onset_scores={},
                               unexpected_pitch_scores=np.zeros(T), broadband_onset=np.zeros(T),
                               frame_energy=np.linspace(1.0, 0.5, T), observations=np.zeros((T, 3)), tracked_pitches=())
    return rescore(b, spec.to_score_model())


class CacheHelpers(unittest.TestCase):
    def setUp(self):
        self.spec = load_piece("c_major_rh")
        self.b = fake_bundle(self.spec)

    def test_rescore_tracks_score_pitches(self):
        self.assertEqual(sorted(self.b.note_scores), [60, 62, 64, 65, 67, 69, 71, 72])
        self.assertEqual(len(self.b.segments), 8)

    def test_splice_reindexes_time(self):
        s = splice(self.b, np.arange(10, 80))
        self.assertEqual(s.audio.num_frames, 70)
        self.assertEqual(s.candidate_scores[62][0], 0.9)
        self.assertEqual(self.b.audio.num_frames, 80)          # original untouched

    def test_reference_onsets(self):
        on = synth.reference_onsets(self.b, self.spec)
        self.assertEqual(list(on), [i * 10 + synth.PRE for i in range(8)])


class Generators(unittest.TestCase):
    def setUp(self):
        self.spec = load_piece("ode_melody")
        self.b = fake_bundle(self.spec)
        self.cases = list(synth.generate("fake", self.b, self.spec))

    def kind(self, k):
        return [c for c in self.cases if c.kind == k]

    def test_every_kind_is_generated(self):
        self.assertEqual({c.kind for c in self.cases}, set(synth.KINDS))

    def test_sub_gold(self):
        c = self.kind("sub")[0]
        e = c.gold.errors[0]
        self.assertEqual((e.type, e.note_id, e.played), ("wrong_pitch", "RH1", 64))
        self.assertNotEqual(c.spec.notes()[0].pitch, 64)

    def test_miss_del_removes_the_note(self):
        c = next(c for c in self.kind("miss_del") if c.meta["note"] == 2)      # F4, the third note
        self.assertEqual(c.bundle.audio.num_frames, self.b.audio.num_frames - 10)
        self.assertEqual(float(c.bundle.candidate_scores[65][:40].max()), 0.0)
        self.assertEqual(c.gold.errors[0].note_id, "RH3")

    def test_extra_dup_and_restart_lengths(self):
        self.assertTrue(all(c.bundle.audio.num_frames == self.b.audio.num_frames + 10 for c in self.kind("extra_dup")))
        c = next(c for c in self.kind("restart") if c.meta["stop"] == 6 and c.meta["resume"] == 4)
        self.assertEqual(c.bundle.audio.num_frames, self.b.audio.num_frames + 20)
        self.assertEqual(c.meta["restart"], ("RH6", "RH5"))
        self.assertTrue(all(c.meta["stop"] - c.meta["resume"] >= 2 for c in self.kind("restart")))

    def test_stop_early_marks_the_rest_missed(self):
        c = self.kind("stop_early")[-1]
        self.assertEqual([e.note_id for e in c.gold.errors], ["RH14", "RH15"])


class MissedInRuns(unittest.TestCase):
    def test_missed_position_inside_a_repeated_run_is_not_penalised(self):
        spec = load_piece("ode_melody")                       # RH1, RH2 are both E4
        gold = TakeLabels("ode_melody", False, [GoldError("missed", note_id="RH2")])
        pred = TakePrediction(notes=[NotePrediction(n.id, "missed" if n.id == "RH1" else "correct", n.pitch)
                                     for n in spec.notes()])
        ev = evaluate_take("t", spec, gold, pred)
        self.assertEqual((ev.counts["strict.tp"], ev.counts["strict.fp"], ev.counts["strict.fn"]), (1, 0, 0))

    def test_no_move_across_different_pitches(self):
        spec = load_piece("ode_melody")                       # RH3 is F4
        gold = TakeLabels("ode_melody", False, [GoldError("missed", note_id="RH2")])
        pred = TakePrediction(notes=[NotePrediction(n.id, "missed" if n.id == "RH3" else "correct", n.pitch)
                                     for n in spec.notes()])
        ev = evaluate_take("t", spec, gold, pred)
        self.assertEqual((ev.counts["strict.tp"], ev.counts["strict.fp"], ev.counts["strict.fn"]), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
