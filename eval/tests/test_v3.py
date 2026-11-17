"""v3 behaviour on a synthetic perfect performance and edits of it (no audio needed)."""
from __future__ import annotations
import unittest

from eval import synth
from eval.adapters import get_adapter
from eval.metrics import evaluate_take
from eval.pieces import load_piece
from eval.tests.test_synth import fake_bundle


class V3OnFakePerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_piece("ode_melody")
        cls.bundle = fake_bundle(cls.spec, frames_per_note=20, in_rhythm=True)
        cls.adapter = get_adapter("v3")
        cls.cases = list(synth.generate("fake", cls.bundle, cls.spec))

    def run_kind(self, kind):
        out = []
        for c in self.cases:
            if c.kind == kind:
                pred = self.adapter.predict_bundle(c.bundle, c.spec)
                out.append((c, pred, evaluate_take("t", c.spec, c.gold, pred)))
        return out

    def assert_all_exact(self, kind, keys=("strict", "extra")):
        for c, pred, ev in self.run_kind(kind):
            bad = {f"{k}.{x}": ev.counts[f"{k}.{x}"] for k in keys for x in ("fp", "fn") if ev.counts[f"{k}.{x}"]}
            self.assertEqual(bad, {}, f"{kind} {c.meta}")

    def test_clean_take_has_no_flags(self):
        (c, pred, ev), = self.run_kind("clean")
        self.assertEqual(ev.pred_errors, 0)
        self.assertEqual(pred.restarts, [])

    def test_wrong_pitch_is_named(self):
        for c, pred, ev in self.run_kind("sub"):
            n = pred.notes[c.meta["note"]]
            self.assertEqual((n.label, n.played_pitch), ("wrong_pitch", c.gold.errors[0].played), str(c.meta))

    def test_missed_notes(self):
        self.assert_all_exact("miss_del")
        self.assert_all_exact("miss_ins_repeat")

    def test_extra_notes(self):
        self.assert_all_exact("extra_dup")

    def test_stopping_early_marks_the_rest_missed(self):
        self.assert_all_exact("stop_early")

    def test_restart_is_located_and_not_penalised(self):
        for c, pred, ev in self.run_kind("restart"):
            self.assertIn(c.meta["restart"], [tuple(r) for r in pred.restarts], str(c.meta))
            self.assertEqual(ev.pred_errors, 0, str(c.meta))

    def test_mistake_before_a_restart_is_kept(self):
        """A wrong key, then going back: the restart is located and the wrong key is reported as a
        mistake of attempt 1, while the replayed (final) notes are all correct."""
        cases = self.run_kind("restart_wrong")
        self.assertTrue(cases)
        for c, pred, ev in cases:
            self.assertIn(c.meta["restart"], [tuple(r) for r in pred.restarts], str(c.meta))
            g = c.gold.earlier[0]
            self.assertIn((1, g.note_id, "wrong_pitch", g.played),
                          [(e.attempt, e.note_id, e.label, e.played_pitch) for e in pred.earlier], str(c.meta))
            self.assertEqual([n.note_id for n in pred.notes if n.label != "correct"], [], str(c.meta))

    def test_clean_restart_reports_no_earlier_mistake(self):
        for c, pred, ev in self.run_kind("restart"):
            self.assertEqual(pred.earlier, [], str(c.meta))

    def test_silent_gap_is_late(self):
        for c, pred, ev in self.run_kind("gap"):
            self.assertEqual((ev.counts["late.tp"], ev.counts["late.fp"], ev.counts["late.fn"]), (1, 0, 0), str(c.meta))

    def test_held_gap_is_late_too_and_never_held_long(self):
        """Late = the note started late, whatever the reason. Keeping the previous key down and
        waiting is `late` on the next note; held_long (overlap) is not something v3 emits."""
        for c, pred, ev in self.run_kind("hold"):
            self.assertEqual((ev.counts["late.tp"], ev.counts["late.fp"], ev.counts["late.fn"]), (1, 0, 0), str(c.meta))
            self.assertFalse(any("held_long" in n.flags for n in pred.notes), str(c.meta))
        self.assertNotIn("held_long", get_adapter("v3").capabilities)


if __name__ == "__main__":
    unittest.main()


class RelativeStrikeLevels(unittest.TestCase):
    """Strike strength is judged against that key's level in the recording, not a fixed number."""

    @staticmethod
    def events(levels):
        import stft_hmm_v3 as m
        return m, [m.Event(10 * i, {p: s}, [(p, s)], full={p}) for i, (p, s) in enumerate(levels)]

    def test_quiet_key_is_raised_and_loud_key_is_left_alone(self):
        m, evs = self.events([(60, 0.75)] * 3 + [(76, 0.29), (75, 0.30), (76, 0.31)])
        cfg = m.V3Config()
        s = m.key_scales(evs, cfg)
        self.assertEqual(s[60], 1.0)
        self.assertGreaterEqual(0.29 * s[76], cfg.sub_min_salience)      # now a clear strike of that key

    def test_one_faint_strike_cannot_set_its_own_standard(self):
        m, evs = self.events([(71, 0.70)] * 3 + [(69, 0.70)] * 2 + [(72, 0.20)])
        cfg = m.V3Config()
        self.assertLess(0.20 * m.key_scales(evs, cfg)[72], cfg.match_min_salience)

    def test_nothing_is_struck_in_silence(self):
        import copy
        import stft_hmm_v3 as m
        spec = load_piece("ode_melody")
        bundle = copy.copy(fake_bundle(spec, frames_per_note=20, in_rhythm=True))
        cfg = m.V3Config()
        n = len(m.detect_events(bundle, cfg))
        self.assertGreater(n, 0)
        bundle.frame_energy = bundle.frame_energy * 0.0
        self.assertEqual(m.detect_events(bundle, cfg), [])


class NewPieceShapes(unittest.TestCase):
    """Perfectly heard notes of pieces v3 was never tuned on: rests, eighth notes, other registers."""

    def test_clean_performance_of_each_round3_piece_has_no_flags(self):
        adapter = get_adapter("v3")
        for name in ("twinkle", "buns", "frere", "c_low", "c_high", "rests"):
            spec = load_piece(name)
            pred = adapter.predict_bundle(fake_bundle(spec, frames_per_note=20, in_rhythm=True), spec)
            bad = [(n.note_id, n.label, sorted(n.flags)) for n in pred.notes if n.label != "correct" or n.flags]
            self.assertEqual((bad, pred.extras, pred.restarts), ([], [], []), name)


class WindowedAlignment(unittest.TestCase):
    """align() searches only the moves a player makes; it must give exactly the exhaustive answer."""

    def test_matches_exhaustive_search_on_a_long_piece_with_restarts(self):
        import dataclasses
        import numpy as np
        import stft_hmm_v3 as m
        spec = load_piece("ode_melody")
        bundle = fake_bundle(spec, frames_per_note=20, in_rhythm=True)
        cfg = dataclasses.replace(m.V3Config(), max_skip=10_000)       # forward window covers everything
        ev = m.detect_events(bundle, cfg)
        pos = m.score_positions(bundle)
        span = ev[-1].frame + 20
        reps = 6
        # the piece played 6 times in a row = one long piece; each new copy reads as a restart
        E = [dataclasses.replace(e, frame=e.frame + k * span) for k in range(reps) for e in ev]
        P = [m.Position(k * len(pos) + q.index, q.beat + k * 16, q.pitches, q.segment_index, q.bar_start)
             for k in range(reps) for q in pos]
        spb = np.full(len(E), 0.5)
        spf = bundle.audio.seconds_per_frame
        got, cost = m.align(E, P, cfg, spb, spf, m.back_costs(P, cfg))
        dense = np.where(m.back_window(P, cfg), m.back_costs(P, cfg)[None, :], np.inf)
        want, want_cost = m.align_dense(E, P, cfg, spb, spf, dense)
        self.assertEqual([(x.event, x.position, x.kind) for x in got], [(x.event, x.position, x.kind) for x in want])
        self.assertAlmostEqual(cost, want_cost, places=9)


class SecondKeyAndStrayKey(unittest.TestCase):
    """Extras that are not a separate event: a key struck together with the right note, and a key
    pressed well before the note it would otherwise be read as 'corrected into'."""

    def setUp(self):
        import copy
        self.spec = load_piece("ode_melody")
        self.bundle = copy.deepcopy(fake_bundle(self.spec, frames_per_note=20, in_rhythm=True))
        self.adapter = get_adapter("v3")

    def test_key_struck_together_with_the_note_is_an_extra(self):
        b, f = self.bundle, 3 * 20                          # RH4 (G4) starts at frame 60
        b.candidate_scores[69][f:f + 20] = 0.6              # A4 brushed with it
        b.candidate_onset_scores[69][f + synth.PRE] = 1.0
        pred = self.adapter.predict_bundle(b, self.spec)
        self.assertEqual([n.label for n in pred.notes], ["correct"] * 15)
        self.assertEqual([x.pitch for x in pred.extras], [69])

    def test_key_pressed_long_before_the_first_note_is_an_extra_not_a_correction(self):
        from eval.cache import splice
        import numpy as np
        b = self.bundle
        T = b.audio.num_frames
        b2 = splice(b, np.concatenate([np.arange(0, 20), np.arange(0, T)]))    # the first note's slot twice
        for p in (64, 65):                                  # the first copy sounds F4, then silence, not E4
            b2.candidate_scores[p] = b2.candidate_scores[p].copy()
            b2.candidate_onset_scores[p] = b2.candidate_onset_scores[p].copy()
        b2.candidate_scores[65][0:8], b2.candidate_scores[64][0:20] = 0.9, 0.0
        b2.candidate_onset_scores[65][synth.PRE], b2.candidate_onset_scores[64][0:20] = 1.0, 0.0
        b3 = splice(b2, np.concatenate([np.arange(0, 20), np.full(40, 19), np.arange(20, b2.audio.num_frames)]))
        pred = self.adapter.predict_bundle(b3, self.spec)
        self.assertEqual(pred.notes[0].label, "correct")
        self.assertEqual([x.pitch for x in pred.extras], [65])
