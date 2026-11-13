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
