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
