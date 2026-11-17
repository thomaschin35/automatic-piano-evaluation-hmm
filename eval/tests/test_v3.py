"""v3 behaviour on a synthetic perfect performance and edits of it (no audio needed)."""
from __future__ import annotations
import dataclasses
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
        waiting is `late` on the next note, never held_long."""
        for c, pred, ev in self.run_kind("hold"):
            self.assertEqual((ev.counts["late.tp"], ev.counts["late.fp"], ev.counts["late.fn"]), (1, 0, 0), str(c.meta))
            self.assertFalse(any("held_long" in n.flags for n in pred.notes), str(c.meta))
        self.assertIn("held_long", get_adapter("v3").capabilities)


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


def ringing_bundle(spec, release_db_per_frame=3.0):
    """fake_bundle with a spectrum: each key sounds its first six harmonics while it is down and,
    once released, is damped by release_db_per_frame (a piano key falls fast when let go)."""
    import numpy as np
    import stft_features as sf
    from eval.cache import rescore
    b = fake_bundle(spec, frames_per_note=20, in_rhythm=True)
    T, n_bins = b.audio.num_frames, 2049
    db = np.full((n_bins, T), -80.0, dtype=np.float32)
    frames = np.arange(T)
    for n in spec.notes():
        on = int(round(20 * (n.start_beat - spec.notes()[0].start_beat)))
        off = int(round(20 * (n.end_beat - spec.notes()[0].start_beat)))
        lvl = np.where(frames < on, -80.0, np.where(frames < off, -10.0 - 0.3 * (frames - on),
                                                     -10.0 - 0.3 * (off - on) - release_db_per_frame * (frames - off)))
        f0 = sf.midi_to_frequency(n.pitch) * (n_bins - 1) * 2 / b.audio.sample_rate
        for h in range(1, 7):
            k = int(round(h * f0))
            if k + 1 < n_bins:
                db[k - 1:k + 2] = np.maximum(db[k - 1:k + 2], np.maximum(lvl - 3.0 * h, -80.0))
    b.audio = dataclasses.replace(b.audio, stft_db=db)
    return rescore(b, spec.to_score_model())


class HeldTooLong(unittest.TestCase):
    """held_long = the key was not released when the next note came on time; it goes on the held note."""

    @classmethod
    def setUpClass(cls):
        cls.spec = load_piece("ode_melody")
        cls.bundle = ringing_bundle(cls.spec)
        cls.adapter = get_adapter("v3")
        cls.cases = list(synth.generate("fake", cls.bundle, cls.spec, ("clean", "overlap", "hold")))

    def held(self, pred):
        return [n.note_id for n in pred.notes if "held_long" in n.flags]

    def test_clean_performance_has_no_held_note(self):
        c = next(c for c in self.cases if c.kind == "clean")
        self.assertEqual(self.held(self.adapter.predict_bundle(c.bundle, c.spec)), [])

    def test_key_sounding_under_the_next_note_is_held_long(self):
        cases = [c for c in self.cases if c.kind == "overlap"]
        self.assertTrue(cases)
        found = 0
        for c in cases:
            pred = self.adapter.predict_bundle(c.bundle, c.spec)
            got = self.held(pred)
            self.assertLessEqual(set(got), {c.gold.errors[0].note_id}, str(c.meta))     # never on another note
            self.assertFalse(any("late" in n.flags for n in pred.notes), str(c.meta))
            found += got == [c.gold.errors[0].note_id]
        self.assertGreaterEqual(found, len(cases) - 2)       # a next note shorter than the wait is not judged

    def test_late_next_note_is_late_not_held_long(self):
        for c in (c for c in self.cases if c.kind == "hold"):
            pred = self.adapter.predict_bundle(c.bundle, c.spec)
            self.assertEqual(self.held(pred), [], str(c.meta))
            self.assertIn(c.gold.errors[0].note_id, [n.note_id for n in pred.notes if "late" in n.flags], str(c.meta))
