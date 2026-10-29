"""Unit tests for the eval harness: python -m unittest discover eval/tests"""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest

import score_models
from score_models import slice_score_to_segments
from stft_hmm import SegmentResult

from eval.adapters import segments_to_note_predictions
from eval.contract import ExtraNotePrediction, NotePrediction, TakePrediction
from eval.labels import GoldError, TakeLabels, load_labels
from eval.metrics import evaluate_take, summarize
from eval.pieces import all_pieces, load_piece, parse_pitch


def write_tmp(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


def all_correct(spec, **overrides) -> TakePrediction:
    notes = []
    for n in spec.notes():
        notes.append(overrides.get(n.id, NotePrediction(n.id, "correct", n.pitch)))
    return TakePrediction(notes=notes)


class TestPieces(unittest.TestCase):
    def test_parse_pitch(self):
        self.assertEqual(parse_pitch("C4"), 60)
        self.assertEqual(parse_pitch("F#4"), 66)
        self.assertEqual(parse_pitch("Db5"), 73)
        self.assertEqual(parse_pitch("rest"), 0)
        with self.assertRaises(ValueError):
            parse_pitch("H4")

    def test_all_specs_load(self):
        pieces = all_pieces()
        self.assertEqual(set(pieces), {"c_major_rh", "c_major_both", "e_major_rh", "e_major_rh_down", "e_major_both",
                                       "ode_melody", "ode_harmony", "canon_lh", "canon_both"})

    def test_note_ids_and_measures(self):
        spec = load_piece("ode_harmony")
        ids = spec.note_by_id()
        self.assertEqual(len([n for n in ids if n.startswith("RH")]), 15)
        self.assertEqual(len([n for n in ids if n.startswith("LH")]), 4)
        self.assertEqual((ids["RH14"].measure, ids["RH14"].beat_in_measure), (4, 2.5))
        self.assertEqual((ids["LH2"].pitch, ids["LH2"].measure), (55, 2))

    def test_both_hands_parallel_octave(self):
        for name in ("c_major_both", "e_major_both"):
            ids = load_piece(name).note_by_id()
            for i in range(1, 9):
                self.assertEqual(ids[f"RH{i}"].pitch - ids[f"LH{i}"].pitch, 12)

    def test_ode_matches_score_models(self):
        for spec_name, model_name in (("ode_harmony", "ode_to_joy_with_harmony"),
                                      ("ode_melody", "ode_to_joy_without_harmony")):
            ours = load_piece(spec_name).to_score_model()
            ref = score_models.SCORE_MODELS[model_name]()
            self.assertEqual(ours.melody, ref.melody)
            self.assertEqual(ours.harmony, ref.harmony)
            self.assertEqual(ours.total_beats, ref.total_beats)

    def test_registered_and_sliceable(self):
        model = score_models.get_score_model("c_major_both")
        segs = slice_score_to_segments(model)
        self.assertEqual(len(segs), 8)
        self.assertEqual(segs[0].expected_pitches, (48, 60))

    def test_canon(self):
        lh, both = load_piece("canon_lh"), load_piece("canon_both")
        self.assertEqual((len(lh.notes()), lh.total_beats), (64, 32.0))
        ids = both.note_by_id()
        self.assertEqual((ids["RH1"].name, ids["RH1"].measure), ("F#5", 5))
        self.assertEqual([ids[f"LH{i}"].name for i in range(1, 9)],
                         ["D3", "A3", "D4", "F#4", "A2", "E3", "A3", "C#4"])
        self.assertEqual(ids["LH21"].name, "D2")                      # m3 beat 3 low D
        self.assertEqual(both.to_score_model().melody.pitches[0], 0)  # RH rests m1-4

    def test_wav_path_for_raw_folder(self):
        from eval.cli import wav_path_for
        self.assertEqual(wav_path_for(Path("/x/eval-interviews/raw/a.m4a")),
                         Path("/x/eval-interviews/processed/a.wav"))


class TestLabels(unittest.TestCase):
    spec = load_piece("c_major_rh")

    def test_good_labels(self):
        p = write_tmp('piece = "c_major_rh"\nperfect = false\n'
                      '[[error]]\nnote = "RH4"\ntype = "wrong_pitch"\nplayed = "E4"\n'
                      '[[error]]\nafter = "START"\ntype = "extra"\n')
        lab = load_labels(p, self.spec)
        self.assertEqual(lab.note_labels(self.spec)["RH4"], "wrong_pitch")
        self.assertEqual(lab.played(), {"RH4": 64})
        self.assertEqual(lab.extras()[0].after, None)

    def test_rejects_bad_id(self):
        p = write_tmp('piece = "c_major_rh"\n[[error]]\nnote = "RH9"\ntype = "missed"\n')
        with self.assertRaises(ValueError):
            load_labels(p, self.spec)

    def test_rejects_perfect_with_errors(self):
        p = write_tmp('piece = "c_major_rh"\nperfect = true\n[[error]]\nnote = "RH1"\ntype = "missed"\n')
        with self.assertRaises(ValueError):
            load_labels(p, self.spec)

    def test_rejects_wrong_and_missed_same_note(self):
        p = write_tmp('piece = "c_major_rh"\n[[error]]\nnote = "RH1"\ntype = "missed"\n'
                      '[[error]]\nnote = "RH1"\ntype = "wrong_pitch"\n')
        with self.assertRaises(ValueError):
            load_labels(p, self.spec)

    def test_todo_skips_validation(self):
        p = write_tmp('piece = "c_major_rh"\ntodo = true\n')
        self.assertTrue(load_labels(p, self.spec).todo)


class TestMetrics(unittest.TestCase):
    spec = load_piece("c_major_rh")

    def test_counts(self):
        gold = TakeLabels("c_major_rh", False, [GoldError("wrong_pitch", "RH4", played=64),
                                                GoldError("missed", "RH6")])
        pred = all_correct(self.spec,
                           RH4=NotePrediction("RH4", "wrong_pitch", 64),        # TP, right played pitch
                           RH2=NotePrediction("RH2", "missed"))                 # FP
        ev = evaluate_take("t", self.spec, gold, pred)
        m = summarize([ev])["overall"]
        self.assertEqual((m["error"]["tp"], m["error"]["fp"], m["error"]["fn"]), (1, 1, 1))
        self.assertAlmostEqual(m["error"]["f1"], 0.5)
        self.assertEqual(m["wrong_pitch"]["tp"], 1)
        self.assertEqual((m["missed"]["fp"], m["missed"]["fn"]), (1, 1))
        self.assertEqual(m["played_pitch_accuracy"], 1.0)
        self.assertEqual(m["confusion"]["missed"]["correct"], 1)
        self.assertEqual(m["pass_fail_accuracy"], 1.0)

    def test_wrong_type_still_detects_error(self):
        gold = TakeLabels("c_major_rh", False, [GoldError("missed", "RH3")])
        pred = all_correct(self.spec, RH3=NotePrediction("RH3", "wrong_pitch", 62))
        m = summarize([evaluate_take("t", self.spec, gold, pred)])["overall"]
        self.assertEqual(m["error"]["tp"], 1)
        self.assertEqual((m["missed"]["fn"], m["wrong_pitch"]["fp"]), (1, 1))

    def test_extras_within_one_note(self):
        gold = TakeLabels("c_major_rh", False, [GoldError("extra", after="RH2"),
                                                GoldError("extra", after="RH7")])
        pred = all_correct(self.spec)
        pred.extras = [ExtraNotePrediction("RH3"), ExtraNotePrediction("RH5")]
        m = summarize([evaluate_take("t", self.spec, gold, pred)])["overall"]
        self.assertEqual((m["extra"]["tp"], m["extra"]["fp"], m["extra"]["fn"]), (1, 1, 1))

    def test_flags_and_unsupported(self):
        gold = TakeLabels("c_major_rh", False, [GoldError("held_long", "RH8")])
        pred = all_correct(self.spec)
        pred.capabilities = frozenset({"wrong_pitch", "missed"})
        m = summarize([evaluate_take("t", self.spec, gold, pred)])["overall"]
        self.assertEqual(m["held_long"]["fn"], 1)
        self.assertTrue(m["held_long"]["unsupported"])
        self.assertIn("extra", m["unsupported_types"])
        self.assertEqual(m["combined"]["fn"], 1)

    def test_perfect_take_false_alarms(self):
        gold = TakeLabels("c_major_rh", True)
        clean = evaluate_take("a", self.spec, gold, all_correct(self.spec))
        noisy = evaluate_take("b", self.spec, gold,
                              all_correct(self.spec, RH1=NotePrediction("RH1", "missed")))
        m = summarize([clean, noisy])["overall"]
        self.assertEqual(m["perfect_takes"], {"n": 2, "clean": 1, "false_alarms_per_take": 0.5})
        self.assertEqual(m["pass_fail_accuracy"], 0.5)
        self.assertIsNone(m["error"]["recall"])

    def test_crashed_take(self):
        gold = TakeLabels("c_major_rh", False, [GoldError("missed", "RH1")])
        ev = evaluate_take("t", self.spec, gold, None)
        m = summarize([ev])["overall"]
        self.assertEqual(m["failed_takes"], ["t"])
        self.assertEqual(m["error"]["fn"], 1)

    def test_prediction_validation(self):
        with self.assertRaises(ValueError):
            TakePrediction(notes=[NotePrediction("RH1", "correct")]).validate(
                [n.id for n in self.spec.notes()])


class TestV2Mapping(unittest.TestCase):
    def _result(self, seg, detected, substitutes, start):
        return SegmentResult(segment_index=seg.index, expected_pitches=seg.expected_pitches,
                             start_frame=start, end_frame=start + 9,
                             expected_start_frame=0, expected_end_frame=0,
                             detected_pitches=tuple(detected), has_ghost_wrong=False,
                             substitutes=tuple(substitutes))

    def test_segment_mapping(self):
        spec = load_piece("ode_harmony")
        segs = slice_score_to_segments(spec.to_score_model())
        results = []
        for s in segs:
            det, subs = list(s.expected_pitches), []
            if s.start_beat == 0.0:            # RH1 E4 wrong (played F4), LH1 C3 fine
                det, subs = [48], [(64, 65)]
            if s.start_beat == 4.0:            # RH5 G4 + LH2 G3 both missing, nothing else heard
                det = []
            results.append(self._result(s, det, subs, start=int(s.start_beat * 10)))
        preds = {p.note_id: p for p in segments_to_note_predictions(results, segs, spec, 0.1, 0.5)}
        self.assertEqual(len(preds), len(spec.notes()))
        self.assertEqual((preds["RH1"].label, preds["RH1"].played_pitch), ("wrong_pitch", 65))
        self.assertEqual(preds["LH1"].label, "correct")
        self.assertEqual((preds["RH5"].label, preds["LH2"].label), ("missed", "missed"))
        self.assertEqual(preds["RH2"].label, "correct")
        self.assertAlmostEqual(preds["RH2"].onset_time_s, 1.5)   # 10 frames * 0.1 + 0.5 offset


class TestSheetImport(unittest.TestCase):
    def _sheet(self, rows):
        head = "take,piece,note_id,pitch,measure,beat,\"label (correct, wrong_pitch, missed)\",played,flag,comment\n"
        return write_tmp(head + "\n".join(rows) + "\n")

    def _takes(self):
        from eval.labels import Take
        return [Take("t1", "c_major_rh", Path("x.wav"), "dev"),
                Take("gone", "c_major_rh", Path("y.wav"), "dev", exclude="backtrack")]

    def _rows(self, overrides=None):
        overrides = overrides or {}
        spec = load_piece("c_major_rh")
        return [overrides.get(n.id, f"t1,c_major_rh,{n.id},{n.name},{n.measure},{n.beat_in_measure:g},correct,,,")
                for n in spec.notes()]

    def test_import_and_render(self):
        from eval.sheet import parse_sheet, render_label_file
        rows = self._rows({"RH4": 't1,c_major_rh,RH4,F4,1,4,wrong_pitch,E4,,"slipped, then fixed"',
                           "RH5": "t1,c_major_rh,RH5,G4,2,1,correct,A4,Late,",
                           "RH6": "t1,c_major_rh,RH6,A4,2,2,MISSED,,,"})
        rows += ["t1,c_major_rh,RH2,,,,extra,D#4,,", "gone,c_major_rh,RH1,C4,1,1,missed,,,"]
        res = parse_sheet(self._sheet(rows), self._takes(), all_pieces())
        self.assertEqual(res.problems, [])
        self.assertEqual(res.skipped, ["gone"])
        self.assertEqual(len(res.warnings), 1)                  # RH5 correct+played -> wrong_pitch
        path = write_tmp(render_label_file(res.takes["t1"], "sheet.csv"))
        spec = load_piece("c_major_rh")
        lab = load_labels(path, spec)
        labels = lab.note_labels(spec)
        self.assertEqual((labels["RH4"], labels["RH5"], labels["RH6"]), ("wrong_pitch", "wrong_pitch", "missed"))
        self.assertEqual(lab.played(), {"RH4": 64, "RH5": 69})
        self.assertEqual(lab.flags(), {"RH5": {"late"}})
        self.assertEqual(lab.extras()[0].after, "RH2")

    def test_all_correct_is_perfect(self):
        from eval.sheet import parse_sheet, render_label_file
        res = parse_sheet(self._sheet(self._rows()), self._takes()[:1], all_pieces())
        self.assertTrue(load_labels(write_tmp(render_label_file(res.takes["t1"], "s")),
                                    load_piece("c_major_rh")).perfect)

    def test_problems_block_import(self):
        from eval.sheet import parse_sheet
        rows = self._rows({"RH3": "t1,c_major_rh,RH3,G4,1,3,correct,,,",      # pitch column shifted
                           "RH4": "t1,c_major_rh,RH4,F4,1,4,wrong,,,"})       # bad label
        rows = rows[:-1]                                                       # RH8 row missing
        res = parse_sheet(self._sheet(rows), self._takes()[:1], all_pieces())
        self.assertEqual(len(res.problems), 3)


if __name__ == "__main__":
    unittest.main()
