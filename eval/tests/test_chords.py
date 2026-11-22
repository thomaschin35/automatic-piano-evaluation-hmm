"""Chords within one hand: piece format, note ids, score model voices, MusicXML reading."""
from __future__ import annotations
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.pieces import all_pieces, load_piece, parse_chord   # noqa: E402
from score_models import slice_score_to_segments              # noqa: E402

CHORD_TOML = """
name = "{name}"
title = "chord test"
time_signature = [4, 4]
[hands]
RH = [["C4+E4+G4", 1], ["D4", 1], ["G4+B4", 2]]
LH = [["C3", 2], ["G2+D3", 2]]
"""


def _piece(text: str, name: str = "chordtest"):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{name}.toml"
        p.write_text(text.format(name=name))
        return load_piece(p)


class ChordFormat(unittest.TestCase):
    def test_parse_chord(self):
        self.assertEqual(parse_chord("C4"), (60,))
        self.assertEqual(parse_chord("G4+C4+E4"), (60, 64, 67))
        self.assertEqual(parse_chord("rest"), ())
        with self.assertRaises(ValueError):
            parse_chord("C4+rest")
        with self.assertRaises(ValueError):
            parse_chord("C4+C4")

    def test_ids_low_to_high_and_consecutive(self):
        spec = _piece(CHORD_TOML)
        ids = {n.id: (n.name, n.start_beat) for n in spec.notes()}
        self.assertEqual([ids[f"RH{i}"] for i in range(1, 7)],
                         [("C4", 0), ("E4", 0), ("G4", 0), ("D4", 1), ("G4", 2), ("B4", 2)])
        self.assertEqual([ids[f"LH{i}"] for i in range(1, 4)], [("C3", 0), ("G2", 2), ("D3", 2)])
        self.assertTrue(spec.has_chords)
        self.assertFalse(spec.is_mono)

    def test_segments_expect_every_chord_tone(self):
        segs = slice_score_to_segments(_piece(CHORD_TOML).to_score_model())
        self.assertEqual([s.onset_pitches for s in segs],
                         [(48, 60, 64, 67), (62,), (43, 50, 67, 71)])
        self.assertEqual(segs[1].expected_pitches, (48, 62))          # D4 struck over the held C3
        model = _piece(CHORD_TOML).to_score_model()
        self.assertEqual(model.melody.pitches, [67, 62, 71])          # right hand: top note
        self.assertEqual(model.harmony.pitches, [48, 43])             # left hand: bottom note

    def test_existing_pieces_unchanged(self):
        for name, spec in all_pieces().items():
            if spec.pending:
                continue
            self.assertFalse(spec.has_chords, name)
            self.assertEqual(spec.to_score_model().extra, (), name)
            for seg in slice_score_to_segments(spec.to_score_model()):
                self.assertEqual(seg.extra_pitches, (), name)

    def test_chord_piece_equals_two_hand_piece(self):
        """c_major_both written as one hand of octave chords gives the same segments."""
        both = load_piece("c_major_both")
        rh, lh = both.hands["RH"], both.hands["LH"]
        events = ", ".join(f'["{_n(a[0][0])}+{_n(b[0][0])}", {a[1]}]' for a, b in zip(rh, lh))
        one = _piece(f'name = "{{name}}"\ntime_signature = [4, 4]\n[hands]\nRH = [{events}]\n')
        key = lambda segs: [(s.start_beat, s.onset_pitches, s.expected_pitches) for s in segs]
        self.assertEqual(key(slice_score_to_segments(one.to_score_model())),
                         key(slice_score_to_segments(both.to_score_model())))


def _n(midi: int) -> str:
    from score_models import midi_to_note_name
    return midi_to_note_name(midi)


class ChordMusicXML(unittest.TestCase):
    XML = """<?xml version="1.0"?><score-partwise><part id="P1"><measure number="1">
<attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time><staves>2</staves></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><staff>1</staff></note>
<note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><staff>1</staff></note>
<note><chord/><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><staff>1</staff></note>
<note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><staff>1</staff></note>
<backup><duration>4</duration></backup>
<note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration><staff>2</staff></note>
<note><chord/><pitch><step>G</step><octave>3</octave></pitch><duration>4</duration><staff>2</staff></note>
</measure></part></score-partwise>"""

    def test_chords_kept(self):
        from score_reader import piece_from_musicxml, spec_from_dict, spec_to_dict
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.musicxml"
            p.write_text(self.XML)
            spec, warnings = piece_from_musicxml(p)
        self.assertEqual(spec.hands["RH"], [((60, 64, 67), 2.0), ((62,), 2.0)])
        self.assertEqual(spec.hands["LH"], [((48, 55), 4.0)])
        self.assertFalse(any("chord" in w for w in warnings))
        d = spec_to_dict(spec)
        self.assertEqual(d["hands"]["RH"][0], ["C4+E4+G4", 2.0])
        self.assertEqual(spec_from_dict(d), spec)


if __name__ == "__main__":
    unittest.main()
