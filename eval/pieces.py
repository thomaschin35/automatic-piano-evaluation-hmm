"""
pieces.py
Loads piece specs (eval/pieces/*.toml) into numbered ScoreNotes and converts
them to the model's ScoreModel. Note ids are per hand and 1-based (RH4 = the
4th right-hand note); rests are not numbered.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib

import score_models
from score_models import ScoreModel, ScoreTrack

from eval import EVAL_DIR

PIECES_DIR = EVAL_DIR / "pieces"
HANDS = ("RH", "LH")

_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")


def parse_pitch(name: str) -> int:
    """'C4' -> 60, 'F#4' -> 66, 'Db5' -> 73, 'rest' -> 0."""
    name = name.strip()
    if name.lower() == "rest":
        return 0
    m = _PITCH_RE.match(name)
    if not m:
        raise ValueError(f"Cannot parse pitch name '{name}' (expected e.g. C4, F#4, Bb3, rest)")
    letter, accidental, octave = m.groups()
    pc = _PITCH_CLASSES[letter.upper()] + {"#": 1, "b": -1, "": 0}[accidental]
    return (int(octave) + 1) * 12 + pc


@dataclass(frozen=True)
class ScoreNote:
    id: str
    hand: str
    index: int
    pitch: int
    start_beat: float
    end_beat: float
    measure: int
    beat_in_measure: float

    @property
    def name(self) -> str:
        return score_models.midi_to_note_name(self.pitch)


@dataclass(frozen=True)
class PieceSpec:
    name: str
    title: str
    time_signature: tuple[int, int]
    bpm: float
    pickup_beats: float
    hands: dict[str, list[tuple[int, float]]]   # hand -> [(midi or 0 for rest, beats)]
    pending: bool = False

    @property
    def beats_per_measure(self) -> float:
        num, den = self.time_signature
        return num * 4.0 / den

    @property
    def total_beats(self) -> float:
        return max((sum(d for _, d in evs) for evs in self.hands.values()), default=0.0)

    def notes(self) -> list[ScoreNote]:
        """All sounding notes, ordered by start beat then hand (RH first)."""
        out = []
        for hand in HANDS:
            beat, idx = 0.0, 0
            for pitch, dur in self.hands.get(hand, []):
                if pitch > 0:
                    idx += 1
                    rel = beat - self.pickup_beats
                    measure = int(rel // self.beats_per_measure) + 1
                    out.append(ScoreNote(
                        id=f"{hand}{idx}", hand=hand, index=idx, pitch=pitch,
                        start_beat=beat, end_beat=beat + dur,
                        measure=measure,
                        beat_in_measure=rel - (measure - 1) * self.beats_per_measure + 1,
                    ))
                beat += dur
        out.sort(key=lambda n: (n.start_beat, HANDS.index(n.hand)))
        return out

    def note_by_id(self) -> dict[str, ScoreNote]:
        return {n.id: n for n in self.notes()}

    def to_score_model(self) -> ScoreModel:
        if self.pending:
            raise ValueError(f"Piece '{self.name}' is pending transcription")

        def track(hand: str) -> ScoreTrack:
            evs = self.hands.get(hand) or [(0, self.total_beats)]
            return ScoreTrack(pitches=[p for p, _ in evs], durations_beats=[d for _, d in evs])

        return ScoreModel(
            name=self.name,
            melody=track("RH"),
            harmony=track("LH"),
            total_beats=self.total_beats,
            bpm_ref=self.bpm,
        )


def load_piece(name_or_path: str | Path) -> PieceSpec:
    path = Path(name_or_path)
    if not path.suffix:
        path = PIECES_DIR / f"{name_or_path}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    hands = {}
    for hand, events in raw.get("hands", {}).items():
        if hand not in HANDS:
            raise ValueError(f"{path.name}: unknown hand '{hand}' (use RH/LH)")
        hands[hand] = [(parse_pitch(p), float(d)) for p, d in events]
    spec = PieceSpec(
        name=raw["name"],
        title=raw.get("title", raw["name"]),
        time_signature=tuple(raw.get("time_signature", [4, 4])),
        bpm=float(raw.get("bpm", 120.0)),
        pickup_beats=float(raw.get("pickup_beats", 0.0)),
        hands=hands,
        pending=bool(raw.get("pending", False)),
    )
    if spec.name != path.stem:
        raise ValueError(f"{path.name}: name '{spec.name}' must match the file name")
    if not spec.pending:
        lengths = {h: sum(d for _, d in evs) for h, evs in hands.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"{path.name}: hands have different lengths in beats: {lengths}")
    return spec


def all_pieces() -> dict[str, PieceSpec]:
    return {p.stem: load_piece(p) for p in sorted(PIECES_DIR.glob("*.toml"))}


def register_with_score_models() -> None:
    """Make every ready piece available to stft_features.run_stage1_and_stage2 by name."""
    for name, spec in all_pieces().items():
        if not spec.pending:
            score_models.SCORE_MODELS[name] = spec.to_score_model


register_with_score_models()
