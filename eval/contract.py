"""
contract.py  --  The per-note prediction format every model adapter must return.

This is the interface between the evaluator and any model. A model plugs in
by implementing

    predict(audio_path: str, piece: PieceSpec, out_dir: Path) -> TakePrediction

and registering it in eval/adapters.py. Notes are identified by the ids that
eval/pieces.py assigns (hand + 1-based index within that hand, e.g. "RH4");
`PieceSpec.notes()` gives the full list with pitches and beats.

Rules:
  * Return one NotePrediction for every ScoreNote in the piece.
  * `label` is the note-level verdict: "correct", "wrong_pitch" or "missed".
  * `flags` may add held_long / early / late to any note (including correct ones).
  * Inserted notes that belong to no score note go in `extras`, positioned by
    the score note they follow (`after_note_id`, None = before the first note).
  * `capabilities` lists which error types the model can emit at all; types
    not listed are reported as N/A instead of being read as "never happens".
  * Times are in seconds of the ORIGINAL audio file (add any trim offset).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal

NOTE_LABELS = ("correct", "wrong_pitch", "missed")
NOTE_ERRORS = ("wrong_pitch", "missed")
FLAGS = ("held_long", "early", "late")
EXTRA = "extra"
ERROR_TYPES = NOTE_ERRORS + (EXTRA,) + FLAGS


@dataclass(frozen=True)
class NotePrediction:
    note_id: str
    label: Literal["correct", "wrong_pitch", "missed"]
    played_pitch: int | None = None
    onset_time_s: float | None = None
    flags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ExtraNotePrediction:
    after_note_id: str | None
    pitch: int | None = None
    time_s: float | None = None


@dataclass
class TakePrediction:
    notes: list[NotePrediction]
    extras: list[ExtraNotePrediction] = field(default_factory=list)
    capabilities: frozenset[str] = frozenset(ERROR_TYPES)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "notes": [{**asdict(n), "flags": sorted(n.flags)} for n in self.notes],
            "extras": [asdict(e) for e in self.extras],
            "capabilities": sorted(self.capabilities),
            "artifacts": self.artifacts,
        }

    def validate(self, note_ids: list[str]) -> None:
        got = [n.note_id for n in self.notes]
        if sorted(got) != sorted(note_ids):
            missing = set(note_ids) - set(got)
            unknown = set(got) - set(note_ids)
            raise ValueError(f"Prediction note ids mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")
        for n in self.notes:
            if n.label not in NOTE_LABELS:
                raise ValueError(f"{n.note_id}: unknown label '{n.label}'")
            if not n.flags <= set(FLAGS):
                raise ValueError(f"{n.note_id}: unknown flags {sorted(n.flags - set(FLAGS))}")
