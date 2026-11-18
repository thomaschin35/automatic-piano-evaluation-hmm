"""
labels.py
Ground-truth labels and the manifest.

A label file (eval/labels/<piece>/<take>.toml) pins every mistake to a score
note id from `python -m eval.cli score <piece>`. Notes not listed are correct.

    piece = "c_major_rh"
    perfect = false

    [[error]]
    note = "RH4"
    type = "wrong_pitch"     # wrong_pitch | missed | held_long | early | late
    played = "E4"            # optional, wrong_pitch only

    [[error]]
    after = "RH2"            # extra note between RH2 and RH3 ("START" = before the first note)
    type = "extra"
    played = "D#4"           # optional

A file with `todo = true` has not been filled in yet and is skipped by `run`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from eval import EVAL_DIR, REPO_DIR
from eval.contract import ERROR_TYPES, EXTRA, FLAGS, NOTE_ERRORS
from eval.pieces import PieceSpec, parse_pitch

LABELS_DIR = EVAL_DIR / "labels"
MANIFEST_PATH = EVAL_DIR / "manifest.toml"
SPLITS = ("dev", "test")


@dataclass(frozen=True)
class GoldError:
    type: str
    note_id: str | None = None      # for note errors and flags
    after: str | None = None        # for extras; None = before the first note
    played: int | None = None


@dataclass
class TakeLabels:
    piece: str
    perfect: bool
    errors: list[GoldError] = field(default_factory=list)
    todo: bool = False

    def note_labels(self, spec: PieceSpec) -> dict[str, str]:
        out = {n.id: "correct" for n in spec.notes()}
        for e in self.errors:
            if e.type in NOTE_ERRORS:
                out[e.note_id] = e.type
        return out

    def played(self) -> dict[str, int]:
        return {e.note_id: e.played for e in self.errors
                if e.type == "wrong_pitch" and e.played is not None}

    def flags(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for e in self.errors:
            if e.type in FLAGS:
                out.setdefault(e.note_id, set()).add(e.type)
        return out

    def extras(self) -> list[GoldError]:
        return [e for e in self.errors if e.type == EXTRA]


def load_labels(path: str | Path, spec: PieceSpec) -> TakeLabels:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    if raw.get("piece") != spec.name:
        raise ValueError(f"{path}: piece '{raw.get('piece')}' does not match manifest piece '{spec.name}'")
    todo = bool(raw.get("todo", False))
    errors = []
    for i, e in enumerate(raw.get("error", []), 1):
        where = f"{path.name} error #{i}"
        etype = e.get("type")
        if etype not in ERROR_TYPES:
            raise ValueError(f"{where}: unknown type '{etype}' (use one of {', '.join(ERROR_TYPES)})")
        played = parse_pitch(e["played"]) if "played" in e else None
        if etype == EXTRA:
            after = e.get("after")
            if after is None:
                raise ValueError(f"{where}: extra notes need 'after' (a note id or \"START\")")
            errors.append(GoldError(EXTRA, after=None if after == "START" else after, played=played))
        else:
            if "note" not in e:
                raise ValueError(f"{where}: '{etype}' needs 'note' (e.g. note = \"RH4\")")
            errors.append(GoldError(etype, note_id=e["note"], played=played))
    labels = TakeLabels(piece=spec.name, perfect=bool(raw.get("perfect", False)), errors=errors, todo=todo)
    if not todo:
        validate_labels(labels, spec, path)
    return labels


def validate_labels(labels: TakeLabels, spec: PieceSpec, path: Path) -> None:
    ids = spec.note_by_id()
    if labels.perfect and labels.errors:
        raise ValueError(f"{path}: perfect = true but {len(labels.errors)} errors are listed")
    if not labels.perfect and not labels.errors:
        raise ValueError(f"{path}: perfect = false but no errors are listed")
    seen_note_errors: dict[str, str] = {}
    for e in labels.errors:
        ref = e.after if e.type == EXTRA else e.note_id
        if ref is not None and ref not in ids:
            raise ValueError(f"{path}: unknown note id '{ref}' (valid: {', '.join(ids)})")
        if e.type in NOTE_ERRORS:
            if e.note_id in seen_note_errors:
                raise ValueError(f"{path}: {e.note_id} has both '{seen_note_errors[e.note_id]}' and '{e.type}'")
            seen_note_errors[e.note_id] = e.type
        if e.type == "wrong_pitch" and e.played is not None and e.played == ids[e.note_id].pitch:
            raise ValueError(f"{path}: {e.note_id} wrong_pitch 'played' equals the score pitch")


# ---------------------------------------------------------------- manifest
@dataclass(frozen=True)
class Take:
    id: str
    piece: str
    audio: Path
    split: str
    notes: str = ""
    exclude: str = ""       # non-empty = never scored or labeled; the text says why

    @property
    def labels_path(self) -> Path:
        return LABELS_DIR / self.piece / f"{self.id}.toml"


def load_manifest(path: Path = MANIFEST_PATH) -> list[Take]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    takes, seen = [], set()
    for t in raw.get("take", []):
        if t["id"] in seen:
            raise ValueError(f"manifest: duplicate take id '{t['id']}'")
        if t.get("split", "dev") not in SPLITS:
            raise ValueError(f"manifest: take '{t['id']}' has split '{t.get('split')}' (use dev/test)")
        seen.add(t["id"])
        takes.append(Take(id=t["id"], piece=t["piece"], audio=REPO_DIR / t["audio"],
                          split=t.get("split", "dev"), notes=t.get("notes", ""),
                          exclude=t.get("exclude", "")))
    return takes


def append_manifest_entry(take_id: str, piece: str, audio_rel: str, split: str,
                          path: Path = MANIFEST_PATH) -> None:
    entry = (f'\n[[take]]\nid = "{take_id}"\npiece = "{piece}"\n'
             f'audio = "{audio_rel}"\nsplit = "{split}"\n')
    with open(path, "a") as f:
        f.write(entry)


def label_template(spec: PieceSpec) -> str:
    lines = [
        f'piece = "{spec.name}"',
        "todo = true          # delete this line once the labels below are filled in",
        "perfect = false      # set true for a clean take (and list no errors)",
        "",
        "# Score notes (id  pitch  measure/beat):",
    ]
    lines += [f"#   {n.id:<5} {n.name:<4} m{n.measure} b{n.beat_in_measure:g}" for n in spec.notes()]
    lines += [
        "",
        "# One [[error]] block per mistake. Examples:",
        "# [[error]]",
        '# note = "RH4"',
        '# type = "wrong_pitch"   # wrong_pitch | missed | held_long | early | late',
        '# played = "E4"          # optional',
        "#",
        "# [[error]]",
        '# after = "RH2"          # extra note between RH2 and RH3 ("START" = before the first note)',
        '# type = "extra"',
        '# played = "D#4"         # optional',
        "",
    ]
    return "\n".join(lines)
