"""
sheet.py
Converts the shared Google Sheets labeling sheet (exported as .csv or .tsv)
into eval/labels/<piece>/<take>.toml. The sheet is the source of truth; the
generated label files say so in their header.

Expected columns (from `python -m eval.cli sheet-template`):
    take, piece, note_id, pitch, measure, beat, label..., played, flag, comment
The label column only has to start with "label", so a header such as
"label (correct, wrong_pitch, missed)" works.

Rules:
  * Every score note of a take must appear exactly once, with the score's pitch
    (guards against rows that were sorted or pasted out of place).
  * label: correct | wrong_pitch | missed | extra (case-insensitive).
    An extra row's note_id is the note it follows, or START.
  * label = correct but `played` names a different pitch -> imported as
    wrong_pitch, with a warning (the note was clearly not played as written).
  * flag: held_long | early | late (case-insensitive), allowed on any note.
  * Excluded takes (manifest `exclude`) are skipped.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import csv
import json

from eval.contract import FLAGS, NOTE_LABELS
from eval.labels import LABELS_DIR, Take
from eval.pieces import PieceSpec, parse_pitch

ROW_LABELS = NOTE_LABELS + ("extra",)


@dataclass
class SheetTake:
    take: Take
    errors: list[dict] = field(default_factory=list)      # TOML-ready [[error]] tables
    comments: list[str] = field(default_factory=list)


@dataclass
class SheetResult:
    takes: dict[str, SheetTake]
    problems: list[str]
    warnings: list[str]
    skipped: list[str]


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    delim = "\t" if text.split("\n", 1)[0].count("\t") > 0 else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    if not rows:
        raise ValueError(f"{path}: no rows")
    label_col = next((k for k in rows[0] if k and k.strip().lower().startswith("label")), None)
    missing = [c for c in ("take", "piece", "note_id", "pitch") if c not in rows[0]]
    if label_col is None or missing:
        raise ValueError(f"{path}: missing columns {missing + ([] if label_col else ['label'])}")
    out = []
    for i, r in enumerate(rows, start=2):           # row 1 is the header
        out.append({
            "row": i,
            "take": (r.get("take") or "").strip(),
            "piece": (r.get("piece") or "").strip(),
            "note_id": (r.get("note_id") or "").strip(),
            "pitch": (r.get("pitch") or "").strip(),
            "label": (r.get(label_col) or "").strip().lower() or "correct",
            "played": (r.get("played") or "").strip(),
            "flag": (r.get("flag") or "").strip().lower(),
            "comment": (r.get("comment") or "").strip(),
        })
    return out


def parse_sheet(path: Path, takes: list[Take], pieces: dict[str, PieceSpec]) -> SheetResult:
    by_id = {t.id: t for t in takes}
    res = SheetResult({}, [], [], [])
    seen: dict[str, set[str]] = {}
    for r in read_rows(path):
        where = f"row {r['row']} ({r['take']} {r['note_id']})"
        if not r["take"]:
            continue
        take = by_id.get(r["take"])
        if take is None:
            res.problems.append(f"{where}: take not in manifest")
            continue
        if take.exclude:
            if take.id not in res.skipped:
                res.skipped.append(take.id)
            continue
        if r["piece"] != take.piece:
            res.problems.append(f"{where}: piece '{r['piece']}' but manifest says '{take.piece}'")
            continue
        spec = pieces[take.piece]
        ids = spec.note_by_id()
        st = res.takes.setdefault(take.id, SheetTake(take))
        label, flag = r["label"], r["flag"]
        if label not in ROW_LABELS:
            res.problems.append(f"{where}: unknown label '{r['label']}' (use {', '.join(ROW_LABELS)})")
            continue
        played = None
        if r["played"]:
            try:
                played = parse_pitch(r["played"])
            except ValueError as e:
                res.problems.append(f"{where}: {e}")
                continue

        if label == "extra":
            if r["note_id"] != "START" and r["note_id"] not in ids:
                res.problems.append(f"{where}: extra must follow a note id of {take.piece} or START")
                continue
            err = {"after": r["note_id"], "type": "extra"}
            if played is not None:
                err["played"] = r["played"]
            if r["comment"]:
                err["comment"] = r["comment"]
            st.errors.append(err)
            continue

        note = ids.get(r["note_id"])
        if note is None:
            res.problems.append(f"{where}: no such note in {take.piece}")
            continue
        if r["pitch"] and r["pitch"] != note.name:
            res.problems.append(f"{where}: pitch column says {r['pitch']} but the score note is {note.name}")
            continue
        if note.id in seen.setdefault(take.id, set()):
            res.problems.append(f"{where}: note listed twice")
            continue
        seen[take.id].add(note.id)
        if flag and flag not in FLAGS:
            res.problems.append(f"{where}: unknown flag '{r['flag']}' (use {', '.join(FLAGS)})")
            continue

        if label == "correct" and played is not None and played != note.pitch:
            res.warnings.append(f"{where}: label 'correct' but played {r['played']} -> imported as wrong_pitch")
            label = "wrong_pitch"
        if label == "wrong_pitch" and played is not None and played == note.pitch:
            res.problems.append(f"{where}: wrong_pitch but played equals the score pitch {note.name}")
            continue
        if label in ("wrong_pitch", "missed"):
            err = {"note": note.id, "type": label}
            if label == "wrong_pitch" and played is not None:
                err["played"] = r["played"]
            if r["comment"]:
                err["comment"] = r["comment"]
            st.errors.append(err)
        if flag:
            st.errors.append({"note": note.id, "type": flag})
        if r["comment"] and label == "correct":
            st.comments.append(f"{note.id}: {r['comment']}")

    for tid, st in res.takes.items():
        missing = [n.id for n in pieces[st.take.piece].notes() if n.id not in seen.get(tid, set())]
        if missing:
            res.problems.append(f"{tid}: {len(missing)} score notes have no row (e.g. {', '.join(missing[:5])})")
    for t in takes:
        if not t.exclude and t.id not in res.takes:
            res.warnings.append(f"{t.id}: in the manifest but not in the sheet (label file left unchanged)")
    return res


def _toml_value(v) -> str:
    return json.dumps(v, ensure_ascii=False)        # JSON strings are valid TOML basic strings


def render_label_file(st: SheetTake, source_name: str) -> str:
    lines = [f"# Generated from '{source_name}' by `python -m eval.cli import-sheet`.",
             "# Edit the shared sheet and re-import rather than editing this file.",
             f'piece = "{st.take.piece}"',
             f"perfect = {'false' if st.errors else 'true'}"]
    for c in st.comments:
        lines.append(f"# note {c}")
    for err in st.errors:
        lines += ["", "[[error]]"] + [f"{k} = {_toml_value(v)}" for k, v in err.items()]
    return "\n".join(lines) + "\n"


def write_label_files(res: SheetResult, source_name: str) -> list[Path]:
    paths = []
    for st in res.takes.values():
        p = LABELS_DIR / st.take.piece / f"{st.take.id}.toml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_label_file(st, source_name))
        paths.append(p)
    return paths
