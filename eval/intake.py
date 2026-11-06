"""
intake.py
Finds recordings that are not in the manifest yet and works out their piece from
the file name, using the rules in eval/intake.toml.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import tomllib

from eval import EVAL_DIR, REPO_DIR
from eval.labels import Take

INTAKE_PATH = EVAL_DIR / "intake.toml"
AUDIO_EXTS = (".m4a", ".wav", ".mp3", ".aac", ".flac")
_NAME = re.compile(r"^(\d*)([a-z]+)$")


@dataclass(frozen=True)
class Rule:
    key: str
    hands: str
    words: tuple[str, ...]
    piece: str


@dataclass(frozen=True)
class Candidate:
    path: Path
    take_id: str
    piece: str | None
    problem: str = ""       # non-empty = will not be registered; says why


def load_intake(path: Path = INTAKE_PATH) -> tuple[list[Path], list[Rule]]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    folders = [REPO_DIR / p for p in raw.get("folders", [])]
    rules = [Rule(r["key"], r.get("hands", "1h"), tuple(r.get("words", ())), r["piece"])
             for r in raw.get("rule", [])]
    return folders, rules


def infer_piece(name: str, rules: list[Rule]) -> tuple[str | None, str]:
    """('ode_harmony', '') for '1ode-2h-3'; (None, reason) when no rule fits."""
    parts = name.lower().split("-")
    m = _NAME.match(parts[0])
    if not m:
        return None, f"can't read '{parts[0]}' as <player><piece>, e.g. 1ode"
    key = m.group(2)
    rest = set(parts[1:])
    hands = next((p for p in parts[1:] if p in ("1h", "2h")), "1h")
    rule_words = {w for r in rules for w in r.words}
    for r in rules:
        if r.key == key and r.hands == hands and set(r.words) <= rest \
                and not (rest & rule_words) - set(r.words):
            return r.piece, ""
    return None, f"no rule in {INTAKE_PATH.name} for piece '{key}', {hands}" + (
        f", {'/'.join(sorted(rest & rule_words))}" if rest & rule_words else "")


def _digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def find_new_recordings(takes: list[Take], pieces: dict, folders: list[Path] | None = None,
                        rules: list[Rule] | None = None) -> list[Candidate]:
    """Audio files in the intake folders whose path is not in the manifest yet."""
    default_folders, default_rules = load_intake()
    folders = default_folders if folders is None else folders
    rules = default_rules if rules is None else rules
    known_paths = {t.audio.resolve() for t in takes}
    known_ids = {t.id for t in takes}
    files = sorted(p for d in folders if d.exists() for p in d.iterdir()
                   if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
    digests: dict[str, Path] = {}
    for p in files:                         # every file, so a copy of a registered take is caught too
        digests.setdefault(_digest(p), p)
    out, new_ids = [], set()
    for p in files:
        if p.resolve() in known_paths:
            continue
        tid = p.stem
        twin = digests[_digest(p)]
        if twin != p:
            out.append(Candidate(p, tid, None, f"identical to {twin.name}; delete or rename one"))
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", tid):
            out.append(Candidate(p, tid, None, "file name has spaces or symbols; rename it"))
            continue
        if tid in known_ids or tid in new_ids:
            out.append(Candidate(p, tid, None, f"take id '{tid}' already used; rename the file"))
            continue
        piece, why = infer_piece(tid, rules)
        if piece is None:
            out.append(Candidate(p, tid, None, why))
        elif piece not in pieces:
            out.append(Candidate(p, tid, None, f"{INTAKE_PATH.name} names unknown piece '{piece}'"))
        else:
            new_ids.add(tid)
            out.append(Candidate(p, tid, piece))
    return out
