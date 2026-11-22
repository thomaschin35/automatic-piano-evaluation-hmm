"""
score_reader.py
Turns sheet music into the PieceSpec the grader follows.

    PDF / image --(homr, optical music recognition)--> MusicXML --(parser below)--> PieceSpec

homr lives in its own virtual env (.venv-omr, see ui/README.md) and is called as a subprocess, so
its dependencies never touch the algorithm's env. MusicXML / .mxl files skip the recognition step.

Each staff becomes one hand. Notes that start together on a staff are one chord of that hand
(all its tones last as long as the longest one); a note still sounding when the next one starts is
cut there, so each hand is a sequence of single notes, chords and rests.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from score_models import midi_to_note_name
from eval.pieces import PieceSpec, parse_chord

ROOT = Path(__file__).resolve().parent
OMR_ENV = ROOT / ".venv-omr"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
XML_SUFFIXES = {".musicxml", ".xml", ".mxl"}
GRID = 256            # onsets snap to 1/256 beat: binary fractions, so float sums of durations stay exact
_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class ScoreReadError(Exception):
    """The file could not be turned into a score; the message is shown to the user."""


# ------------------------------------------------------------- score <-> plain dict (JSON / TOML shape)
def spec_to_dict(spec: PieceSpec) -> dict:
    return {
        "name": spec.name, "title": spec.title, "time_signature": list(spec.time_signature),
        "bpm": spec.bpm, "pickup_beats": spec.pickup_beats,
        "hands": {h: [["+".join(midi_to_note_name(p) for p in ps) if ps else "rest", d] for ps, d in evs]
                  for h, evs in spec.hands.items()},
    }


def spec_from_dict(d: dict) -> PieceSpec:
    """Same shape as eval/pieces/*.toml. A shorter hand is padded with a rest so both end together."""
    hands = {}
    for hand, events in (d.get("hands") or {}).items():
        if hand not in ("RH", "LH"):
            raise ScoreReadError(f"Unknown hand '{hand}' (use RH / LH)")
        try:
            evs = [(parse_chord(str(p)), float(dur)) for p, dur in events]
        except (ValueError, TypeError) as e:
            raise ScoreReadError(f"{hand}: {e}") from e
        if any(dur <= 0 for _, dur in evs):
            raise ScoreReadError(f"{hand}: every note needs a length above 0 beats")
        if any(ps for ps, _ in evs):
            hands[hand] = evs
    if not hands:
        raise ScoreReadError("The score has no notes")
    total = max(sum(dur for _, dur in evs) for evs in hands.values())
    for evs in hands.values():
        short = total - sum(dur for _, dur in evs)
        if short > 1e-9:
            evs.append(((), short))
    num, den = d.get("time_signature") or [4, 4]
    return PieceSpec(
        name=str(d.get("name") or "uploaded"), title=str(d.get("title") or "Uploaded score"),
        time_signature=(int(num), int(den)), bpm=float(d.get("bpm") or 100.0),
        pickup_beats=float(d.get("pickup_beats") or 0.0), hands=hands,
    )


# ------------------------------------------------------------- MusicXML -> notes
@dataclass
class _Note:
    onset: Fraction      # beats (quarter notes) from the start of the file
    dur: Fraction
    midi: int
    tie_stop: bool


@dataclass
class _Page:
    staves: dict[tuple[int, int], list[_Note]]     # (part, staff number) -> notes
    clefs: dict[tuple[int, int], str]              # first clef sign per staff
    total: Fraction
    time_signature: tuple[int, int] | None
    pickup: Fraction
    bpm: float | None
    title: str
    irregular: int                                 # bars whose content did not fill the time signature


def _xml_root(path: Path) -> ET.Element:
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.startswith("META-INF") and n.lower().endswith((".xml", ".musicxml"))]
            if not names:
                raise ScoreReadError("The .mxl file has no score inside")
            data = z.read(names[0])
    else:
        data = path.read_bytes()
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ScoreReadError(f"Not a readable MusicXML file: {e}") from e
    for el in root.iter():                           # drop namespaces, if any
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    if root.tag != "score-partwise":
        raise ScoreReadError(f"Unsupported MusicXML layout '{root.tag}' (need score-partwise)")
    return root


def _parse_page(path: Path, allow_pickup: bool) -> _Page:
    root = _xml_root(path)
    staves: dict[tuple[int, int], list[_Note]] = defaultdict(list)
    clefs: dict[tuple[int, int], str] = {}
    time_sig, bpm, pickup, irregular, total = None, None, Fraction(0), 0, Fraction(0)
    for pi, part in enumerate(root.findall("part")):
        divisions, nominal, start = 1, None, Fraction(0)
        for mi, measure in enumerate(part.findall("measure")):
            cursor = last_onset = reach = Fraction(0)
            for el in measure:
                if el.tag == "attributes":
                    if el.findtext("divisions"):
                        divisions = int(el.findtext("divisions"))
                    t = el.find("time")
                    if t is not None and t.findtext("beats") and t.findtext("beat-type"):
                        sig = (int(t.findtext("beats")), int(t.findtext("beat-type")))
                        nominal = Fraction(sig[0] * 4, sig[1])
                        time_sig = time_sig or sig
                    for c in el.findall("clef"):
                        clefs.setdefault((pi, int(c.get("number", 1))), c.findtext("sign") or "G")
                elif el.tag == "note":
                    if el.find("grace") is not None:
                        continue
                    dur = Fraction(int(float(el.findtext("duration") or 0)), divisions)
                    chord = el.find("chord") is not None
                    onset = last_onset if chord else cursor
                    pitch = el.find("pitch")
                    if pitch is not None and el.find("rest") is None:
                        midi = (int(pitch.findtext("octave")) + 1) * 12 + _STEPS[pitch.findtext("step")] \
                            + int(float(pitch.findtext("alter") or 0))
                        staves[(pi, int(el.findtext("staff") or 1))].append(_Note(
                            start + onset, dur, midi, any(t.get("type") == "stop" for t in el.findall("tie"))))
                    if not chord:
                        last_onset, cursor = cursor, cursor + dur
                        reach = max(reach, cursor)
                elif el.tag in ("backup", "forward"):
                    step = Fraction(int(float(el.findtext("duration") or 0)), divisions)
                    cursor = max(Fraction(0), cursor - step) if el.tag == "backup" else cursor + step
                    reach = max(reach, cursor)
                elif el.tag in ("direction", "sound") and bpm is None:
                    s = el if el.tag == "sound" else el.find("sound")
                    if s is not None and s.get("tempo"):
                        bpm = float(s.get("tempo"))
            if nominal is None:
                length = reach
            elif mi == 0 and allow_pickup and 0 < reach < nominal:
                length = reach                                   # a short first bar is a pickup
                if pi == 0:
                    pickup = reach
            else:
                length = max(nominal, reach)                     # short bars are padded with rests
                irregular += reach != nominal and reach > 0 and pi == 0
            start += length
        total = max(total, start)
    title = (root.findtext("work/work-title") or root.findtext("movement-title") or "").strip()
    return _Page(dict(staves), clefs, total, time_sig, pickup, bpm, title, irregular)


def _q(x: Fraction) -> float:
    return round(float(x) * GRID) / GRID


def _line(notes: list[_Note], total: Fraction) -> list[tuple[tuple[int, ...], float]]:
    """One staff -> [(midi pitches low to high, () for a rest, beats)].

    Tied notes are joined. Notes that start together are one chord, as long as its longest tone.
    A chord still sounding when the next one starts is cut there."""
    merged: list[_Note] = []
    for n in sorted(notes, key=lambda n: (n.onset, n.midi)):
        prev = next((m for m in reversed(merged) if m.midi == n.midi and m.onset + m.dur == n.onset), None)
        if n.tie_stop and prev is not None:
            prev.dur += n.dur
        else:
            merged.append(_Note(n.onset, n.dur, n.midi, False))
    by_onset: dict[Fraction, list[_Note]] = defaultdict(list)
    for n in merged:
        by_onset[n.onset].append(n)
    seq = [(onset, tuple(sorted({n.midi for n in ns})), max(n.dur for n in ns))
           for onset, ns in sorted(by_onset.items())]
    events: list[tuple[tuple[int, ...], float]] = []
    t = 0.0
    for i, (onset, pitches, dur) in enumerate(seq):
        on = _q(onset)
        end = min(_q(onset + dur), _q(seq[i + 1][0]) if i + 1 < len(seq) else _q(total))
        if end <= on or on < t:
            continue
        if on > t:
            events.append(((), on - t))
        events.append((pitches, end - on))
        t = end
    if _q(total) > t:
        events.append(((), _q(total) - t))
    return events


def piece_from_musicxml(paths: Path | list[Path], title: str = "") -> tuple[PieceSpec, list[str]]:
    """MusicXML file(s), one per page in reading order -> (PieceSpec, warnings)."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    pages = [_parse_page(Path(p), allow_pickup=(i == 0)) for i, p in enumerate(paths)]
    hands: dict[str, list[tuple[tuple[int, ...], float]]] = {"RH": [], "LH": []}
    warnings, ignored, irregular = [], 0, 0
    for page in pages:
        keys = sorted(page.staves)
        if len(keys) == 1:
            assign = {"LH" if page.clefs.get(keys[0]) == "F" else "RH": keys[0]}
        else:
            assign = dict(zip(("RH", "LH"), keys))
            ignored += max(0, len(keys) - 2)
        for hand in hands:
            if hand in assign:
                evs = _line(page.staves[assign[hand]], page.total)
            else:
                evs = [((), _q(page.total))] if page.total else []
            hands[hand] += evs
        irregular += page.irregular
    hands = {h: evs for h, evs in hands.items() if any(ps for ps, _ in evs)}
    if not hands:
        raise ScoreReadError("No notes were found in the score")
    if ignored:
        warnings.append(f"{ignored} extra staff/staves were ignored (only the first two are used).")
    if irregular:
        warnings.append(f"{irregular} bar(s) did not add up to the time signature and were padded with rests. "
                        "Check those bars below; a note may have been misread.")
    first = pages[0]
    spec = PieceSpec(
        name="uploaded", title=title or first.title or "Uploaded score",
        time_signature=first.time_signature or (4, 4), bpm=first.bpm or 100.0,
        pickup_beats=_q(first.pickup), hands=hands,
    )
    if first.time_signature is None:
        warnings.append("No time signature was found; 4/4 is assumed.")
    return spec, warnings


# ------------------------------------------------------------- PDF / image -> MusicXML (homr)
def omr_to_musicxml(path: Path, timeout: int = 900) -> Path:
    """PDF or image -> MusicXML, written beside the input. homr renders PDF pages itself (300 dpi)
    and merges all pages into one file."""
    homr = OMR_ENV / "bin" / "homr"
    if not homr.exists():
        raise ScoreReadError("The sheet-music reader is not installed. From the repo root run: "
                             "python3 -m venv .venv-omr && .venv-omr/bin/pip install homr")
    try:
        r = subprocess.run([str(homr), str(path)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ScoreReadError(f"Reading the notes took longer than {timeout} s and was stopped") from e
    out = path.with_suffix(".musicxml")
    if r.returncode != 0 or not out.exists():
        log = (r.stderr or r.stdout).strip()
        if "No noteheads or clefs found" in log or "staff" in log.lower() and "Found 0" in log:
            raise ScoreReadError("No music was found in the file. Use a clear, upright page of printed sheet music.")
        raise ScoreReadError(f"The notes could not be read (homr): {log[-400:]}")
    return out


def read_score(path: str | Path) -> tuple[PieceSpec, list[str]]:
    """Any supported score file (.pdf, .png, .jpg, .musicxml, .xml, .mxl, .toml) -> (PieceSpec, warnings)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if not path.is_file():
        raise ScoreReadError(f"Score file not found: {path}")
    if suffix == ".toml":
        from eval.pieces import load_piece
        return load_piece(path), []
    if suffix in XML_SUFFIXES:
        return piece_from_musicxml(path, title="")
    if suffix == ".pdf" or suffix in IMAGE_SUFFIXES:
        with tempfile.TemporaryDirectory(prefix="omr_") as tmp:      # homr writes its files beside the input
            src = Path(tmp) / f"score{suffix}"
            shutil.copyfile(path, src)
            spec, warnings = piece_from_musicxml(omr_to_musicxml(src))
        warnings.insert(0, "The notes were read automatically from the page. Check them before grading: "
                           "a misread note would be reported as a wrong note.")
        return spec, warnings
    raise ScoreReadError(f"Unsupported score file type '{suffix}' (use PDF, PNG, JPG, MusicXML or a piece .toml)")
