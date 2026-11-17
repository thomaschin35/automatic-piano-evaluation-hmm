"""
generalise.py  --  python -m eval.generalise [setting ...]
Is a v3 setting fitted to particular recordings? Every numeric setting is moved by -25% and +25%
(one at a time) and the one-hand real takes are scored again, split by piece and by player.

A setting that leaves every group where it was is not fitted. One that moves only one piece or
one player was probably chosen for those takes and is a risk on new music; one that moves every
group is simply important. Real takes only (seconds per setting); the synthetic suite is in
`python -m eval.sensitivity`.
"""
from __future__ import annotations
from dataclasses import asdict
import sys

from eval.adapters import get_adapter
from eval.cache import load_bundle
from eval.check import _block, _ready_takes
from eval.metrics import evaluate_take
from eval.pieces import all_pieces
from eval import synth

SKIP = {"tempo_passes", "max_skip", "back_bars", "back_min_notes", "max_extras", "presence_window",
        "min_frames_after", "cluster_gap", "same_pitch_gap", "sounding_frames", "level_span"}   # counts


def takes():
    pieces = all_pieces()
    return [(t, spec, labels, load_bundle(audio, t.piece)) for t, spec, labels, audio in _ready_takes(pieces)
            if t.piece in synth.MONO_PIECES]


def groups(ts):
    g = {"all": lambda t: True}
    for p in sorted({t.piece for t, *_ in ts}):
        g[p] = lambda t, p=p: t.piece == p
    for pl in sorted({t.id[0] for t, *_ in ts if t.id[0].isdigit()}):
        g[f"player {pl}"] = lambda t, pl=pl: t.id[0] == pl
    return g


def score(ts, adapter, overrides=None) -> dict[str, tuple[float, int]]:
    """group -> (goal F1, wrong verdicts: false + not found over all classes)"""
    evs = [(t, evaluate_take(t.id, spec, labels, adapter.predict_bundle(b, spec, overrides))) for t, spec, labels, b in ts]
    out = {}
    for name, keep in groups(ts).items():
        blk = _block([e for t, e in evs if keep(t)])
        out[name] = (blk["goal"]["f1"] or 0.0, blk["goal"]["fp"] + blk["goal"]["fn"])
    return out


def main() -> int:
    adapter = get_adapter("v3")
    cfg = adapter.config()
    ts = takes()
    base = score(ts, adapter)
    names = sys.argv[1:] or [k for k, v in cfg.items()
                             if isinstance(v, float) and not isinstance(v, bool) and v != 0 and k not in SKIP]
    cols = list(base)
    print(f"{'setting':<24}{'value':>8}  " + "  ".join(f"{c[:12]:>12}" for c in cols))
    print(f"{'(as shipped) errors':<24}{'':>8}  " + "  ".join(f"{base[c][1]:>12}" for c in cols))
    risky = []
    for name in names:
        for mult in (0.75, 1.25):
            val = cfg[name] * mult
            r = score(ts, adapter, {name: val})
            moved = [c for c in cols if c != "all" and r[c][1] != base[c][1]]
            if moved:
                print(f"{name:<24}{val:>8.3g}  " + "  ".join(
                    f"{r[c][1] - base[c][1]:>+12d}" if r[c][1] != base[c][1] else f"{'.':>12}" for c in cols), flush=True)
                pieces_moved = [c for c in moved if not c.startswith("player")]
                if len(pieces_moved) == 1:
                    risky.append((name, mult, pieces_moved[0]))
    print(f"\n{len(names)} settings x 2 checked; a row is printed only when some group's error count moved "
          "(+ = more wrong verdicts).")
    if risky:
        print("Moves only one piece (possibly fitted to it): " + ", ".join(f"{n} x{m} -> {p}" for n, m, p in risky))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
