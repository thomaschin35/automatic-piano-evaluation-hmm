"""
runs.py
Run directories (eval/runs/<timestamp>_<commit>[-dirty]/), console summaries
and run-to-run comparison.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import csv
import json
import subprocess

from eval import EVAL_DIR, REPO_DIR

RUNS_DIR = EVAL_DIR / "runs"


def git_info() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "--short", "HEAD") or "nogit",
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "dirty_files": status.splitlines()}


def new_run_dir(model: str) -> tuple[Path, dict]:
    info = git_info()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{model}_{info['commit']}{'-dirty' if info['dirty'] else ''}"
    path = RUNS_DIR / name
    path.mkdir(parents=True, exist_ok=False)
    return path, info


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def resolve_run(name: str) -> Path:
    p = Path(name)
    if p.is_dir():
        return p
    p = RUNS_DIR / name
    if p.is_dir():
        return p
    matches = sorted(RUNS_DIR.glob(f"*{name}*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"No unique run matching '{name}' in {RUNS_DIR}")


# ---------------------------------------------------------------- printing
def _fmt(x, pct=False) -> str:
    if x is None:
        return "  -  "
    return f"{x * 100:5.1f}" if pct else f"{x:5.2f}"


def _f1_cell(block: dict, key: str) -> str:
    m = block[key]
    if m.get("unsupported"):
        return "  n/a"
    return _fmt(m["f1"])


def print_summary(metrics: dict, title: str = "") -> None:
    cols = ["takes", "notes", "err P", "err R", "err F1", "wrong", "missed", "extra",
            "held", "early", "late", "comb", "clean", "pass%"]
    header = f"{'piece':<16}" + "".join(f"{c:>7}" for c in cols)
    print("\n" + "=" * len(header))
    if title:
        print(title)
    print(header)
    print("-" * len(header))
    rows = list(metrics["by_piece"].items()) + [("OVERALL", metrics["overall"])]
    for name, b in rows:
        pt = b["perfect_takes"]
        cells = [
            f"{b['n_takes']:>5}", f"{b['n_notes']:>5}",
            _fmt(b["error"]["precision"]), _fmt(b["error"]["recall"]), _fmt(b["error"]["f1"]),
            _f1_cell(b, "wrong_pitch"), _f1_cell(b, "missed"), _f1_cell(b, "extra"),
            _f1_cell(b, "held_long"), _f1_cell(b, "early"), _f1_cell(b, "late"),
            _fmt(b["combined"]["f1"]),
            f"{pt['clean']}/{pt['n']}" if pt["n"] else "  -  ",
            _fmt(b["pass_fail_accuracy"], pct=True),
        ]
        if name == "OVERALL":
            print("-" * len(header))
        print(f"{name:<16}" + "".join(f"{c:>7}" for c in cells))
    o = metrics["overall"]
    print("-" * len(header))
    print("F1 columns: '-' = nothing to score, 'n/a' = model cannot emit this type. "
          "clean = perfect takes with zero flagged errors.")
    if o["played_pitch_n"]:
        print(f"Played-pitch accuracy on detected wrong notes: {_fmt(o['played_pitch_accuracy'], pct=True)}% "
              f"(n={o['played_pitch_n']})")
    if o["failed_takes"]:
        print(f"FAILED takes (counted as all-correct predictions): {', '.join(o['failed_takes'])}")
    conf = o["confusion"]
    print("\nConfusion (rows = gold, cols = predicted):")
    labs = list(conf)
    print(f"{'':>13}" + "".join(f"{l:>13}" for l in labs))
    for g in labs:
        print(f"{g:>13}" + "".join(f"{conf[g][p]:>13}" for p in labs))
    print("=" * len(header) + "\n")


def print_take_errors(rows: list[dict]) -> None:
    """One line per note where gold and prediction disagree."""
    bad = [r for r in rows if not r["label_ok"] or r["gold_flags"] != r["pred_flags"]]
    if not bad:
        print("No per-note disagreements.")
        return
    print("Per-note disagreements:")
    for r in bad:
        print(f"  {r['take']:<28} {r['note_id']:<5} {r['pitch']:<4} m{r['measure']} b{r['beat']:g}  "
              f"gold={r['gold']}{'+' + r['gold_flags'] if r['gold_flags'] else ''}  "
              f"pred={r['pred']}{'+' + r['pred_flags'] if r['pred_flags'] else ''}"
              f"{'  played=' + str(r['pred_played']) if r['pred'] == 'wrong_pitch' else ''}")


# ---------------------------------------------------------------- compare
def _read_rows(run: Path) -> dict[tuple[str, str], dict]:
    with open(run / "per_note.csv") as f:
        return {(r["take"], r["note_id"]): r for r in csv.DictReader(f)}


def compare_runs(a: Path, b: Path) -> None:
    ma = json.loads((a / "metrics.json").read_text())
    mb = json.loads((b / "metrics.json").read_text())
    ra, rb = _read_rows(a), _read_rows(b)
    common_takes = {t for t, _ in ra} & {t for t, _ in rb}
    print(f"\nA = {a.name}\nB = {b.name}")
    only = ({t for t, _ in ra} ^ {t for t, _ in rb})
    if only:
        print(f"Takes in only one run (ignored for note flips): {', '.join(sorted(only))}")
    print("\nNote: metric deltas use each run's full take set; flips use common takes only.")
    keys = ["error", "wrong_pitch", "missed", "extra", "combined"]
    print(f"\n{'piece':<16}" + "".join(f"{k + ' F1':>18}" for k in keys) + f"{'clean':>14}")
    pieces = sorted(set(ma["by_piece"]) | set(mb["by_piece"])) + ["OVERALL"]
    for p in pieces:
        ba = ma["overall"] if p == "OVERALL" else ma["by_piece"].get(p)
        bb = mb["overall"] if p == "OVERALL" else mb["by_piece"].get(p)
        cells = []
        for k in keys:
            fa = ba[k]["f1"] if ba else None
            fb = bb[k]["f1"] if bb else None
            delta = f"{fb - fa:+.2f}" if fa is not None and fb is not None else "  "
            cells.append(f"{_fmt(fa).strip()}->{_fmt(fb).strip()} {delta}")
        ca = f"{ba['perfect_takes']['clean']}/{ba['perfect_takes']['n']}" if ba else "-"
        cb = f"{bb['perfect_takes']['clean']}/{bb['perfect_takes']['n']}" if bb else "-"
        print(f"{p:<16}" + "".join(f"{c:>18}" for c in cells) + f"{ca + '->' + cb:>14}")
    fixed, broken = [], []
    for key in sorted(set(ra) & set(rb)):
        if key[0] not in common_takes:
            continue
        oa, ob = ra[key]["label_ok"] == "True", rb[key]["label_ok"] == "True"
        if oa != ob:
            (fixed if ob else broken).append((key, ra[key], rb[key]))
    print(f"\nNotes fixed in B: {len(fixed)}   Notes broken in B: {len(broken)}")
    for tag, items in (("FIXED", fixed), ("BROKEN", broken)):
        for (take, nid), x, y in items:
            print(f"  {tag:<6} {take:<28} {nid:<5} {x['pitch']:<4} gold={x['gold']:<11} "
                  f"A={x['pred']:<11} B={y['pred']}")
    print()
