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
# F1 columns shown everywhere: the two headlines, then the error types in scope
F1_KEYS = [("error", "err F1"), ("goal", "goal F1"), ("wrong_pitch", "wrong"), ("missed", "missed"),
           ("extra", "extra"), ("late", "late"), ("restart", "restart")]


def _fmt(x, pct=False) -> str:
    if x is None:
        return "  -  "
    return f"{x * 100:5.1f}" if pct else f"{x:5.2f}"


def _f1(block: dict | None, key: str):
    """F1, or 'n/a' when the model cannot emit this type, or None when there is nothing to score."""
    if block is None or key not in block:                   # older runs lack newer keys
        return None
    return "n/a" if block[key].get("unsupported") else block[key]["f1"]


def report_rows(metrics: dict) -> list[tuple[str, dict]]:
    """(name, block): each group's pieces then the group total (MONOPHONIC, POLYPHONIC), then OVERALL."""
    from eval.pieces import all_pieces
    group_of = {n: s.group for n, s in all_pieces().items()}
    out = []
    for g, gb in metrics.get("by_group", {}).items():
        out += [(p, b) for p, b in metrics["by_piece"].items() if group_of.get(p) == g] + [(g.upper(), gb)]
    return (out or list(metrics["by_piece"].items())) + [("OVERALL", metrics["overall"])]


def print_summary(metrics: dict, title: str = "") -> None:
    cols = ["takes", "notes", "err P", "err R"] + [c for _, c in F1_KEYS] + ["clean", "pass%"]
    header = f"{'piece':<16}" + "".join(f"{c:>8}" for c in cols)
    line = "-" * len(header)
    print("\n" + "=" * len(header) + (f"\n{title}" if title else "") + f"\n{header}\n{line}")
    for name, b in report_rows(metrics):
        pt = b["perfect_takes"]
        cells = [b["n_takes"], b["n_notes"], _fmt(b["error"]["precision"]), _fmt(b["error"]["recall"])]
        cells += [v if v == "n/a" else _fmt(v) for v in (_f1(b, k) for k, _ in F1_KEYS)]
        cells += [f"{pt['clean']}/{pt['n']}" if pt["n"] else "  -  ", _fmt(b["pass_fail_accuracy"], pct=True)]
        print(f"{name:<16}" + "".join(f"{c:>8}" for c in cells) + ("\n" + line if name.isupper() else ""))
    o = metrics["overall"]
    print("err = any wrong/missed note flagged on the right note; goal = wrong, missed, extra and late, type must match.\n"
          "'-' = nothing to score, 'n/a' = model cannot emit this type. clean = perfect takes with zero flags.")
    if o["played_pitch_n"]:
        print(f"Played-pitch accuracy on detected wrong notes: {_fmt(o['played_pitch_accuracy'], pct=True)}% "
              f"(n={o['played_pitch_n']})")
    if o["failed_takes"]:
        print(f"FAILED takes (counted as all-correct predictions): {', '.join(o['failed_takes'])}")
    conf = o["confusion"]
    labs = [l for l in conf if l == "correct" or sum(conf[l].values()) or sum(conf[g][l] for g in conf)]
    print("\nConfusion (rows = your label, cols = model):")
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
    print(f"\n{'piece':<16}" + "".join(f"{c:>19}" for _, c in F1_KEYS) + f"{'clean':>16}")
    for name, _ in report_rows(mb):
        ba, bb = (m["overall"] if name == "OVERALL" else m.get("by_group", {}).get(name.lower()) or m["by_piece"].get(name)
                  for m in (ma, mb))
        cells = []
        for k, _ in F1_KEYS:
            fa, fb = _f1(ba, k), _f1(bb, k)
            num = isinstance(fa, float) and isinstance(fb, float)
            cells.append(f"{fa if isinstance(fa, str) else _fmt(fa).strip()}->"
                         f"{fb if isinstance(fb, str) else _fmt(fb).strip()}" + (f" {fb - fa:+.2f}" if num else ""))
        clean = lambda b: f"{b['perfect_takes']['clean']}/{b['perfect_takes']['n']}" if b else "-"
        print(f"{name:<16}" + "".join(f"{c:>19}" for c in cells) + f"{clean(ba) + '->' + clean(bb):>16}")
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
