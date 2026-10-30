"""
plots.py
Charts for one run directory (reads metrics.json and per_note.csv), written to
<run>/plots/:

  overview.png    per-piece error precision / recall / F1, plus recall per error type
  confusion.png   gold vs predicted note label, row-normalized, with counts
  note_map.png    every take x every score note: agreement, catches, false alarms, misses

Each plot_* function returns a matplotlib Figure (shown inline in
eval/results.ipynb); plot_run() saves them all.

Colors are a colorblind-validated set (dataviz reference palette, light mode):
three categorical slots for series, one blue ramp for magnitude.
"""
from __future__ import annotations
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]            # blue, orange, aqua: all-pairs CVD-safe
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# note-map outcome classes
AGREE_FILL = "#ecebe6"                                  # neutral: model and labels agree, note correct
OUTCOMES = {
    "caught": ("Error caught", SERIES[0]),
    "false_alarm": ("False alarm (note was correct)", SERIES[1]),
    "missed": ("Error missed", SERIES[2]),
}
ERROR_TYPES = [("wrong_pitch", "Wrong pitch"), ("missed", "Missed note"), ("extra", "Extra note"),
               ("held_long", "Held long"), ("early", "Early"), ("late", "Late")]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": INK_2, "axes.edgecolor": AXIS,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _clean_axes(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fmt(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def _title(fig, title, subtitle):
    fig.text(0.012, 0.985, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.985 - 0.055 * (6 / fig.get_figheight()), subtitle, fontsize=10, color=INK_2, va="top")


# ---------------------------------------------------------------- overview
def plot_overview(metrics: dict, run_name: str = ""):
    pieces = list(metrics["by_piece"])
    rows = pieces + ["OVERALL"]
    blocks = [metrics["by_piece"][p] for p in pieces] + [metrics["overall"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 0.48 * len(rows) + 2.4),
                                  gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.55})
    keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    band, h = 0.8, 0.8 / 3
    for i, (b, row) in enumerate(zip(blocks, rows)):
        if b["error"]["recall"] is None:          # nothing labeled wrong: only false alarms are meaningful
            fp = b["error"]["fp"]
            ax.text(0.01, i, f"no labeled errors  ·  {fp} false alarm{'s' if fp != 1 else ''}",
                    va="center", fontsize=8.5, color=INK_2 if fp else MUTED)
            continue
        for j, (k, _) in enumerate(keys):
            v = b["error"][k]
            y = i - band / 2 + h * (j + 0.5)
            ax.barh(y, max(v, 0.004), height=h * 0.86, color=SERIES[j])
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    labels = []
    for row, b in zip(rows, blocks):
        pt = b["perfect_takes"]
        clean = f"   clean {pt['clean']}/{pt['n']}" if pt["n"] else ""
        labels.append(f"{row}\n{b['n_takes']} takes{clean}")
    ax.set_yticks(range(len(rows)), labels, fontsize=9)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(len(rows) - 1.5, color=AXIS, linewidth=0.8)
    ax.set_ylim(len(rows) - 0.45, -0.6)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("score (0–1)")
    _clean_axes(ax)
    ax.set_title("Note-level error detection, by piece", loc="left", fontsize=11, color=INK, pad=10)
    ax.legend(handles=[Patch(color=c, label=l) for c, (_, l) in zip(SERIES, keys)],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=9)

    # recall per error type, overall
    o = metrics["overall"]
    ys = range(len(ERROR_TYPES))
    for y, (k, label) in zip(ys, ERROR_TYPES):
        m = o[k]
        gold = m["tp"] + m["fn"]
        if m.get("unsupported"):
            ax2.text(0.01, y, f"model can't detect yet  ({gold} labeled)", va="center", fontsize=8.5, color=MUTED)
        elif gold == 0:
            ax2.text(0.01, y, "none labeled", va="center", fontsize=8.5, color=MUTED)
        else:
            r = m["recall"]
            ax2.barh(y, max(r, 0.004), height=0.3, color=SERIES[0])
            ax2.text(r + 0.015, y, f"{m['tp']}/{gold} caught", va="center", fontsize=8.5, color=INK_2)
    ax2.set_yticks(list(ys), [l for _, l in ERROR_TYPES])
    ax2.set_ylim(len(ERROR_TYPES) - 0.4, -0.6)
    ax2.set_xlim(0, 1.25)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.set_xlabel("recall (share of labeled errors caught)")
    _clean_axes(ax2)
    ax2.set_title("Recall by error type, every take shown", loc="left", fontsize=11, color=INK, pad=10)

    o_err = o["error"]
    _title(fig, f"Error F1 {_fmt(o_err['f1'])}  ·  precision {_fmt(o_err['precision'])}  ·  "
                f"recall {_fmt(o_err['recall'])}",
           f"{run_name}  ·  {o['n_takes']} takes, {o['n_notes']} notes  ·  perfect takes with zero false alarms: "
           f"{o['perfect_takes']['clean']}/{o['perfect_takes']['n']}")
    fig.subplots_adjust(left=0.15, right=0.98, top=1 - 1.35 / fig.get_figheight(), bottom=0.9 / fig.get_figheight())
    return fig


# ---------------------------------------------------------------- confusion
def plot_confusion(metrics: dict, run_name: str = ""):
    conf = metrics["overall"]["confusion"]
    labs = list(conf)
    names = {"correct": "Correct", "wrong_pitch": "Wrong pitch", "missed": "Missed"}
    counts = [[conf[g][p] for p in labs] for g in labs]
    fracs = [[c / max(1, sum(row)) for c in row] for row in counts]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(fracs, cmap=cmap, vmin=0, vmax=1)
    for i, row in enumerate(counts):
        for j, c in enumerate(row):
            f = fracs[i][j]
            color = "#ffffff" if f > 0.55 else INK
            ax.text(j, i - 0.08, f"{c}", ha="center", va="center", fontsize=14, fontweight="bold", color=color)
            ax.text(j, i + 0.2, f"{f:.0%} of row", ha="center", va="center", fontsize=8.5, color=color)
    ax.set_xticks(range(3), [names[l] for l in labs])
    ax.set_yticks(range(3), [names[l] for l in labs])
    ax.set_xlabel("Model said")
    ax.set_ylabel("Your label")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x + 0.5 for x in range(2)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(2)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    _title(fig, "What the model said for each note",
           f"{run_name}  ·  rows sum to 100%; diagonal = agreement")
    fig.subplots_adjust(left=0.2, right=0.96, top=0.74, bottom=0.04)
    return fig


# ---------------------------------------------------------------- note map
def _outcome(r: dict) -> str | None:
    g, p = r["gold"], r["pred"]
    if g == "correct" and p == "correct":
        return None
    if g == "correct":
        return "false_alarm"
    if p == "correct":
        return "missed"
    return "caught"


def plot_note_map(rows: list[dict], run_name: str = ""):
    pieces: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        pieces.setdefault(r["piece"], {}).setdefault(r["take"], []).append(r)
    order = sorted(pieces, key=lambda p: (-max(len(v) for v in pieces[p].values()), p))
    heights = [len(pieces[p]) + 1.6 for p in order]
    fig, axes = plt.subplots(len(order), 1, figsize=(14, 0.32 * sum(heights) + 2.2),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.9})
    axes = [axes] if len(order) == 1 else list(axes)
    max_notes = max(len(v) for p in order for v in pieces[p].values())
    for ax, piece in zip(axes, order):
        takes = sorted(pieces[piece])
        notes = pieces[piece][takes[0]]
        for ti, take in enumerate(takes):
            for ni, r in enumerate(pieces[piece][take]):
                oc = _outcome(r)
                color = AGREE_FILL if oc is None else OUTCOMES[oc][1]
                ax.add_patch(plt.Rectangle((ni + 0.06, ti + 0.08), 0.88, 0.84, color=color, linewidth=0))
                if oc == "caught" and r["gold"] != r["pred"]:
                    ax.plot(ni + 0.5, ti + 0.5, marker="x", color="#ffffff", markersize=4.5, mew=1.4)
                if r["gold_flags"]:
                    ax.plot(ni + 0.5, ti + 0.5, marker="o", markersize=3.2, color=INK,
                            markeredgecolor="none")
        ax.set_xlim(0, max_notes)
        ax.set_ylim(len(takes), 0)
        ax.set_yticks([t + 0.5 for t in range(len(takes))], takes, fontsize=8.5)
        # mark each measure start with its note id
        ticks, labels, last_m = [], [], None
        for ni, r in enumerate(notes):
            if r["measure"] != last_m:
                ticks.append(ni + 0.5)
                labels.append(f"m{r['measure']}")
                last_m = r["measure"]
                if ni:
                    ax.axvline(ni, color=AXIS, linewidth=0.8)
        ax.set_xticks(ticks, labels, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{piece}  ({len(notes)} notes, score order)", loc="left", fontsize=10.5, color=INK, pad=4)

    handles = [Patch(color=AGREE_FILL, label="Correct, model agrees")]
    handles += [Patch(color=c, label=l) for l, c in OUTCOMES.values()]
    handles += [Line2D([], [], marker="x", color=SERIES[0], linestyle="none", mew=1.6, label="caught, but wrong error type"),
                Line2D([], [], marker="o", color=INK, linestyle="none", markersize=4,
                       label="labeled held/early/late (model can't flag)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 1 - 0.95 / fig.get_figheight()),
               ncol=6, frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.4)
    _title(fig, "Every note of every take", f"{run_name}  ·  one row per take, one cell per score note")
    fig.subplots_adjust(left=0.1, right=0.99, top=1 - 1.6 / fig.get_figheight(), bottom=0.3 / fig.get_figheight())
    return fig


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    with open(run_dir / "per_note.csv") as f:
        rows = list(csv.DictReader(f))
    return metrics, rows


def plot_run(run_dir: Path) -> list[Path]:
    """Draw every chart for a run and save them as PNGs in <run>/plots/."""
    metrics, rows = load_run(run_dir)
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    figs = {"overview": plot_overview(metrics, run_dir.name),
            "confusion": plot_confusion(metrics, run_dir.name),
            "note_map": plot_note_map(rows, run_dir.name)}
    paths = []
    for name, fig in figs.items():
        paths.append(out / f"{name}.png")
        fig.savefig(paths[-1], dpi=160)
        plt.close(fig)
    return paths
