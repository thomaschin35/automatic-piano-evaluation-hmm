"""
plots.py
Charts for one run directory (reads metrics.json and per_note.csv), written to
<run>/plots/:

  overview.png    error precision / recall / F1 per piece, per group and overall, plus recall per type
  confusion.png   gold vs predicted class (note labels, extras, timing flags), row-normalized;
                  confusion_monophonic.png / confusion_polyphonic.png for each group
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
# error types in the goal score (held_long is reported apart)
ERROR_TYPES = [("wrong_pitch", "Wrong pitch"), ("missed", "Missed note"), ("extra", "Extra note"), ("late", "Late"),
               ("restart", "Went back (restart)")]

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
    """Left: error precision / recall / F1 per piece, per group (mono / poly) and overall.
    Right: recall per error type, overall."""
    from eval.runs import report_rows
    rows, blocks = zip(*report_rows(metrics))
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
    for i, (row, lab) in enumerate(zip(rows, ax.get_yticklabels())):
        if row.isupper():                                   # group and overall totals
            lab.set_fontweight("bold")
            ax.axhline(i + 0.5, color=AXIS, linewidth=0.8)
    ax.set_ylim(len(rows) - 0.45, -0.6)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("score (0–1)")
    _clean_axes(ax)
    ax.set_title("Error detection", loc="left", fontsize=11, color=INK, pad=10)
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
    _title(fig, f"Error F1 {_fmt(o_err['f1'])}  ·  goal F1 {_fmt(o['goal']['f1'])}  ·  precision {_fmt(o_err['precision'])}  ·  "
                f"recall {_fmt(o_err['recall'])}",
           f"{run_name}  ·  {o['n_takes']} takes, {o['n_notes']} notes  ·  perfect takes with zero false alarms: "
           f"{o['perfect_takes']['clean']}/{o['perfect_takes']['n']}")
    fig.subplots_adjust(left=0.15, right=0.98, top=1 - 1.35 / fig.get_figheight(), bottom=0.9 / fig.get_figheight())
    return fig


# ---------------------------------------------------------------- confusion
def plot_confusion(metrics: dict, run_name: str = "", block: dict | None = None):
    """Your label vs the model's class per note (its label, else its timing flag), extras included."""
    b = block or metrics["overall"]
    conf = b["confusion"]
    names = {"correct": "Correct", "wrong_pitch": "Wrong pitch", "missed": "Missed", "extra": "Extra",
             "held_long": "Held long", "late": "Late"}
    labs = [l for l in conf if l == "correct" or sum(conf[l].values()) or sum(conf[g][l] for g in conf)]
    counts = [[conf[g][p] for p in labs] for g in labs]
    fracs = [[c / max(1, sum(row)) for c in row] for row in counts]
    k = len(labs)
    fig, ax = plt.subplots(figsize=(2.2 + 1.15 * k, 2.0 + 1.0 * k))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(fracs, cmap=cmap, vmin=0, vmax=1)
    for i, row in enumerate(counts):
        for j, c in enumerate(row):
            f = fracs[i][j]
            color = "#ffffff" if f > 0.55 else (INK if c else MUTED)
            ax.text(j, i - 0.1, f"{c}", ha="center", va="center", fontsize=13 if c else 10,
                    fontweight="bold" if c else "normal", color=color)
            if c:
                ax.text(j, i + 0.22, f"{f:.0%}", ha="center", va="center", fontsize=8, color=color)
    col_names = [names[l] if l != "correct" else "Correct /\nnothing" for l in labs]
    row_names = [f"{names[l] if l != 'correct' else 'Correct / none'}  ({sum(r)})" for l, r in zip(labs, counts)]
    ax.set_xticks(range(k), col_names, fontsize=9)
    ax.set_yticks(range(k), row_names, fontsize=9)
    ax.set_xlabel("Model said")
    ax.set_ylabel("Your label")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x + 0.5 for x in range(k - 1)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(k - 1)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    sub = (f"{run_name}  ·  rows sum to 100%; diagonal = agreement\nOne class per note: its label, else its "
           "timing flag. Extras have no score note: a missed extra is Extra -> nothing, a false one is none -> Extra.")
    _title(fig, "What the model said for each note", sub)
    fig.subplots_adjust(left=0.24, right=0.97, top=1 - 2.0 / fig.get_figheight(), bottom=0.04)
    return fig


# ---------------------------------------------------------------- note map
def _outcome(r: dict) -> str | None:
    from eval.metrics import note_class
    g = note_class(r["gold"], set(filter(None, r["gold_flags"].split("|"))))
    p = note_class(r["pred"], set(filter(None, r.get("pred_flags", "").split("|"))))
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
                       label="labeled held/late")]
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
            **{f"confusion_{g}": plot_confusion(metrics, f"{run_dir.name} · {g}", b)
               for g, b in metrics.get("by_group", {}).items()},
            "note_map": plot_note_map(rows, run_dir.name)}
    paths = []
    for name, fig in figs.items():
        paths.append(out / f"{name}.png")
        fig.savefig(paths[-1], dpi=160)
        plt.close(fig)
    return paths


# ---------------------------------------------------------------- one take (v3)
VERDICT_STYLE = {                                       # marker, face, edge
    "correct": ("o", "#ffffff", INK),
    "wrong_pitch": ("s", SERIES[1], INK),
    "missed": ("v", SERIES[2], INK),
}
FLAG_TEXT = {"held_long": "held", "late": "late"}


def plot_take(bundle, an, rep, spec, title: str = "", rows: list[dict] | None = None):
    """v3 decode of one take.

    Top: per-key presence over time (the features v3 reads), with each matched strike marked at the
    pitch the score expects, coloured by the model's verdict; wrong keys and extras at the key played.
    Bottom: the alignment path, score note against time. A drop in the line is a restart.
    rows (per_note.csv rows for this take) adds your labels to the note names on the left; notes
    where you and the model disagree are drawn in orange."""
    import numpy as np
    from score_models import midi_to_note_name
    spf, off = an.seconds_per_frame, bundle.audio.trim_offset_seconds
    t = lambda f: f * spf + off
    by_beat: dict[float, list] = {}
    for n in spec.notes():
        by_beat.setdefault(round(n.start_beat, 4), []).append(n)
    names = ["+".join(f"{n.id} {n.name}" for n in by_beat.get(round(p.beat, 4), [])) for p in an.positions]
    gold = {}
    for r in rows or []:
        g = r["gold"] if r["gold"] != "correct" else ""
        g = "+".join(x for x in [g, r["gold_flags"]] if x)
        p = r["pred"] if r["pred"] != "correct" else ""
        p = "+".join(x for x in [p, r["pred_flags"]] if x)
        gold[r["note_id"]] = (g, g != p)

    pitches = {p for pos in an.positions for p in pos.pitches}
    pitches |= {v.played for vs in rep.verdicts.values() for v in vs if v.played is not None}
    pitches |= {x[1] for x in rep.extras}
    lo, hi = min(pitches) - 2, max(pitches) + 2
    keys = [k for k in range(lo, hi + 1) if k in bundle.candidate_scores]
    roll = np.vstack([bundle.candidate_scores[k] for k in keys])
    T = roll.shape[1]
    n_pos = len(an.positions)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 3.2 + 0.16 * len(keys) + 0.22 * n_pos), sharex=True,
                                  gridspec_kw={"height_ratios": [len(keys) + 4, n_pos + 4], "hspace": 0.12})
    ax.imshow(roll, aspect="auto", origin="lower", cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP),
              vmin=0, vmax=max(0.3, float(np.percentile(roll, 99.5))),
              extent=(t(0), t(T), keys[0] - 0.5, keys[-1] + 0.5), interpolation="nearest")
    ax.set_yticks(keys, [midi_to_note_name(k) for k in keys], fontsize=7.5)
    for lab, k in zip(ax.get_yticklabels(), keys):
        if k in {p for pos in an.positions for p in pos.pitches}:
            lab.set_fontweight("bold"); lab.set_color(INK)
    for e in an.events:                                      # every detected strike, matched or not
        ax.axvline(t(e.frame), color=AXIS, linewidth=0.6, zorder=1, alpha=0.8)
    for pi, vs in rep.verdicts.items():
        for v in vs:
            if v.frame is None:
                continue
            m, face, edge = VERDICT_STYLE[v.label]
            ax.plot(t(v.frame), v.pitch, marker=m, markersize=6, mfc=face, mec=edge, mew=0.9, zorder=4)
            if v.label == "wrong_pitch" and v.played is not None:
                ax.plot(t(v.frame), v.played, marker="s", markersize=6, mfc="none", mec=SERIES[1], mew=1.6, zorder=4)
            for f in sorted(v.flags):
                ax.text(t(v.frame), v.pitch + 0.9, FLAG_TEXT.get(f, f), fontsize=7.5, color=SERIES[1],
                        fontweight="bold", ha="center", zorder=5)
    for _, pitch, frame in rep.extras:
        ax.plot(t(frame), pitch, marker="X", markersize=8, mfc=SERIES[1], mec=INK, mew=0.7, zorder=4)
    for pi, evs in rep.visits.items():                    # earlier passes over a note (before a restart)
        for e in evs[:-1]:
            for q in an.positions[pi].pitches:
                ax.plot(t(an.events[e].frame), q, marker="o", markersize=5, mfc="none", mec=INK_2, mew=0.9, zorder=4)
    ax.set_ylabel("key")
    ax.tick_params(length=0)
    handles = [Line2D([], [], marker=VERDICT_STYLE["correct"][0], linestyle="none", mfc="#ffffff", mec=INK, label="correct"),
               Line2D([], [], marker="s", linestyle="none", mfc=SERIES[1], mec=INK, label="wrong pitch (hollow = key played)"),
               Line2D([], [], marker="v", linestyle="none", mfc=SERIES[2], mec=INK, label="missed"),
               Line2D([], [], marker="X", linestyle="none", mfc=SERIES[1], mec=INK, markersize=8, label="extra"),
               Line2D([], [], marker="o", linestyle="none", mfc="none", mec=INK_2, label="earlier pass (before a restart)"),
               Line2D([], [], color=AXIS, label="detected strike")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6, frameon=False, fontsize=8.5)

    # alignment path
    matched = sorted(an.matches, key=lambda m: m.event)
    xs = [t(an.events[m.event].frame) for m in matched]
    ys = [m.position for m in matched]
    ax2.plot(xs, ys, color=SERIES[0], linewidth=1.4, zorder=2, drawstyle="steps-post")
    for m, x, y in zip(matched, xs, ys):
        vs = rep.verdicts[m.position]
        lab = vs[0].label if all(v.label == vs[0].label for v in vs) else "wrong_pitch"
        mk, face, edge = VERDICT_STYLE[lab]
        ax2.plot(x, y, marker=mk, markersize=6, mfc=face, mec=edge, mew=0.9, zorder=4)
    seen = sorted(set(ys))
    for pi in range(n_pos):
        if pi in seen or not xs:
            continue
        before = [x for x, y in zip(xs, ys) if y < pi]
        after = [x for x, y in zip(xs, ys) if y > pi]
        x = (before[-1] + after[0]) / 2 if before and after else (before[-1] if before else after[0])
        ax2.plot(x, pi, marker="v", markersize=6, mfc=SERIES[2], mec=INK, mew=0.9, zorder=4)
    for a, b in rep.restarts:
        x = next(x for m, x in zip(matched, xs) if m.position == b and m.kind == "back")
        ax2.annotate(f"went back to {names[b].split()[0]}", (x, b), xytext=(-8, 0), textcoords="offset points",
                     fontsize=8, color=INK, ha="right", va="center", fontweight="bold")
    for _, pitch, frame in rep.extras:
        ax2.axvline(t(frame), color=SERIES[1], linewidth=1, linestyle=":", zorder=1)
    labels = []
    for pi, nm in enumerate(names):
        ids = [s.split()[0] for s in nm.split("+")]
        g = [gold[i][0] for i in ids if i in gold and gold[i][0]]
        labels.append(nm + (f"  · {', '.join(g)}" if g else ""))
    ax2.set_yticks(range(n_pos), labels, fontsize=7.5)
    for lab, nm in zip(ax2.get_yticklabels(), names):
        ids = [s.split()[0] for s in nm.split("+")]
        if any(gold.get(i, ("", False))[1] for i in ids):
            lab.set_color(SERIES[1]); lab.set_fontweight("bold")
    ax2.set_ylim(n_pos - 0.5, -0.5)
    ax2.set_xlim(t(0), t(T))
    ax2.set_xlabel("seconds")
    ax2.set_ylabel("score note" + ("  (· your label; orange = disagrees)" if rows else ""))
    _clean_axes(ax2, grid_axis="y")
    _title(fig, title or "v3 alignment",
           "Top: key presence and the model's verdicts.  Bottom: which score note each strike was matched to.")
    fig.subplots_adjust(left=0.13, right=0.99, top=1 - 1.25 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
    return fig


# ---------------------------------------------------------------- tempo over time
def plot_tempo(rows: list[dict], spec, title: str = ""):
    """Tempo over time for one take (rows = that take's per_note.csv rows).

    Thin line: tempo between each pair of consecutive notes (per beat).  Thick steps: tempo per
    measure, drawn across the seconds that measure took.  Dashed: the take's average tempo, so a
    steady player stays near it.  Uses the model's onsets (see eval/tempo.py)."""
    from eval.tempo import beat_tempo, measure_tempo, steadiness
    beats, measures = beat_tempo(rows, spec), measure_tempo(rows, spec)
    st = steadiness(beats)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    if not beats:
        ax.text(0.5, 0.5, "no onsets for this take", ha="center", transform=ax.transAxes, color=MUTED)
        return fig
    ax.plot([p.t_mid for p in beats], [p.bpm for p in beats], color=SERIES[0], linewidth=1.2, alpha=0.75,
            marker="o", markersize=4, mfc=SURFACE, mec=SERIES[0], mew=1.2, zorder=3, label="per beat (note to note)")
    for i, p in enumerate(measures):
        ax.plot([p.t_start, p.t_end], [p.bpm, p.bpm], color=SERIES[1], linewidth=3, solid_capstyle="round",
                zorder=4, label="per measure" if i == 0 else None)
        ax.axvline(p.t_start, color=GRID, linewidth=0.8, zorder=1)
        ax.text(p.t_start, 1.0, f" {p.label}", transform=ax.get_xaxis_transform(), fontsize=8, color=MUTED,
                va="bottom", ha="left")
    ax.axhline(st["mean_bpm"], color=INK_2, linewidth=1, linestyle="--", zorder=2,
               label=f"average {st['mean_bpm']:.0f} BPM")
    slow = min(beats, key=lambda p: p.bpm)
    fast = max(beats, key=lambda p: p.bpm)
    for p, va, dy in ((slow, "top", -8), (fast, "bottom", 8)):
        note = "  (pause or restart?)" if p is slow and p.bpm < 0.5 * st["mean_bpm"] else ""
        ax.annotate(f"{p.label}  {p.bpm:.0f}{note}", (p.t_mid, p.bpm), xytext=(0, dy), textcoords="offset points",
                    fontsize=8, color=INK_2, ha="center", va=va)
    lo, hi = min(p.bpm for p in beats), max(p.bpm for p in beats)
    pad = max(10, 0.12 * (hi - lo))
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xlabel("seconds")
    ax.set_ylabel("tempo (BPM, quarter-note beats)")
    _clean_axes(ax, grid_axis="y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3, frameon=False, fontsize=9)
    _title(fig, title or "Tempo over time",
           f"Variation {st['cv_pct']:.0f}% (std / mean, per beat).  Score tempo {spec.bpm:.0f} BPM for reference.")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.70, bottom=0.13)
    return fig


def plot_tempo_piece(rows: list[dict], spec, title: str = ""):
    """Note-to-note tempo for every take of one piece, x = position in the score, so takes line up.
    Tempo is relative to each take's own average (100% = that take's average), so fast and slow
    players are comparable.  Thin gray: one take each.  Blue: median across takes, with the band
    holding the middle half of takes.  A dip everyone shares is a spot players slow down at."""
    import numpy as np
    from eval.tempo import beat_tempo, steadiness
    by_take: dict[str, list[dict]] = {}
    for r in rows:
        if r["piece"] == spec.name:
            by_take.setdefault(r["take"], []).append(r)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    per_x: dict[float, list[float]] = {}
    names: dict[float, str] = {}
    for rs in by_take.values():
        pts = beat_tempo(rs, spec)
        if not pts:
            continue
        mean = steadiness(pts)["mean_bpm"]
        xs = [p.beat_start for p in pts]
        ys = [100 * p.bpm / mean for p in pts]
        ax.plot(xs, ys, color=AXIS, linewidth=0.8, alpha=0.6, zorder=2)
        for p, x, y in zip(pts, xs, ys):
            per_x.setdefault(x, []).append(y)
            names.setdefault(x, p.label.split("→")[0])
    xs = sorted(x for x in per_x if len(per_x[x]) >= 3)
    if xs:
        q = np.array([np.percentile(per_x[x], [25, 50, 75]) for x in xs])
        ax.fill_between(xs, q[:, 0], q[:, 2], color=SERIES[0], alpha=0.18, linewidth=0, zorder=3,
                        label="middle half of takes")
        ax.plot(xs, q[:, 1], color=SERIES[0], linewidth=2.2, marker="o", markersize=5, mfc=SERIES[0],
                mec=SURFACE, mew=1.2, zorder=4, label="median across takes")
        ax.set_xticks(xs, [names[x] for x in xs], fontsize=8)
        lo = int(np.argmin(q[:, 1]))
        ax.annotate(f"slowest: after {names[xs[lo]]}  {q[lo, 1]:.0f}%", (xs[lo], q[lo, 1]), xytext=(0, -14),
                    textcoords="offset points", fontsize=8.5, color=INK, va="top",
                    ha="right" if lo == len(xs) - 1 else "left" if lo == 0 else "center")
        bpm_ = spec.beats_per_measure
        for b in np.arange(spec.pickup_beats, max(xs) + 1e-6, bpm_)[1:]:
            ax.axvline(b - 0.5, color=GRID, linewidth=0.8, zorder=1)
        allv = np.concatenate([q[:, 0], q[:, 2]])
        ax.set_ylim(max(0, allv.min() - 25), allv.max() + 25)
    ax.axhline(100, color=INK_2, linewidth=1, linestyle="--", zorder=3, label="take's own average")
    ax.plot([], [], color=AXIS, linewidth=1, label="one take")
    ax.set_xlabel("tempo from this note to the next (score order; faint lines = bar lines)")
    ax.set_ylabel("tempo, % of take average")
    _clean_axes(ax, grid_axis="y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, frameon=False, fontsize=9)
    _title(fig, title or f"Tempo through the piece · {spec.name}",
           f"{len(by_take)} takes.  Below 100% = slower than that player's average at this point.")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.72, bottom=0.14)
    return fig
