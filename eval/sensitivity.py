"""
sensitivity.py  --  python -m eval.sensitivity
How much do the v3 scores depend on each hand-set number? Every listed setting is moved by
-25% and +25% (one at a time) and the scoreboard is rebuilt. A setting whose +-25% leaves the
scores where they are was not fitted to these recordings; one that moves them is a risk.
"""
from __future__ import annotations
import sys

import numpy as np

from eval import check
from eval.adapters import get_adapter

SETTINGS = ("onset_floor", "onset_full", "match_min_salience", "sub_min_salience", "extra_report_salience",
            "ghost_ratio", "smudge_ratio", "sub_cost", "skip_cost", "extra_cost", "timing_weight",
            "slow_ratio", "back_cost_bar", "raw_rise_db")


def row(board: dict) -> tuple:
    r = board["real"]
    synth_exact = float(np.mean([b["exact"] for b in board["synth"].values()])) if board["synth"] else float("nan")
    return (r["mono"]["goal"]["f1"], r["mono"]["strict"]["f1"], r["mono"]["clean"], r["all"]["strict"]["f1"],
            r["all"]["clean"], synth_exact, board["crashes"])


def main() -> int:
    adapter = get_adapter("v3")
    base_cfg = adapter.config()
    head = f"{'setting':<24}{'value':>8} | mono goal  mono strict  mono clean | all strict  all clean | synth exact | crashes"
    print(head)
    fmt = lambda name, val, r: (f"{name:<24}{val:>8.3g} |   {r[0]:.2f}       {r[1]:.2f}        {r[2]:>2}     |"
                                f"    {r[3]:.2f}       {r[4]:>2}     |    {r[5]:.3f}    | {r[6]}")
    print(fmt("(as shipped)", 0, row(check.build(adapter))), flush=True)
    for name in (sys.argv[1:] or SETTINGS):
        for mult in (0.75, 1.25):
            val = base_cfg[name] * mult
            print(fmt(name, val, row(check.build(adapter, overrides={name: val}))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
