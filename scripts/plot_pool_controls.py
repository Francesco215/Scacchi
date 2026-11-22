"""The replay pool against matched no-pool controls, Hex 8 and Hex 10.

Each pair differs only in the pool: same code, int8 network, fused search,
outcome losses off, 1,024 self-play games per iteration. The pool runs add 256
re-search lanes, which count as searched positions.

  Hex 8   tnj90uci  pool          4lzxy4zi  no pool
  Hex 10  hbwhn2fy  pool          z2l7fxid  no pool (stopped at iteration 287, resumed from its
                                  iteration-250 checkpoint as pxwf6251)

Hex 8 also shows q9wqllha (no pool, 4,096 games per iteration, XLA network) on
the positions axis only: it is the run the pool was first compared with, and
the gap to it is the iteration size, not the pool. It is left off the
wall-clock axis because its network kernel differs.

Evals: 512 games against a fixed reference (8_solved, 10_reference). Dots are
raw evals with 95% binomial intervals; lines are a Gaussian-weighted mean in
log x (0.08 decades). ``train/hours`` counts training only, not evaluation.

    python scripts/plot_pool_controls.py reports/loss_pool/pool_vs_control
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

H = Path(__file__).resolve().parents[1] / "outputs/loss_pool"
POOL, CTRL, BIG = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
EVAL_GAMES = 512
SIGMA = 0.08  # smoothing width, decades of x

PANELS = [
    ("Hex 8", "vs 8_solved", {"positions": (5e5, 2e7), "minutes": (0.5, 11)}, [
        ("tnj90uci", "pool, 1,024 games + 256 re-search lanes", H / "h8/research_bits_int8.jsonl", POOL, True),
        ("4lzxy4zi", "no pool, 1,024 games", H / "h8/nopool_1024_int8.jsonl", CTRL, True),
        ("q9wqllha", "no pool, 4,096 games (XLA network)", H / "h8/ref.jsonl", BIG, False),
    ]),
    ("Hex 10", "vs 10_reference", {"positions": (2e6, 1e8), "minutes": (2, 70)}, [
        ("hbwhn2fy", "pool, 1,024 games + 256 re-search lanes", H / "h10/pool_int8_h10.jsonl", POOL, True),
        ("z2l7fxid", "no pool, 1,024 games", H / "h10/nopool_int8_h10_1024.merged.jsonl", CTRL, True),
    ]),
]
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)


def load(path):
    rows = [json.loads(line) for line in open(path)]
    ev = [r for r in rows if "eval/vs_baseline/win_rate" in r and r["train/frames"] > 0]
    return {
        "positions": np.array([r.get("train/searches", r["train/frames"]) for r in ev], float),
        "minutes": np.array([r["train/hours"] * 60 for r in ev]),
        "win": np.array([r["eval/vs_baseline/win_rate"] for r in ev]),
        "end_minutes": rows[-1]["train/hours"] * 60,
    }


def smooth(x, y):
    lx = np.log10(x)
    grid = np.linspace(lx.min(), lx.max(), 300)
    w = np.exp(-0.5 * ((grid[:, None] - lx[None, :]) / SIGMA) ** 2)
    return 10**grid, (w @ y) / w.sum(axis=1)


def first_crossing(x, y, level):
    above = np.nonzero(y >= level)[0]
    return None if len(above) == 0 else float(x[above[0]])


def style(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.01, 0.68)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5)


def main(out: str) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    crossings = {}
    for row, (board, opponent, limits, runs) in enumerate(PANELS):
        for col, key in enumerate(("positions", "minutes")):
            ax = axes[row, col]
            style(ax)
            for run, label, path, color, on_clock in runs:
                if key == "minutes" and not on_clock:
                    continue
                d = load(path)
                x, y = d[key], d["win"]
                err = 1.96 * np.sqrt(np.clip(y * (1 - y), 1e-4, None) / EVAL_GAMES)
                ax.errorbar(x, y, yerr=err, fmt="o", ms=2.6, color=color, alpha=0.35, elinewidth=0.7, capsize=0, zorder=2)
                sx, sy = smooth(x, y)
                ax.plot(sx, sy, color=color, lw=2, zorder=3, label=f"{label}  ({run})")
                crossings.setdefault((board, run, label), {})[key] = [first_crossing(sx, sy, lv) for lv in LEVELS]
            ax.set_xlim(*limits[key])
            if key == "positions":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
                ax.set_xlabel("positions searched (self-play + re-search), log scale", color=INK, fontsize=9.5)
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.set_xlabel("training wall-clock, minutes, log scale", color=INK, fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{board}: win rate {opponent}", color=INK, fontsize=9.5)
            ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    axes[0, 0].set_title("Against search work", loc="left", color=INK, fontsize=11.5)
    axes[0, 1].set_title("Against wall-clock (same kernels)", loc="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")

    # Smoothed first crossing of each level: positions searched, then minutes.
    for key, scale, unit in (("positions", 1e6, "M"), ("minutes", 1, " min")):
        print(f"\n{key} to reach each win rate")
        print("| Board | Run | " + " | ".join(f"{int(lv * 100)}%" for lv in LEVELS) + " |")
        print("|---|---|" + "---|" * len(LEVELS))
        for (board, run, label), c in crossings.items():
            if key in c:
                print(f"| {board} | {label} (`{run}`) | " + " | ".join("-" if v is None else f"{v / scale:.1f}{unit}" for v in c[key]) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/loss_pool/pool_vs_control")
