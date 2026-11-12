"""Hex 8: the replay pool with re-search, on the int8 megakernel, against the baselines.

Runs (all evaluated against checkpoints/8_solved, Gumbel 32 simulations, 512
games; ``train/hours`` counts training iterations only, not evaluation):

  riemy292  binned_8 before the fused search (5395774), 4096 games/iteration,
            outcome losses on (value 1, Q 0.2)
  q9wqllha  same code as the pool runs (c7d67f1), no pool, 4096 games,
            XLA network, fused search, outcome losses off
  dimvaohv  pool + 256 re-search lanes, 1024 games, XLA network
  tnj90uci  dimvaohv's config with network_kernel=int8 (ae0ca7f)

Eval win rates are noisy (512 games: about +-2.2 points), so each curve is a
Gaussian-weighted mean of the eval points in log time; dots are the raw evals.

    python scripts/plot_pool_megakernel.py reports/loss_pool/pool_megakernel.png
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

WT = Path(__file__).resolve().parents[2]  # .claude/worktrees
RUNS = [  # fixed categorical order: slot 1 is the headline run
    ("tnj90uci", "pool + int8 megakernel", WT / "megakernel/outputs/loss_pool/h8/research_bits_int8.jsonl", "#2a78d6"),
    ("dimvaohv", "pool, XLA network", WT / "megakernel/outputs/loss_pool/h8/research_bits.jsonl", "#eb6834"),
    ("q9wqllha", "no pool, XLA network", WT / "megakernel/outputs/loss_pool/h8/ref.jsonl", "#1baf7a"),
    ("riemy292", "binned_8 (before fused search)", WT / "binned-weighted-sum/outputs/weighted_sum_scaling/binned_8.jsonl", "#eda100"),
]
TARGET = 0.5
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
SIGMA = 0.10  # smoothing width, decades of the x axis


def load(path):
    rows = [json.loads(line) for line in open(path)]
    evals = [r for r in rows if "eval/vs_baseline/win_rate" in r]
    return {
        "minutes": np.array([r["train/hours"] * 60 for r in evals]),
        # Positions searched: self-play plus re-search lanes (only the pool runs have the latter).
        "frames": np.array([r.get("train/searches", r["train/frames"]) for r in evals], float),
        "win": np.array([r["eval/vs_baseline/win_rate"] for r in evals]),
        "iter_s": float(np.median([r["train/iter_seconds"] for r in rows[2:]])),
        "total_min": rows[-1]["train/hours"] * 60,
        "total_frames": rows[-1].get("train/searches", rows[-1]["train/frames"]),
    }


def smooth(x, y):
    lx = np.log10(x)
    grid = np.linspace(lx.min(), lx.max(), 300)
    w = np.exp(-0.5 * ((grid[:, None] - lx[None, :]) / SIGMA) ** 2)
    return 10 ** grid, (w @ y) / w.sum(axis=1)


def first_crossing(x, y, level):
    above = np.nonzero(y >= level)[0]
    return None if len(above) == 0 else float(x[above[0]])


def style(ax, xlabel):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.axhline(TARGET, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)


def main(out: str) -> None:
    data = {run: load(path) for run, _, path, _ in RUNS}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    summary = {}
    for col, (key, label, fmt) in enumerate([("minutes", "training wall-clock (minutes, log)", None),
                                              ("frames", "positions searched, self-play + re-search (log)", None)]):
        ax = axes[col]
        style(ax, label)
        for run, name, _, color in RUNS:
            d = data[run]
            x, y = d[key], d["win"]
            ax.scatter(x, y, s=10, color=color, alpha=0.35, lw=0, zorder=2)
            sx, sy = smooth(x, y)
            ax.plot(sx, sy, color=color, lw=2, zorder=3, label=f"{name} ({run})")
            cross = first_crossing(sx, sy, TARGET)
            summary.setdefault(run, {})[key] = cross
            if cross is not None:
                ax.annotate("", xy=(cross, TARGET), xytext=(cross, TARGET - 0.13),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4), zorder=4)
        if key == "frames":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.text(ax.get_xlim()[0] * 1.05, TARGET + 0.012, "parity with solved Hex 8 checkpoint", color=MUTED, fontsize=8.5)
    axes[0].set_ylabel("eval win rate vs 8_solved", color=INK)
    axes[0].set_title("Win rate against wall-clock", loc="left", color=INK, fontsize=12)
    axes[1].set_title("Win rate against search work", loc="left", color=INK, fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=INK)

    # Panel 3: minutes to reach the target (smoothed), one bar per run.
    ax = axes[2]
    names = [name for _, name, _, _ in RUNS]
    minutes = [summary[run]["minutes"] for run, _, _, _ in RUNS]
    y = np.arange(len(RUNS))[::-1]
    for yi, (run, name, _, color), m in zip(y, RUNS, minutes):
        if m is None:
            ax.barh(yi, data[run]["total_min"], color=color, alpha=0.25, height=0.55)
            ax.text(data[run]["total_min"] * 1.02, yi, f"not reached in {data[run]['total_min']:.0f} min", va="center", fontsize=9, color=INK)
        else:
            ax.barh(yi, m, color=color, height=0.55)
            ax.text(m * 1.02 + 0.1, yi, f"{m:.1f} min", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [f"{name}\n{run}" for (run, name, _, _) in RUNS], fontsize=8.5, color=INK)
    ax.set_xlabel("minutes of training to reach 50% (smoothed curve)", color=INK)
    ax.set_title("Time to parity", loc="left", color=INK, fontsize=12)
    ax.set_xlim(0, max(m if m is not None else data[r]["total_min"] for (r, *_), m in zip(RUNS, minutes)) * 1.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    for run, name, _, _ in RUNS:
        d = data[run]
        late = d["win"][d["minutes"] >= 0.75 * d["total_min"]]
        print(f"{run:9s} {name:32s} s/it {d['iter_s']:5.2f}  total {d['total_min']:5.1f} min  {d['total_frames'] / 1e6:5.1f}M frames  "
              f"to 50%: {summary[run]['minutes']} min, {summary[run]['frames']} frames  last-quarter mean {late.mean():.3f} (n={len(late)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/loss_pool/pool_megakernel.png")
