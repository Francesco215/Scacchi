"""Figures for the int8 search-network report (PR #39).

Two figures: where the speed comes from (tensor-core ceilings, network call,
full iteration), and whether training is unchanged (win rate against
wall-clock time, value NLL against frames, error against the bf16 network).
The bar values are the measurements quoted in the PR (RTX 5090, Hex 7, batch
4096, trained binned_7 checkpoint); the curves are read from the run
histories given on the command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_time_curves import load, series  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
INT8, FP8, BF16, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def arrow(ax, start, end, text, text_xy, color=INK, **kwargs):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2), **kwargs)
    ax.text(*text_xy, text, color=color, fontsize=10.5, fontweight="bold", ha="center", va="center", **kwargs)


def hbars(ax, labels, values, colors, unit, fmt="{:.2f}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v, yi, "  " + fmt.format(v) + unit, va="center", ha="left", color=INK)
    ax.set_xlim(0, max(values) * 1.32)
    return y


def speed_figure(out):
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.3]})

    ax = axes[0]
    y = hbars(ax, ["bf16  (m16n8k16)", "FP8 e4m3  (m16n8k32)", "int8  (m16n8k32)"], [248, 486, 982], [BF16, FP8, INT8], " T", "{:.0f}")
    ax.set(title="Tensor-core ceiling, measured", xlabel="multiply-add operations per second (×10¹²), mma only")
    arrow(ax, (900, y[1]), (900, y[2] + 0.38), "2.0× the FP8 rate", (1010, y[1] + 0.22))

    ax = axes[1]
    values = [5.91, 3.95, 2.27, 1.54]
    y = hbars(ax, ["XLA / cuDNN  bf16", "CUDA kernels  bf16", "CUDA kernels  FP8", "CUDA kernels  int8"], values, [NEUTRAL, BF16, FP8, INT8], " ms")
    for yi, v in zip(y[1:], values[1:]):
        ax.text(max(values) * 1.5, yi, f"{values[0] / v:.1f}×", va="center", ha="right", color=MUTED, fontweight="bold")
    ax.text(max(values) * 1.5, y[0] + 0.48, "speed-up", va="center", ha="right", color=MUTED, fontsize=9.5)
    ax.text(max(values) * 1.5, y[0], "1.0×", va="center", ha="right", color=MUTED, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.52)
    ax.set(title="One network call, 4096 positions", xlabel="milliseconds (12 residual convolutions, stem, heads)")

    ax = axes[2]
    steps = [("original", 11.52, NEUTRAL), ("+ concurrent\nbuffers", 10.30, NEUTRAL), ("+ fused\nsearch", 9.24, NEUTRAL),
             ("+ CUDA\nnetwork", 8.91, BF16), ("+ FP8\nnetwork", 5.75, FP8), ("+ int8\nnetwork", 4.73, INT8)]
    x = np.arange(len(steps))
    ax.bar(x, [s[1] for s in steps], color=[s[2] for s in steps], width=0.62)
    ax.set_xticks(x, [s[0] for s in steps], color=INK, fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    for xi, (_, v, _) in zip(x, steps):
        ax.text(xi, v + 0.15, f"{v:.2f} s", ha="center", va="bottom", color=INK)
    ax.set(title="Full training iteration (self-play, training, eval)", ylabel="seconds", ylim=(0, 14.2))
    arrow(ax, (5.0, 11.6), (5.0, 5.75), "2.4× faster\nthan the original", (4.25, 12.7))
    fig.suptitle("Where the speed comes from   ·   Hex 7, RTX 5090", x=0.008, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5)
    fig.savefig(out, dpi=140)
    print("wrote", out)


def crossing(xy, level):
    """First time the curve reaches `level` (linear interpolation)."""
    for (t0, v0), (t1, v1) in zip(xy[:-1], xy[1:]):
        if v0 < level <= v1:
            return t0 + (t1 - t0) * (level - v0) / (v1 - v0)
    return None


def learning_figure(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1.1, 0.9]})
    colors = {"int8": INT8, "fp8": FP8, "bf16": BF16}
    names = {"int8": "int8 search", "fp8": "FP8 search", "bf16": "bf16 search (reference)"}

    ax = axes[0]
    times = {}
    for label, rows in runs.items():
        key = next(k for row in rows for k in row if k.endswith("win_rate"))
        xy = series(rows, key)
        ax.plot(xy[:, 0], xy[:, 1], color=colors[label], lw=2)
        times[label] = crossing(xy, 0.5)
    ax.set_xscale("log")
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(ax.get_xlim()[1] * 0.93, 0.455, "parity with the\nsolved-game baseline", color=MUTED, fontsize=9.5, va="top", ha="right")
    ax.set(title="Strength against wall-clock time", xlabel="training time, seconds (log scale)", ylabel="eval win rate vs baseline", ylim=(0, 0.72))
    for label, t in times.items():
        ax.plot([t], [0.5], "o", color=colors[label], ms=7, mec=SURFACE, mew=2, zorder=5)
        left = label == "int8"
        ax.text(t * (0.94 if left else 1.05), 0.5 + (0.03 if left else -0.03), f"{t:.0f} s", color=INK, ha="right" if left else "left",
                va="bottom" if left else "top", fontsize=9.5)
    arrow(ax, (times["bf16"], 0.30), (times["int8"], 0.30), f"{times['bf16'] / times['int8']:.1f}× sooner",
          (np.sqrt(times["bf16"] * times["int8"]), 0.335))
    ax.text(times["bf16"] * 1.12, 0.14, names["bf16"], color=INK, fontsize=9.5, ha="left")
    ax.text(times["int8"] * 0.72, 0.2, "int8 and FP8\nsearch", color=INK, fontsize=9.5, ha="right")

    ax = axes[1]
    for label, rows in runs.items():
        xy = np.array([(float(r["train/frames"]), float(r["train/value_binned_loss"])) for r in rows
                       if r.get("train/value_binned_loss") is not None and r.get("train/frames") is not None])
        ax.plot(xy[:, 0] / 1e6, xy[:, 1], color=colors[label], lw=1.8, label=names[label])
    ax.set_yscale("log")
    ax.set_yticks([0.65, 0.7, 0.8, 0.9, 1.0], ["0.65", "0.7", "0.8", "0.9", "1.0"])
    ax.minorticks_off()
    ax.set(title="Same learning per frame", xlabel="self-play frames (millions)", ylabel="value NLL of the search target (log scale)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.annotate("the three runs overlap:\n8-bit search does not\nchange what is learned", xy=(8.5, 0.69), xytext=(9.5, 0.86), color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))

    ax = axes[2]
    plies = ["ply 0", "ply 10", "ply 25"]
    fp8, int8 = [0.371, 0.1225, 0.0564], [0.042, 0.043, 0.025]
    x = np.arange(3)
    ax.bar(x - 0.19, fp8, width=0.34, color=FP8, label="FP8")
    ax.bar(x + 0.19, int8, width=0.34, color=INT8, label="int8")
    ax.set_yscale("log")
    ax.set_ylim(0.015, 0.9)
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"])
    ax.minorticks_off()
    ax.set_xticks(x, plies, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    for xi, a, b in zip(x, fp8, int8):
        ax.text(xi - 0.19, a * 1.07, f"{a:.2f}", ha="center", va="bottom", color=INK, fontsize=9.5)
        ax.text(xi + 0.19, b * 1.07, f"{b:.2f}", ha="center", va="bottom", color=INK, fontsize=9.5)
    arrow(ax, (0.19, 0.3), (0.19, 0.062), "9× closer", (0.72, 0.3))
    ax.set(title="Error against the bf16 network", ylabel="mean |logit difference| (log scale)")
    ax.legend(loc="upper right", fontsize=9.5)
    fig.suptitle("Training is unchanged, only faster   ·   Hex 7, search network in 8 bits, training in bf16", x=0.008, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.5)
    fig.savefig(out, dpi=140)
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--fp8", required=True)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    speed_figure(args.out_dir / "int8_speed.png")
    learning_figure({"bf16": load(args.bf16), "fp8": load(args.fp8), "int8": load(args.int8)}, args.out_dir / "int8_learning.png")


if __name__ == "__main__":
    main()
