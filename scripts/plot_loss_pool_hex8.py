"""Win rate against total searches for the hex8 loss-pool runs, in two panels.

    .venv/bin/python scripts/plot_loss_pool_hex8.py --out reports/loss_pool/hex8_runs.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/loss_pool/h8")
PANELS = [
    (
        "1,024 games per call (64-move windows)",
        [
            ("no pool (4,096 games)", "ref", "black", "-"),
            ("recycle, half fresh", "fifo_r2", "tab:green", "-"),
            ("recycle, quarter fresh", "fifo_r4", "tab:olive", "-"),
            ("+ 256 re-search lanes, uniform rows", "research_uniform", "tab:purple", "-"),
            ("+ 256 re-search lanes, highest-bit rows", "research_bits", "tab:blue", "-"),
            ("+ 256 sampled lanes, 1M pool, recent targets", "reservoir_1024", "tab:blue", "--"),
        ],
    ),
    (
        "256 games per call, 1/8 fresh, sampled re-search, evict by low bits",
        [
            ("no pool (4,096 games)", "ref", "black", "-"),
            ("768 lanes, 64-move windows", "small_sampled_tau1", "tab:orange", "-"),
            ("256 lanes", "lanes_256", "tab:red", "-"),
            ("64 lanes", "lanes_64", "tab:brown", "-"),
            ("768 lanes, 16-move windows from pool positions", "pool_starts_w16", "tab:cyan", "-"),
            ("same at 4x width (1,024 games, 3,072 lanes)", "pool_starts_w16_x4", "tab:cyan", "--"),
            ("1M pool, 768 lanes, train on whole pool", "big_sampled_tau1", "tab:gray", "-"),
            ("1M pool, 768 lanes, train on recent targets", "big_sampled_tau1_young", "tab:gray", "--"),
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    figure, axes = plt.subplots(1, 2, figsize=(17, 6))
    for axis, (title, runs) in zip(axes, PANELS):
        for label, name, color, style in runs:
            rows = [json.loads(line) for line in (ROOT / f"{name}.jsonl").read_text().splitlines() if line.strip()]
            searches = lambda row: row.get("train/searches", row["train/frames"])
            # The loop evaluates before it trains: row i's eval is of the network after row i - 1.
            points = [(searches(rows[i - 1]) / 1e6, row["eval/vs_baseline/win_rate"]) for i, row in enumerate(rows) if i and "eval/vs_baseline/win_rate" in row]
            x, y = np.asarray(points).T
            axis.plot(x, y, marker="o", markersize=3, color=color, linestyle=style, label=label)
        axis.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        axis.set_xlim(0, 10)
        axis.set_ylim(-0.02, 0.7)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("total searches (millions): self-play boards plus re-searched pool boards")
        axis.set_ylabel("win rate against the solved network")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
