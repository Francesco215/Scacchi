"""Six-panel summary of the loss-pool experiments (hex7 and hex8 run histories).

    .venv/bin/python scripts/plot_loss_pool_summary.py --out outputs/loss_pool/summary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/loss_pool")


def load(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def evals(rows, x="train/frames"):
    """Win rate against what produced the evaluated network (the loop evaluates before it trains)."""
    points = [(rows[i - 1].get(x, rows[i - 1]["train/frames"]), r["eval/vs_baseline/win_rate"]) for i, r in enumerate(rows) if i and "eval/vs_baseline/win_rate" in r]
    return np.asarray(points).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    (a, b, c), (d, e, f) = axes
    quality = load("h8/quality.jsonl")
    nows = sorted({r["now_updates"] for r in quality})
    shade = lambda i, n: plt.get_cmap("viridis")(i / max(n - 1, 1))

    for i, now in enumerate(nows):
        row = next(r for r in quality if r["now_updates"] == now and r["age_updates"] == 0)
        a.plot(row["by_bits/fresh/x"], row["by_bits/fresh/use"], marker="o", color=shade(i, len(nows)), label=f"after {now} updates")
    a.set_xlabel("bits of a fresh row (octile means)")
    a.set_ylabel("signal a second search reproduces")
    a.set_title("1. A fresh row's usable signal grows with its bits", fontsize=11)
    a.legend(fontsize=8)

    for i, now in enumerate(nows):
        rows = sorted((r for r in quality if r["now_updates"] == now and 0 < r["age_updates"] < now), key=lambda r: r["age_updates"])
        b.plot([r["age_updates"] for r in rows], [r["worth"] for r in rows], marker="o", color=shade(i, len(nows)), label=f"after {now} updates")
    b.axhline(1.0, color="gray", linestyle=":")
    b.axhline(0.0, color="gray")
    b.set_xscale("log")
    b.set_xlabel("age of the stored target (gradient updates)")
    b.set_ylabel("worth relative to a fresh search")
    b.set_title("2. A stored target loses its worth with age", fontsize=11)
    b.legend(fontsize=8)

    for label, path, color in (
        ("no pool", "h7/ref.jsonl", "black"),
        ("pool, 1.5 replays per row", "h7/fifo_r2.jsonl", "tab:green"),
        ("pool, 4 replays per row", "h7/fifo_r4.jsonl", "tab:orange"),
        ("pool, 10 replays per row", "h7/fifo_m4.jsonl", "tab:red"),
    ):
        x, y = evals(load(path))
        c.plot(x / 1e6, y, marker="o", markersize=3, color=color, label=label)
    c.set_xlabel("self-play frames (millions), hex7")
    c.set_ylabel("win rate against the solved network")
    c.set_title("3. Replaying a target: early speed, lower ceiling", fontsize=11)
    c.legend(fontsize=8)

    for label, path, color in (
        ("no pool", "h8/ref.jsonl", "black"),
        ("recycle targets", "h8/fifo_r2.jsonl", "tab:green"),
        ("re-search uniform rows", "h8/research_uniform.jsonl", "tab:purple"),
        ("re-search highest-bit rows", "h8/research_bits.jsonl", "tab:blue"),
    ):
        x, y = evals(load(path), "train/searches")
        d.plot(x / 1e6, y, marker="o", markersize=3, color=color, label=label)
    d.set_xlim(0, 8)
    d.set_xlabel("total searches (millions), hex8")
    d.set_ylabel("win rate against the solved network")
    d.set_title("4. Re-search pays only on high-bit rows", fontsize=11)
    d.legend(fontsize=8)

    for label, path, color, style in (
        ("uniform selection, train on whole pool", "h8/big_uniform.jsonl", "tab:purple", "-"),
        ("bits selection, train on whole pool", "h8/big_sampled_tau1.jsonl", "tab:blue", "-"),
        ("bits selection, train on recent targets", "h8/big_sampled_tau1_young.jsonl", "tab:blue", "--"),
        ("no pool", "h8/ref.jsonl", "black", "-"),
    ):
        x, y = evals(load(path), "train/searches")
        e.plot(x / 1e6, y, marker="o", markersize=3, color=color, linestyle=style, label=label)
    e.set_xlim(0, 8)
    e.set_xlabel("total searches (millions), hex8, 1M-board pool")
    e.set_ylabel("win rate against the solved network")
    e.set_title("5. A big pool: bits help, old targets hurt", fontsize=11)
    e.legend(fontsize=8)

    labels, before, after, pool = [], [], [], []
    for label, path in (("131k pool\nrows ~125 updates old", "h8/research_bits.jsonl"), ("1M pool\nrows ~2,000 updates old", "h8/reservoir_1024.jsonl")):
        rows = [r for r in load(path) if "research/bits_before" in r]
        rows = rows[len(rows) // 2 :]
        labels.append(label)
        before.append(np.mean([r["research/bits_before"] for r in rows]))
        after.append(np.mean([r["research/bits_after"] for r in rows]))
        pool.append(np.mean([r["fresh/bits"] for r in rows]))
    position = np.arange(len(labels))
    f.bar(position - 0.27, before, 0.27, color="tab:red", label="against the stored target")
    f.bar(position, after, 0.27, color="tab:blue", label="against the new target")
    f.bar(position + 0.27, pool, 0.27, color="lightgray", label="an average fresh row")
    f.set_xticks(position, labels)
    f.set_ylabel("bits of the rows picked for re-search")
    f.set_title("6. High bits mean a hard position only while the target is recent", fontsize=11)
    f.legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    for axis in (c, d, e):
        axis.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        axis.set_ylim(-0.02, 0.7)
    figure.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
