"""Figure for the output of scripts/pool_target_quality.py.

    .venv/bin/python scripts/plot_pool_target_quality.py outputs/loss_pool/h7/quality.jsonl --out outputs/loss_pool/h7/quality.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.source).read_text().splitlines() if line.strip()]
    nows = sorted({r["now_updates"] for r in records})
    colors = {now: plt.get_cmap("viridis")(i / max(len(nows) - 1, 1)) for i, now in enumerate(nows)}
    by_now = {now: sorted((r for r in records if r["now_updates"] == now), key=lambda r: r["age_updates"]) for now in nows}
    step = min(r["age_updates"] for r in records if r["age_updates"] > 0)
    shown_age = lambda age: age if age > 0 else step / 2  # age 0 drawn left of the first step on the log axis

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    (worth, wrong, share_time), (use_bits, share_bits, wrong_bits) = axes

    for now, rows in by_now.items():
        # A target as old as the run was made by the untrained network; leave it out.
        rows = [r for r in rows if r["age_updates"] < now]
        ages = [shown_age(r["age_updates"]) for r in rows]
        worth.plot(ages, [r["worth"] for r in rows], marker="o", color=colors[now], label=f"network after {now} updates")
        wrong.plot(ages, [r["wrong_way_stored"] for r in rows], marker="o", color=colors[now])
        wrong.axhline(rows[0]["wrong_way_fresh"], color=colors[now], linestyle=":", linewidth=1)
    for axis, title in ((worth, "worth of a stored target relative to a fresh search"), (wrong, "share of stored targets pulling the wrong way")):
        axis.set_xscale("log")
        axis.set_xlabel("age of the target (gradient updates); leftmost point is age 0")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.2)
    worth.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    worth.axhline(0.0, color="gray", linewidth=1)
    worth.legend(fontsize=7)
    wrong.axhline(0.5, color="gray", linewidth=1)
    wrong.text(0.02, 0.51, "coin flip", transform=wrong.get_yaxis_transform(), fontsize=7, color="gray")
    wrong.text(0.02, 0.02, "dotted: two fresh searches of the same position", transform=wrong.transAxes, fontsize=7, color="gray")

    share_time.plot(nows, [by_now[now][0]["signal_share_fresh"] for now in nows], marker="o", color="tab:blue")
    share_time.set_ylim(0, 1)
    share_time.set_xlabel("gradient updates")
    share_time.set_title("share of a fresh target's pull that a second search reproduces", fontsize=10)
    share_time.grid(alpha=0.2)

    late = nows[-1]
    late_rows = {r["age_updates"]: r for r in by_now[late]}
    shown = [age for age in sorted(late_rows) if age < late][:: max(1, (len(late_rows) - 1) // 4)]
    for index, now in enumerate(nows):
        row = by_now[now][0]
        use_bits.plot(row["by_bits/fresh/x"], row["by_bits/fresh/use"], marker="o", color=colors[now], label=f"after {now} updates")
    use_bits.set_xlabel("bits of the row (octile means)")
    use_bits.set_title("fresh targets: usable signal against the row's bits", fontsize=10)
    use_bits.legend(fontsize=7)
    use_bits.grid(alpha=0.2)

    age_colors = plt.get_cmap("plasma")
    for index, age in enumerate(shown):
        row = late_rows[age]
        x = np.asarray(row["by_bits/stored/x"])
        use, norm = np.asarray(row["by_bits/stored/use"]), np.asarray(row["by_bits/stored/norm"])
        color = age_colors(index / max(len(shown), 1))
        share_bits.plot(x, use / norm, marker="o", color=color, label=f"target age {age}")
        wrong_bits.plot(x, row["by_bits/stored/wrong_way"], marker="o", color=color, label=f"target age {age}")
    for axis, title in (
        (share_bits, f"signal share against stored bits (network after {late} updates)"),
        (wrong_bits, f"wrong-way share against stored bits (network after {late} updates)"),
    ):
        axis.set_xscale("log")
        axis.set_xlabel("bits the pool would store for the row (octile means)")
        axis.set_title(title, fontsize=10)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    share_bits.set_ylim(0, 1)
    wrong_bits.set_ylim(0, 0.7)
    wrong_bits.axhline(0.5, color="gray", linewidth=1)

    figure.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
