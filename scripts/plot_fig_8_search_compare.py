"""Compare Fig. 8 Elo curves from two search backends.

The script expects each input directory to contain ``fig_8_results.csv`` as
written by ``scripts/fig_8.py``. Checkpoint step 0 is omitted by default.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DEFAULT_OLD_DIR = Path("artifacts/fig_8_old_dirichlet_q")
DEFAULT_DQAZ_DIR = Path("artifacts/fig_8")
DEFAULT_OUT = Path("artifacts/fig_8_search_compare/elo_compare_no_step0.png")


def _read_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "fig_8_results.csv"
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "label": label,
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "tree_size": int(row["tree_size"]),
                    "elo_vs_target": float(row["elo_vs_target"]),
                    "target_tree_size": int(row["target_tree_size"]),
                }
            )
    return rows


def _colors_for_steps(steps: Iterable[int], cmap_name: str) -> dict[int, Any]:
    steps = tuple(sorted(set(steps)))
    cmap = plt.get_cmap(cmap_name)
    if len(steps) == 1:
        return {steps[0]: cmap(0.42)}
    return {
        step: cmap(0.9 - 0.48 * i / (len(steps) - 1))
        for i, step in enumerate(steps)
    }


def _tree_size_plot_position(tree_size: int) -> float:
    if tree_size == 0:
        return -0.5
    if tree_size < 0:
        raise ValueError(f"tree_size must be nonnegative, got {tree_size}")
    return math.log2(tree_size)


def _plot(
    *,
    old_rows: list[dict[str, Any]],
    dqaz_rows: list[dict[str, Any]],
    out_path: Path,
    old_label: str,
    dqaz_label: str,
    old_cmap: str,
    dqaz_cmap: str,
    include_step0: bool,
) -> None:
    rows_by_label = {
        old_label: old_rows,
        dqaz_label: dqaz_rows,
    }
    if not include_step0:
        rows_by_label = {
            label: [row for row in rows if row["checkpoint_step"] != 0]
            for label, rows in rows_by_label.items()
        }
    all_rows = [row for rows in rows_by_label.values() for row in rows]
    if not all_rows:
        raise ValueError("no rows left to plot")
    tree_sizes = sorted({row["tree_size"] for row in all_rows})
    steps = sorted({row["checkpoint_step"] for row in all_rows})
    colors = _colors_for_steps(steps, old_cmap)
    old_lookup = {
        (row["checkpoint_step"], row["tree_size"]): row["elo_vs_target"]
        for row in rows_by_label[old_label]
    }
    dqaz_lookup = {
        (row["checkpoint_step"], row["tree_size"]): row["elo_vs_target"]
        for row in rows_by_label[dqaz_label]
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.2, 6.4))

    styles = {
        old_label: {
            "linestyle": "--",
            "marker": "s",
            "linewidth": 1.35,
            "alpha": 0.86,
        },
        dqaz_label: {
            "linestyle": "-",
            "marker": "o",
            "linewidth": 1.55,
            "alpha": 0.9,
        },
    }
    method_handles: dict[str, Line2D] = {}
    step_handles: list[Line2D] = []
    delta_label_specs: list[dict[str, Any]] = []

    for label, rows in rows_by_label.items():
        if not rows:
            continue
        style = styles[label]
        method_handles[label] = (
            Line2D(
                [0],
                [0],
                color="0.18",
                linestyle=cast(str, style["linestyle"]),
                marker=cast(str, style["marker"]),
                linewidth=cast(float, style["linewidth"]),
                label=label,
            )
        )
        for step in sorted({row["checkpoint_step"] for row in rows}):
            step_rows = sorted(
                (row for row in rows if row["checkpoint_step"] == step),
                key=lambda row: row["tree_size"],
            )
            xs = np.array(
                [_tree_size_plot_position(row["tree_size"]) for row in step_rows],
                dtype=np.float64,
            )
            ys = np.array([row["elo_vs_target"] for row in step_rows], dtype=np.float64)
            ax.plot(
                xs,
                ys,
                color=colors[step],
                linestyle=styles[label]["linestyle"],
                marker=styles[label]["marker"],
                markersize=4.2,
                linewidth=styles[label]["linewidth"],
                alpha=styles[label]["alpha"],
            )

    for step in steps:
        step_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[step],
                marker="s",
                linestyle="",
                markersize=8,
                label=str(step),
            )
        )

    common_tree_sizes = sorted(
        tree_size
        for step, tree_size in old_lookup
        if (step, tree_size) in dqaz_lookup
    )
    comparison_tree_size = (
        512 if 512 in common_tree_sizes else common_tree_sizes[-1] if common_tree_sizes else None
    )
    delta_xs: list[float] = []
    if comparison_tree_size is not None:
        comparison_x = _tree_size_plot_position(comparison_tree_size)
        delta_steps = [
            step
            for step in steps
            if (step, comparison_tree_size) in old_lookup
            and (step, comparison_tree_size) in dqaz_lookup
        ]
        for index, step in enumerate(delta_steps):
            old_elo = old_lookup[(step, comparison_tree_size)]
            dqaz_elo = dqaz_lookup[(step, comparison_tree_size)]
            y_low, y_high = sorted((old_elo, dqaz_elo))
            x = comparison_x + 0.3 + 0.42 * index
            delta_xs.append(x)
            ax.annotate(
                "",
                xy=(x, y_high),
                xytext=(x, y_low),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": colors[step],
                    "alpha": 0.9,
                    "lw": 1.05,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
            delta_label_specs.append(
                {
                    "x": x,
                    "mid_y": 0.5 * (y_low + y_high),
                    "delta": dqaz_elo - old_elo,
                    "color": colors[step],
                }
            )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    if delta_label_specs:
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        min_gap = 0.05 * y_span
        label_positions: list[tuple[dict[str, Any], float]] = []
        last_y = -math.inf
        for spec in sorted(delta_label_specs, key=lambda item: item["mid_y"]):
            label_y = max(spec["mid_y"], last_y + min_gap)
            label_positions.append((spec, label_y))
            last_y = label_y
        top_limit = y_max - 0.03 * y_span
        overflow = label_positions[-1][1] - top_limit
        if overflow > 0:
            label_positions = [(spec, label_y - overflow) for spec, label_y in label_positions]
        bottom_limit = y_min + 0.03 * y_span
        underflow = bottom_limit - label_positions[0][1]
        if underflow > 0:
            label_positions = [(spec, label_y + underflow) for spec, label_y in label_positions]
        for spec, label_y in label_positions:
            ax.text(
                spec["x"] + 0.05,
                label_y,
                rf"$\Delta$ Elo = {spec['delta']:+.0f}",
                color=spec["color"],
                fontsize=7.5,
                va="center",
                ha="left",
            )
    tick_positions = [_tree_size_plot_position(tree_size) for tree_size in tree_sizes]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(tree_size) for tree_size in tree_sizes])
    ax.set_xlabel("num-trajectories")
    ax.set_ylabel("elo compared to ~optimal play engine")

    ax.set_title("test time performance of different methods")
    ax.grid(True, which="both", alpha=0.22)
    ax.margins(x=0.16, y=0.1)
    if tick_positions:
        right = max(tick_positions)
        if delta_xs:
            right = max(right, max(delta_xs) + 1.2)
        ax.set_xlim(min(tick_positions) - 0.4, right)

    method_legend = ax.legend(
        handles=[method_handles[dqaz_label], method_handles[old_label]],
        title="Search backend",
        loc="upper left",
    )
    ax.add_artist(method_legend)
    ax.legend(handles=step_handles, title="Curves", loc="lower right", fontsize=7.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD_DIR)
    parser.add_argument("--dqaz-dir", type=Path, default=DEFAULT_DQAZ_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-label", default=r"$\alpha$ 0-like search")
    parser.add_argument("--dqaz-label", default="recursive thompson search")
    parser.add_argument("--old-cmap", default="viridis")
    parser.add_argument("--dqaz-cmap", default="viridis")
    parser.add_argument(
        "--include-step0",
        action="store_true",
        help="Include checkpoint step 0 instead of dropping it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_rows = _read_rows(args.old_dir, label=args.old_label)
    dqaz_rows = _read_rows(args.dqaz_dir, label=args.dqaz_label)
    _plot(
        old_rows=old_rows,
        dqaz_rows=dqaz_rows,
        out_path=args.out,
        old_label=args.old_label,
        dqaz_label=args.dqaz_label,
        old_cmap=args.old_cmap,
        dqaz_cmap=args.dqaz_cmap,
        include_step0=args.include_step0,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
