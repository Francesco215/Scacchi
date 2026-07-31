#!/usr/bin/env python3
"""Plot comparable training histories from CSV or JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--input must be LABEL=PATH")
    return label, Path(path)


def number(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_rows(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    with path.open(encoding="utf-8") as handle:
        if suffix == ".csv":
            return list(csv.DictReader(handle))
        if suffix in {".jsonl", ".ndjson"}:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"Unsupported input format for {path}; use CSV or JSONL")


def ema(values: list[float], alpha: float) -> list[float]:
    result: list[float] = []
    state = math.nan
    for value in values:
        if math.isfinite(value):
            state = value if not math.isfinite(state) else alpha * value + (1 - alpha) * state
        result.append(state)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=parse_input)
    parser.add_argument("--metrics", required=True, help="Comma-separated metric keys")
    parser.add_argument("--x", default="_step", help="X-axis key (default: _step)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ema", type=float, default=0.08, help="EMA alpha; 0 disables")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    args = parser.parse_args()

    if not 0 <= args.ema <= 1:
        parser.error("--ema must be between 0 and 1")
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    if not metrics:
        parser.error("--metrics must contain at least one key")

    histories = [(label, load_rows(path)) for label, path in args.input]
    figure, axes = plt.subplots(
        len(metrics), 1, figsize=(11, max(3.4, 3.2 * len(metrics))), squeeze=False, sharex=True
    )

    for axis, metric in zip(axes[:, 0], metrics, strict=True):
        plotted = False
        for label, rows in histories:
            points = [(number(row.get(args.x)), number(row.get(metric))) for row in rows]
            points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
            points.sort(key=lambda point: point[0])
            if not points:
                continue
            xs, ys = map(list, zip(*points, strict=True))
            (raw_line,) = axis.plot(xs, ys, alpha=0.22, linewidth=0.8)
            if args.ema:
                axis.plot(xs, ema(ys, args.ema), label=label, color=raw_line.get_color(), linewidth=2)
            else:
                raw_line.set_alpha(0.9)
                raw_line.set_linewidth(1.5)
                raw_line.set_label(label)
            plotted = True
        axis.set_title(metric, loc="left", fontweight="bold")
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)
        if args.log_y:
            axis.set_yscale("log")
        if plotted:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=axis.transAxes)

    axes[-1, 0].set_xlabel(args.x)
    if args.x_min is not None or args.x_max is not None:
        axes[-1, 0].set_xlim(left=args.x_min, right=args.x_max)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
