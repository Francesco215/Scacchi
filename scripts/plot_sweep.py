"""Plot the hex sweep as Elo-vs-compute curves, one color per board size.

Mirrors Andy Jones' Fig. 5: faint per-run sigmoids plus a dark frontier formed
by taking the running max across runs of the same board size.

Reads training history from W&B project ``scacchi-az`` (set via --project).
Elo is computed against the in-run baseline opponent (the ``board_size_solved``
checkpoint loaded by ``scacchi.train``), not vs. perfect play — the absolute
offset will differ from the paper, but the shape of the curves is comparable.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# ---------- Elo helpers ----------

def winrate_to_elo(score: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Convert per-game score (W + 0.5 D in [0, 1]) to Elo diff vs. opponent."""
    s = np.clip(score, eps, 1 - eps)
    return 400.0 * np.log10(s / (1.0 - s))


def returns_to_score(avg_r: np.ndarray) -> np.ndarray:
    """avg_R in [-1, 1] (W=+1, L=-1, D=0) → score in [0, 1]."""
    return 0.5 * (avg_r + 1.0)


# ---------- Data loading ----------

@dataclass
class Run:
    board_size: int
    width: int
    depth: int
    compute: np.ndarray  # FLOPS-seconds (proxy: frames * params)
    frames: np.ndarray
    elo: np.ndarray


def _approx_params(width: int, depth: int, board_size: int) -> int:
    """Rough param count for a boardlaw-style resnet, used only as a compute
    proxy when wandb hasn't logged true FLOPs."""
    # ~2 conv layers per block × (3*3*width*width) + IO + head terms.
    blocks = max(depth, 1)
    body = blocks * 2 * (3 * 3 * width * width)
    io = (board_size * board_size) * width + width * (board_size * board_size + 1)
    return body + io


def load_runs(project: str, entity: str | None) -> list[Run]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = []
    for r in api.runs(path):
        cfg = r.config or {}
        if cfg.get("env_id") != "hex":
            continue
        bs = cfg.get("board_size")
        nc = cfg.get("num_channels")
        nl = cfg.get("num_layers")
        if bs is None or nc is None or nl is None:
            continue

        hist = r.history(
            keys=["train/frames", "eval/vs_baseline/avg_R"],
            samples=10_000,
            pandas=False,
        )
        rows = [h for h in hist if "eval/vs_baseline/avg_R" in h and "train/frames" in h]
        if not rows:
            continue
        rows.sort(key=lambda h: h["train/frames"])
        frames = np.array([h["train/frames"] for h in rows], dtype=np.float64)
        avg_r = np.array([h["eval/vs_baseline/avg_R"] for h in rows], dtype=np.float64)
        elo = winrate_to_elo(returns_to_score(avg_r))

        # Compute proxy: frames × params (board-size-aware), giving a
        # FLOPS-seconds-like quantity that grows with both training time and
        # model size, like the paper's x-axis.
        compute = frames * _approx_params(nc, nl, bs)
        runs.append(Run(bs, nc, nl, compute, frames, elo))
    return runs


# ---------- Plotting ----------

BOARD_COLORS = {
    3: "#b3252b",
    4: "#a3a82c",
    5: "#5fb04b",
    6: "#2da296",
    7: "#3a6fbf",
    8: "#5b3aa6",
    9: "#c14ab5",
}


def plot(runs: list[Run], x: str, out_path: str) -> None:
    by_bs: dict[int, list[Run]] = defaultdict(list)
    for r in runs:
        by_bs[r.board_size].append(r)

    fig, ax = plt.subplots(figsize=(9, 6))
    for bs in sorted(by_bs):
        color = BOARD_COLORS.get(bs, None)
        # Faint per-run lines.
        for r in by_bs[bs]:
            xs = r.compute if x == "compute" else r.frames
            ax.plot(xs, r.elo, color=color, alpha=0.18, linewidth=0.9)

        # Frontier: at each x grid point take the running-max Elo across all
        # runs for this board size.
        all_x = np.concatenate([
            r.compute if x == "compute" else r.frames for r in by_bs[bs]
        ])
        if len(all_x) == 0:
            continue
        grid = np.geomspace(max(all_x.min(), 1.0), all_x.max(), 256)
        frontier = np.full_like(grid, -np.inf, dtype=np.float64)
        for r in by_bs[bs]:
            xs = r.compute if x == "compute" else r.frames
            order = np.argsort(xs)
            xs_s, elo_s = xs[order], r.elo[order]
            interp = np.interp(grid, xs_s, np.maximum.accumulate(elo_s),
                               left=np.nan, right=np.nan)
            mask = ~np.isnan(interp)
            frontier[mask] = np.maximum(frontier[mask], interp[mask])
        frontier[np.isinf(frontier)] = np.nan
        ax.plot(grid, frontier, color=color, linewidth=2.2, label=f"{bs}")
        # Label at the left edge of the frontier, like the paper.
        valid = ~np.isnan(frontier)
        if valid.any():
            i = np.argmax(valid)
            ax.text(grid[i], frontier[i], f" {bs}", color=color,
                    fontsize=12, va="center", ha="left", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Training compute (FLOPS-seconds)" if x == "compute" else "Training samples")
    ax.set_ylabel("Elo v. baseline")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="scacchi-az")
    p.add_argument("--entity", default=None)
    p.add_argument("--x", choices=["compute", "frames"], default="compute")
    p.add_argument("--out", default="sweep_hex_elo.png")
    args = p.parse_args()

    runs = load_runs(args.project, args.entity)
    if not runs:
        raise SystemExit("no hex runs with eval/vs_baseline/avg_R found in W&B project")
    print(f"loaded {len(runs)} runs across board sizes "
          f"{sorted({r.board_size for r in runs})}")
    plot(runs, args.x, args.out)


if __name__ == "__main__":
    main()
