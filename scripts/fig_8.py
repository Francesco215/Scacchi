"""Reproduce the Fig. 8 test-time search sweep for the 8x8 Hex checkpoints.

Jones (2021) Fig. 8 varies the evaluation-time tree size for fixed agents.
This version keeps the 8x8 Boardlaw-Dirichlet architecture fixed and uses the
saved training checkpoints as the curves. Elo is reported relative to the
latest ``checkpoints/8_solved`` model, which is anchored at Elo 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from scacchi.checkpoint import _suppress_orbax_logs, from_pretrained
from scacchi.dirichlet_q_search import posterior_sample_action
from scacchi.envs import make_env
from scacchi.evaluations import _make_model_mcts_policy
from scacchi.network import build_model
from scacchi.train import Config, normalize_config_dict


DEFAULT_CHECKPOINT_DIR = Path("checkpoints/hex_bs8_boardlaw_dirichlet_c1024_l8_seed0")
DEFAULT_TARGET_DIR = Path("checkpoints/8_solved")
DEFAULT_OUT_DIR = Path("artifacts/fig_8")
DEFAULT_TREE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class LoadedCheckpoint:
    step: int
    hours: float
    frames: int
    config: Config
    model: nnx.Module


def _positive_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _checkpoint_steps(checkpoint_dir: Path) -> tuple[int, ...]:
    return tuple(
        sorted(
            int(path.name)
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    )


def _load_model_at_step(
    checkpoint_dir: Path,
    step: int,
    env: Any,
    *,
    rng_seed: int = 0,
) -> LoadedCheckpoint:
    """Load one exact Orbax checkpoint step.

    ``scacchi.checkpoint.from_pretrained`` loads the latest step only, so this
    mirrors that helper while keeping the requested checkpoint step explicit.
    """

    _suppress_orbax_logs()
    checkpoint_dir = checkpoint_dir.resolve()
    with ocp.CheckpointManager(str(checkpoint_dir)) as manager:
        if step not in set(manager.all_steps()):
            raise FileNotFoundError(f"checkpoint step {step} not found in {checkpoint_dir}")

        restored_meta = manager.restore(
            step,
            args=ocp.args.Composite(meta=ocp.args.JsonRestore()),
        )
        meta = restored_meta["meta"]
        config = Config.model_validate(
            normalize_config_dict(meta["config"]),
            extra="ignore",
            context={"model_construction_only": True},
        )
        model = build_model(
            config,
            num_actions=env.num_actions,
            observation_shape=env.observation_shape,
            rngs=nnx.Rngs(rng_seed),
        )
        restored = manager.restore(
            step,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
        )
    nnx.update(model, restored["model"])
    return LoadedCheckpoint(
        step=step,
        hours=float(meta.get("hours", math.nan)),
        frames=int(meta.get("frames", 0)),
        config=config,
        model=model,
    )


def _load_config_at_step(checkpoint_dir: Path, step: int) -> Config:
    _suppress_orbax_logs()
    with ocp.CheckpointManager(str(checkpoint_dir.resolve())) as manager:
        if step not in set(manager.all_steps()):
            raise FileNotFoundError(f"checkpoint step {step} not found in {checkpoint_dir}")
        restored_meta = manager.restore(
            step,
            args=ocp.args.Composite(meta=ocp.args.JsonRestore()),
        )
    return Config.model_validate(
        normalize_config_dict(restored_meta["meta"]["config"]),
        extra="ignore",
        context={"model_construction_only": True},
    )


def _with_eval_settings(
    config: Config,
    *,
    eval_batch_size: int,
    tree_size: int,
    num_search_blocks: int,
) -> Config:
    return config.model_copy(
        update={
            "eval_batch_size": eval_batch_size,
            "num_simulations": tree_size,
            "num_search_blocks": num_search_blocks,
            "train_tree_nodes": False,
            "wavefront_final_action_mode": "posterior_sample",
        }
    )


def _normalize_action_weights(
    action_weights: jax.Array,
    legal_action_mask: jax.Array,
) -> jax.Array:
    weights = jnp.where(legal_action_mask, action_weights, 0.0)
    total = jnp.sum(weights, axis=-1, keepdims=True)
    legal_count = jnp.sum(legal_action_mask, axis=-1, keepdims=True)
    fallback = legal_action_mask.astype(weights.dtype) / jnp.maximum(legal_count, 1)
    return jnp.where(total > 0, weights / jnp.maximum(total, 1e-8), fallback)


def make_stochastic_mcts_evaluate(env: Any, config: Config, target_model: nnx.Module):
    """Evaluate ``model`` against ``target_model`` with sampled final actions.

    The search implementation is reused from ``scacchi.evaluations``. We ignore
    the deterministic ``policy.action`` field and sample from ``action_weights``
    for both players instead.
    """

    eval_batch_size = int(getattr(config, "eval_batch_size", config.selfplay_batch_size))

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: nnx.Module) -> jax.Array:
        my_player = 0

        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, eval_batch_size)
        env_state = jax.vmap(env.init)(init_keys)

        def body_fn(val):
            key, env_state, returns = val
            key, my_search_key, target_search_key, my_sample_key, target_sample_key = (
                jax.random.split(key, 5)
            )

            my_policy = _make_model_mcts_policy(
                env,
                config,
                model,
                my_search_key,
                env_state,
                config.num_simulations,
            )
            target_policy = _make_model_mcts_policy(
                env,
                config,
                target_model,
                target_search_key,
                env_state,
                config.num_simulations,
            )

            my_weights = _normalize_action_weights(
                my_policy.action_weights,
                env_state.legal_action_mask,
            )
            target_weights = _normalize_action_weights(
                target_policy.action_weights,
                env_state.legal_action_mask,
            )
            my_action = posterior_sample_action(
                my_sample_key,
                my_weights,
                env_state.legal_action_mask,
            )
            target_action = posterior_sample_action(
                target_sample_key,
                target_weights,
                env_state.legal_action_mask,
            )

            is_my_turn = env_state.current_player == my_player
            action = jnp.where(is_my_turn, my_action, target_action)
            env_state = jax.vmap(env.step)(env_state, action)
            returns = returns + env_state.rewards[
                jnp.arange(eval_batch_size),
                my_player,
            ]
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, env_state, jnp.zeros(eval_batch_size)),
        )
        return returns

    return evaluate


def returns_to_score(avg_return: float) -> float:
    return 0.5 * (avg_return + 1.0)


def score_to_elo(score: float, *, eps: float = 1e-3) -> float:
    clipped = min(max(score, eps), 1.0 - eps)
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def _result_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["checkpoint_step"]),
        int(row["tree_size"]),
        int(row["num_search_blocks"]),
    )


def _load_existing_results(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[tuple[int, int, int], dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            results[_result_key(row)] = row
    return results


def _append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        raise ValueError("no rows to plot")
    steps = sorted({int(row["checkpoint_step"]) for row in rows})
    cmap = plt.get_cmap("viridis")
    colors = {
        step: cmap(i / max(1, len(steps) - 1))
        for i, step in enumerate(steps)
    }

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    endpoints: list[tuple[int, float, float, Any]] = []
    for step in steps:
        step_rows = sorted(
            (row for row in rows if int(row["checkpoint_step"]) == step),
            key=lambda row: int(row["tree_size"]),
        )
        xs = np.array([row["tree_size"] for row in step_rows], dtype=np.float64)
        ys = np.array([row["elo_vs_target"] for row in step_rows], dtype=np.float64)
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=3.8,
            linewidth=1.3,
            alpha=0.82,
            color=colors[step],
        )
        if len(xs):
            endpoints.append((step, float(xs[-1]), float(ys[-1]), colors[step]))

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["tree_size"]) for row in rows}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Test-time tree size (search simulations)")
    ax.set_ylabel("Elo vs. checkpoints/8_solved (target = 0)")
    ax.set_title("Fig. 8 reproduction: checkpoint curves, stochastic search actions")
    ax.grid(True, which="both", alpha=0.22)
    ax.margins(x=0.08, y=0.08)

    y_min, y_max = ax.get_ylim()
    min_gap = 0.035 * (y_max - y_min)
    label_positions: list[tuple[int, float, float, float, Any]] = []
    last_y = -math.inf
    for step, x, y, color in sorted(endpoints, key=lambda item: item[2]):
        label_y = max(y, last_y + min_gap)
        label_positions.append((step, x, y, label_y, color))
        last_y = label_y
    overflow = label_positions[-1][3] - y_max if label_positions else 0.0
    if overflow > 0:
        label_positions = [
            (step, x, y, label_y - overflow, color)
            for step, x, y, label_y, color in label_positions
        ]
    for step, x, y, label_y, color in label_positions:
        ax.annotate(
            str(step),
            xy=(x, y),
            xytext=(x * 1.035, label_y),
            textcoords="data",
            color=color,
            fontsize=8.5,
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "alpha": 0.55,
                "lw": 0.8,
                "shrinkA": 0,
                "shrinkB": 3,
            },
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    steps = sorted({int(row["checkpoint_step"]) for row in rows})
    tree_sizes = sorted({int(row["tree_size"]) for row in rows})
    lookup = {
        (int(row["checkpoint_step"]), int(row["tree_size"])): float(row["elo_vs_target"])
        for row in rows
    }
    values = np.array(
        [[lookup.get((step, tree_size), np.nan) for tree_size in tree_sizes] for step in steps],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    im = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(tree_sizes)))
    ax.set_xticklabels([str(v) for v in tree_sizes])
    ax.set_yticks(np.arange(len(steps)))
    ax.set_yticklabels([str(v) for v in steps])
    ax.set_xlabel("Test-time tree size")
    ax.set_ylabel("Checkpoint step")
    ax.set_title("Elo vs. checkpoints/8_solved")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Elo")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _summarize_returns(
    returns: np.ndarray,
    *,
    checkpoint: LoadedCheckpoint,
    tree_size: int,
    num_search_blocks: int,
    seed: int,
) -> dict[str, Any]:
    avg_return = float(np.mean(returns))
    score = returns_to_score(avg_return)
    wins = int(np.sum(returns == 1))
    draws = int(np.sum(returns == 0))
    losses = int(np.sum(returns == -1))
    return {
        "checkpoint_step": checkpoint.step,
        "checkpoint_frames": checkpoint.frames,
        "checkpoint_hours": checkpoint.hours,
        "tree_size": int(tree_size),
        "num_search_blocks": int(num_search_blocks),
        "eval_games": int(returns.size),
        "seed": int(seed),
        "avg_return": avg_return,
        "score_vs_target": score,
        "elo_vs_target": score_to_elo(score),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "target": str(DEFAULT_TARGET_DIR),
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoint_dir = args.checkpoint_dir
    target_dir = args.target_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_steps = _checkpoint_steps(checkpoint_dir)
    steps = args.checkpoint_steps or all_steps
    missing = sorted(set(steps) - set(all_steps))
    if missing:
        raise FileNotFoundError(f"checkpoint steps not found: {missing}")

    results_path = out_dir / "fig_8_results.jsonl"
    csv_path = out_dir / "fig_8_results.csv"
    if args.overwrite and results_path.exists():
        results_path.unlink()
    results = _load_existing_results(results_path)

    base_config = _load_config_at_step(checkpoint_dir, steps[0])
    env = make_env(base_config.env_id, base_config.board_size)
    target_model = from_pretrained(str(target_dir), env, rngs=nnx.Rngs(args.seed))

    pending_steps = [
        step
        for step in steps
        if any(
            args.overwrite
            or (step, tree_size, args.num_search_blocks) not in results
            for tree_size in args.tree_sizes
        )
    ]
    checkpoints = {
        step: _load_model_at_step(checkpoint_dir, step, env, rng_seed=args.seed)
        for step in tqdm(pending_steps, desc="load checkpoints", dynamic_ncols=True)
    }

    for tree_size in tqdm(args.tree_sizes, desc="tree size", dynamic_ncols=True):
        eval_config = _with_eval_settings(
            base_config,
            eval_batch_size=args.eval_games,
            tree_size=tree_size,
            num_search_blocks=args.num_search_blocks,
        )
        evaluate = make_stochastic_mcts_evaluate(env, eval_config, target_model)
        for step in tqdm(steps, desc=f"tree {tree_size}", leave=False):
            key = (step, tree_size, args.num_search_blocks)
            if key in results and not args.overwrite:
                continue
            checkpoint = checkpoints[step]
            eval_seed = args.seed + 1009 * step + 9176 * tree_size
            returns = np.asarray(
                jax.device_get(evaluate(jax.random.PRNGKey(eval_seed), checkpoint.model))
            )
            row = _summarize_returns(
                returns,
                checkpoint=checkpoint,
                tree_size=tree_size,
                num_search_blocks=args.num_search_blocks,
                seed=eval_seed,
            )
            row["target"] = str(target_dir)
            results[key] = row
            _append_result(results_path, row)

    rows = sorted(results.values(), key=lambda row: (row["checkpoint_step"], row["tree_size"]))
    _write_csv(csv_path, rows)
    _plot_curves(rows, out_dir / "fig_8.png")
    _plot_heatmap(rows, out_dir / "fig_8_heatmap.png")
    with (out_dir / "fig_8_summary.json").open("w") as f:
        json.dump(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "target_dir": str(target_dir),
                "target_elo": 0,
                "tree_sizes": list(args.tree_sizes),
                "num_search_blocks": args.num_search_blocks,
                "eval_games": args.eval_games,
                "num_points": len(rows),
                "pngs": ["fig_8.png", "fig_8_heatmap.png"],
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--checkpoint-steps", type=_int_csv, default=None)
    parser.add_argument("--tree-sizes", type=_positive_int_csv, default=DEFAULT_TREE_SIZES)
    parser.add_argument("--eval-games", type=int, default=256)
    parser.add_argument(
        "--num-search-blocks",
        type=int,
        default=1,
        help="Use 1 so the plotted tree size equals num_simulations.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.eval_games <= 0:
        parser.error("--eval-games must be positive")
    if args.num_search_blocks <= 0:
        parser.error("--num-search-blocks must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = run(args)
    if not rows:
        raise SystemExit("no results produced")
    print(f"wrote {args.out_dir / 'fig_8.png'}")
    print(f"wrote {args.out_dir / 'fig_8_heatmap.png'}")


if __name__ == "__main__":
    main()
