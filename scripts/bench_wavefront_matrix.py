from __future__ import annotations

import argparse
import itertools
import json
import time
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scacchi.dirichlet_tree.arena_search import BatchedPosteriorArenaSearch, TIMING_BUCKETS
from scacchi.dirichlet_tree.search import search_config_from_any
from scacchi.envs import make_env
from scacchi.posterior_tree import run_posterior_tree_search, split_batched_state


ABLATION_VARIANTS = (
    "object_tree",
    "arena_no_hybrid_selector",
    "arena_hybrid_selector",
    "arena_no_grouped_expansion",
    "arena_grouped_expansion",
    "arena_no_lane_indexed_step",
    "arena_lane_indexed_step",
    "arena_no_stable_lane_batch",
    "arena_stable_lane_batch",
    "arena_no_padded_eval_gathers",
    "arena_padded_eval_gathers",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeatable wavefront arena benchmark matrices. Run once with "
            "JAX_PLATFORMS=cpu and once on the target GPU/default backend."
        )
    )
    parser.add_argument("--benchmark", choices=["matrix", "ablation"], default="matrix")
    parser.add_argument("--batches", default="128,512,2048")
    parser.add_argument("--simulations", default="1,4,8,16,32")
    parser.add_argument("--board-sizes", default="5,9,11")
    parser.add_argument("--root-modes", default="initial,prefilled")
    parser.add_argument("--policy-mc-samples", default="1,16,64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefill-steps", type=int, default=4)
    parser.add_argument("--variants", default="arena_hybrid_selector")
    parser.add_argument("--require-backend", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--print-runs", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke subset.")
    args = parser.parse_args()

    backend = jax.default_backend()
    if args.require_backend is not None and backend != args.require_backend:
        raise RuntimeError(f"expected JAX backend {args.require_backend!r}, got {backend!r}")

    if args.benchmark == "ablation":
        variants = _parse_strings(args.variants)
        if args.variants == "arena_hybrid_selector":
            variants = list(ABLATION_VARIANTS)
        batches = _parse_ints(args.batches)
        simulations = _parse_ints(args.simulations)
        board_sizes = _parse_ints(args.board_sizes)
        root_modes = _parse_strings(args.root_modes)
        policy_samples = _parse_ints(args.policy_mc_samples)
        if args.quick:
            batches, simulations, board_sizes, root_modes, policy_samples = [128], [4], [5], ["prefilled"], [1]
    else:
        variants = _parse_strings(args.variants)
        batches = _parse_ints(args.batches)
        simulations = _parse_ints(args.simulations)
        board_sizes = _parse_ints(args.board_sizes)
        root_modes = _parse_strings(args.root_modes)
        policy_samples = _parse_ints(args.policy_mc_samples)
        if args.quick:
            batches, simulations, board_sizes, root_modes, policy_samples = [128], [1, 4], [5], ["initial"], [1]

    output_file = open(args.output, "w", encoding="utf-8") if args.output else None
    try:
        for board_size, batch, sims, root_mode, samples, variant in itertools.product(
            board_sizes,
            batches,
            simulations,
            root_modes,
            policy_samples,
            variants,
        ):
            if variant == "object_tree" and batch > 512:
                continue
            record = _run_case(
                backend=backend,
                board_size=board_size,
                batch=batch,
                simulations=sims,
                root_mode=root_mode,
                prefill_steps=args.prefill_steps,
                policy_mc_samples=samples,
                variant=variant,
                repeats=max(1, int(args.repeats)),
                print_runs=args.print_runs,
            )
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            if output_file is not None:
                output_file.write(line + "\n")
                output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()


def _run_case(
    *,
    backend: str,
    board_size: int,
    batch: int,
    simulations: int,
    root_mode: str,
    prefill_steps: int,
    policy_mc_samples: int,
    variant: str,
    repeats: int,
    print_runs: bool,
) -> dict[str, Any]:
    env = make_env("hex", board_size)
    root_state_batch = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(0), batch))
    _block_until_ready(root_state_batch)
    actual_prefill = 0
    if root_mode == "prefilled":
        actual_prefill = min(int(prefill_steps), board_size * board_size - 1)
        root_state_batch = _prefill_hex_states(
            env,
            root_state_batch,
            batch=batch,
            board_size=board_size,
            steps=actual_prefill,
        )
        _block_until_ready(root_state_batch)
    elif root_mode != "initial":
        raise ValueError(f"unknown root_mode: {root_mode}")

    elapsed_values: list[float] = []
    throughput_values: list[float] = []
    timing_values: dict[str, list[float]] = {bucket: [] for bucket in TIMING_BUCKETS}
    total_completed = int(batch) * int(simulations)
    for repeat in range(repeats):
        elapsed, timing = _run_once(
            env=env,
            root_state_batch=root_state_batch,
            batch=batch,
            simulations=simulations,
            board_size=board_size,
            policy_mc_samples=policy_mc_samples,
            variant=variant,
            rng_key=jax.random.PRNGKey(repeat),
        )
        elapsed_values.append(elapsed)
        throughput_values.append(total_completed / elapsed)
        for bucket in TIMING_BUCKETS:
            timing_values[bucket].append(float(timing.get(bucket, 0.0)))
        if print_runs:
            print(
                json.dumps(
                    {
                        "kind": "run",
                        "backend": backend,
                        "board_size": board_size,
                        "batch": batch,
                        "simulations": simulations,
                        "root_mode": root_mode,
                        "prefill_steps": actual_prefill,
                        "policy_mc_samples": policy_mc_samples,
                        "variant": variant,
                        "repeat": repeat + 1,
                        "elapsed_s": elapsed,
                        "completed_evals_per_s": total_completed / elapsed,
                        "timing_s": timing,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return {
        "kind": "summary",
        "backend": backend,
        "devices": [str(device) for device in jax.devices()],
        "board_size": board_size,
        "batch": batch,
        "simulations": simulations,
        "root_mode": root_mode,
        "prefill_steps": actual_prefill,
        "policy_mc_samples": policy_mc_samples,
        "variant": variant,
        "repeats": repeats,
        "total_completed_evals": total_completed,
        "elapsed_s": _time_stats(elapsed_values),
        "completed_evals_per_s": _throughput_stats(throughput_values),
        "timing_s": {bucket: _time_stats(values) for bucket, values in timing_values.items()},
    }


def _run_once(
    *,
    env: Any,
    root_state_batch: Any,
    batch: int,
    simulations: int,
    board_size: int,
    policy_mc_samples: int,
    variant: str,
    rng_key: jax.Array,
) -> tuple[float, dict[str, float]]:
    num_actions = int(env.num_actions)

    def leaf_evaluator(obs):
        rows = obs.shape[0]
        return (
            jnp.zeros((rows, num_actions), dtype=jnp.float32),
            jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (rows, 1)),
            jnp.ones((rows, num_actions, 3), dtype=jnp.float32),
        )

    config = _config_for_variant(
        batch=batch,
        simulations=simulations,
        board_size=board_size,
        policy_mc_samples=policy_mc_samples,
        variant=variant,
    )
    start = time.perf_counter()
    if variant == "object_tree":
        output = run_posterior_tree_search(
            env=env,
            root_states=split_batched_state(root_state_batch),
            leaf_evaluator=leaf_evaluator,
            rng_key=rng_key,
            config=config,
        )
        timing = {bucket: 0.0 for bucket in TIMING_BUCKETS}
    else:
        search_config = search_config_from_any(config, num_roots=batch)
        search = BatchedPosteriorArenaSearch(env=env, rng_key=rng_key)
        search.enable_timing(sync=True)
        output = search.search_state_batch(root_state_batch, leaf_evaluator, search_config)
        timing = dict(search.timing)
    _block_until_ready(output)
    return time.perf_counter() - start, timing


def _config_for_variant(
    *,
    batch: int,
    simulations: int,
    board_size: int,
    policy_mc_samples: int,
    variant: str,
) -> SimpleNamespace:
    values = dict(
        search_policy="posterior_tree_wavefront",
        num_simulations=int(simulations),
        search_eval_batch_size=int(batch),
        c_leaf=1.0,
        c_terminal=8.0,
        c_state=0.1,
        c_value_search=1.0,
        policy_mc_samples=int(policy_mc_samples),
        backup_mc_samples=1,
        selfplay_action_source="scalar_q_argmax",
        wavefront_num_lanes_per_root=1,
        wavefront_max_depth=max(16, int(simulations) + int(board_size) * int(board_size) + 2),
        wavefront_final_action_mode="argmax_q_mean",
        wavefront_pad_eval_batches=True,
        wavefront_pad_jax_select=False,
        wavefront_np_select_below=1024,
        wavefront_grouped_expansion=True,
        wavefront_lane_indexed_step=True,
        wavefront_stable_lane_batch=True,
        wavefront_pad_pending_observation_gather=True,
    )
    if variant == "object_tree":
        values["search_policy"] = "posterior_tree"
    elif variant == "arena_no_hybrid_selector":
        values["wavefront_np_select_below"] = 0
    elif variant == "arena_hybrid_selector":
        pass
    elif variant == "arena_no_grouped_expansion":
        values["wavefront_grouped_expansion"] = False
    elif variant == "arena_grouped_expansion":
        pass
    elif variant == "arena_no_lane_indexed_step":
        values["wavefront_lane_indexed_step"] = False
    elif variant == "arena_lane_indexed_step":
        pass
    elif variant == "arena_no_stable_lane_batch":
        values["wavefront_stable_lane_batch"] = False
    elif variant == "arena_stable_lane_batch":
        pass
    elif variant == "arena_no_padded_eval_gathers":
        values["wavefront_pad_pending_observation_gather"] = False
    elif variant == "arena_padded_eval_gathers":
        pass
    else:
        raise ValueError(f"unknown variant: {variant}")
    return SimpleNamespace(**values)


def _prefill_hex_states(env, state_batch, *, batch: int, board_size: int, steps: int):
    if steps <= 0:
        return state_batch
    if steps >= board_size * board_size:
        raise ValueError("prefill_steps must be smaller than board_size ** 2")
    step_fn = jax.jit(jax.vmap(env.step))
    area = board_size * board_size
    batch_ids = jnp.arange(batch, dtype=jnp.int32)
    offset = batch_ids % area
    stride = 1 + (batch_ids // area) % (area - 1)
    stride = jnp.where(stride % board_size == 0, stride + 1, stride)
    for step in range(steps):
        actions = (offset + step * stride) % area
        state_batch = step_fn(state_batch, actions)
    return state_batch


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _parse_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _time_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    cold_excluded = array[1:] if array.shape[0] > 1 else array
    return {
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "best": float(np.min(array)),
        "cold_start_excluded_mean": float(np.mean(cold_excluded)),
    }


def _throughput_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    cold_excluded = array[1:] if array.shape[0] > 1 else array
    return {
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "best": float(np.max(array)),
        "cold_start_excluded_mean": float(np.mean(cold_excluded)),
    }


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


if __name__ == "__main__":
    main()
