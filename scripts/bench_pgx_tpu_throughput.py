#!/usr/bin/env python3
"""Measure PGX env.step throughput on a TPU pod.

Run this script on every TPU VM host at the same time, for example with:

    eopod run --worker all "cd /path/to/repo && .venv/bin/python scripts/bench_pgx_tpu_throughput.py"

The script prints one JSON object per batch size from JAX process 0. Throughput
is computed with the full global JAX device count and the slowest host wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pgx
from jax import lax
from jax.experimental import multihost_utils


def _parse_positive_ints(values: Iterable[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            item = int(part)
            if item <= 0:
                raise argparse.ArgumentTypeError("batch sizes must be positive")
            parsed.append(item)
    if not parsed:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return parsed


def _block_until_ready(tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _sync(name: str, enabled: bool) -> None:
    if enabled:
        multihost_utils.sync_global_devices(name)


def _allgather(value: jax.Array, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.asarray([np.asarray(value)])
    return np.asarray(multihost_utils.process_allgather(value, tiled=False))


def _make_benchmark(env_id: str, batch_per_device: int, steps: int, action_mode: str):
    env = pgx.make(env_id)

    def init_device(key):
        keys = jax.random.split(key, batch_per_device)
        return jax.vmap(env.init)(keys)

    def run_device(state, key):
        def step_once(carry, _):
            state, key = carry
            legal_action_mask = state.legal_action_mask
            if action_mode == "random_legal":
                key, action_key = jax.random.split(key)
                logits = jnp.where(legal_action_mask, 0.0, -jnp.inf)
                action = jax.random.categorical(action_key, logits).astype(jnp.int32)
            else:
                action = jnp.argmax(legal_action_mask, axis=-1).astype(jnp.int32)
            state = jax.vmap(env.step)(state, action)
            return (state, key), None

        (state, key), _ = lax.scan(step_once, (state, key), None, length=steps)
        return state, key

    return jax.pmap(init_device), jax.pmap(run_device)


def run_one(args: argparse.Namespace, batch_per_device: int) -> dict[str, object] | None:
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    global_device_count = jax.device_count()
    sync_enabled = not args.no_multihost_sync and process_count > 1

    init_pmapped, run_pmapped = _make_benchmark(
        args.env_id,
        batch_per_device=batch_per_device,
        steps=args.steps,
        action_mode=args.action_mode,
    )

    seed = args.seed + process_index * 100_003
    init_keys = jax.random.split(jax.random.PRNGKey(seed), local_device_count)
    step_keys = jax.random.split(jax.random.PRNGKey(seed + 1), local_device_count)

    _sync(f"pgx-bench-init-{args.env_id}-{batch_per_device}", sync_enabled)
    state = _block_until_ready(init_pmapped(init_keys))

    _sync(f"pgx-bench-compile-{args.env_id}-{batch_per_device}", sync_enabled)
    compile_start = time.perf_counter()
    state, step_keys = _block_until_ready(run_pmapped(state, step_keys))
    compile_plus_first_run_s = time.perf_counter() - compile_start

    for warmup_idx in range(args.warmup_runs):
        _sync(
            f"pgx-bench-warmup-{args.env_id}-{batch_per_device}-{warmup_idx}",
            sync_enabled,
        )
        state, step_keys = _block_until_ready(run_pmapped(state, step_keys))

    wall_times: list[float] = []
    for repeat_idx in range(args.repeats):
        _sync(
            f"pgx-bench-repeat-start-{args.env_id}-{batch_per_device}-{repeat_idx}",
            sync_enabled,
        )
        start = time.perf_counter()
        state, step_keys = _block_until_ready(run_pmapped(state, step_keys))
        wall_times.append(time.perf_counter() - start)
        _sync(
            f"pgx-bench-repeat-done-{args.env_id}-{batch_per_device}-{repeat_idx}",
            sync_enabled,
        )

    local_wall_times = jnp.asarray(wall_times, dtype=jnp.float32)
    all_wall_times = _allgather(local_wall_times, sync_enabled)
    all_compile_times = _allgather(
        jnp.asarray([compile_plus_first_run_s], dtype=jnp.float32),
        sync_enabled,
    ).reshape(-1)

    local_steps_per_run = local_device_count * batch_per_device * args.steps
    global_steps_per_run = global_device_count * batch_per_device * args.steps
    local_steps_per_second = [
        local_steps_per_run / wall_time for wall_time in wall_times
    ]

    if args.print_local:
        print(
            json.dumps(
                {
                    "kind": "pgx_tpu_throughput_local",
                    "host": socket.gethostname(),
                    "process_index": process_index,
                    "batch_per_device": batch_per_device,
                    "steps": args.steps,
                    "wall_s_per_repeat": wall_times,
                    "local_steps_per_second_per_repeat": local_steps_per_second,
                    "compile_plus_first_run_s": compile_plus_first_run_s,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if process_index != 0:
        return None

    max_wall_times = np.max(all_wall_times, axis=0)
    cluster_steps_per_second = [
        global_steps_per_run / float(wall_time) for wall_time in max_wall_times
    ]

    return {
        "kind": "pgx_tpu_throughput",
        "env_id": args.env_id,
        "action_mode": args.action_mode,
        "pgx_version": getattr(pgx, "__version__", "unknown"),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "host": socket.gethostname(),
        "process_count": process_count,
        "local_device_count": local_device_count,
        "global_device_count": global_device_count,
        "batch_per_device": batch_per_device,
        "local_batch": local_device_count * batch_per_device,
        "global_batch": global_device_count * batch_per_device,
        "steps_per_run": args.steps,
        "timed_repeats": args.repeats,
        "warmup_runs_after_compile": args.warmup_runs,
        "global_steps_per_run": global_steps_per_run,
        "max_compile_plus_first_run_s": float(np.max(all_compile_times)),
        "max_wall_s_per_repeat": [float(x) for x in max_wall_times],
        "cluster_steps_per_second_per_repeat": [
            float(x) for x in cluster_steps_per_second
        ],
        "cluster_steps_per_second_best": float(max(cluster_steps_per_second)),
        "cluster_steps_per_second_median": float(
            statistics.median(cluster_steps_per_second)
        ),
        "cluster_steps_per_second_mean": float(
            statistics.fmean(cluster_steps_per_second)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="chess")
    parser.add_argument(
        "--batch-per-device",
        nargs="+",
        default=["128"],
        help="One or more batch sizes per JAX device. Comma separated values are accepted.",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("first_legal", "random_legal"),
        default="first_legal",
    )
    parser.add_argument(
        "--no-multihost-sync",
        action="store_true",
        help="Disable cross-host barriers/allgather. Intended for CPU/local smoke tests.",
    )
    parser.add_argument("--print-local", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")

    batch_sizes = _parse_positive_ints(args.batch_per_device)

    if jax.process_index() == 0:
        print(
            json.dumps(
                {
                    "kind": "pgx_tpu_throughput_header",
                    "env_id": args.env_id,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "pgx_version": getattr(pgx, "__version__", "unknown"),
                    "jax_version": jax.__version__,
                    "jax_backend": jax.default_backend(),
                    "process_count": jax.process_count(),
                    "local_device_count": jax.local_device_count(),
                    "global_device_count": jax.device_count(),
                    "batch_per_device": batch_sizes,
                    "steps": args.steps,
                    "repeats": args.repeats,
                    "warmup_runs": args.warmup_runs,
                    "action_mode": args.action_mode,
                    "multihost_sync": not args.no_multihost_sync
                    and jax.process_count() > 1,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for batch_per_device in batch_sizes:
        result = run_one(args, batch_per_device)
        if result is not None:
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
