from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from scacchi.envs import make_env
from scacchi.posterior_tree import (
    run_posterior_tree_search,
    run_posterior_tree_search_state_batch,
    split_batched_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--simulations", type=int, default=1)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--prefill-steps", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=["state_batch", "split_list", "object_tree", "both"],
        default="both",
    )
    parser.add_argument("--policy-mc-samples", type=int, default=1)
    parser.add_argument("--lanes-per-root", type=int, default=1)
    args = parser.parse_args()

    env = make_env("hex", args.board_size)
    root_state_batch = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(0), args.batch))
    if args.prefill_steps:
        root_state_batch = _prefill_hex_states(
            env,
            root_state_batch,
            batch=args.batch,
            board_size=args.board_size,
            steps=args.prefill_steps,
        )
    config = SimpleNamespace(
        search_policy="posterior_tree_wavefront",
        num_simulations=args.simulations,
        search_eval_batch_size=args.batch,
        c_leaf=1.0,
        c_terminal=8.0,
        c_state=0.1,
        c_value_search=1.0,
        policy_mc_samples=args.policy_mc_samples,
        backup_mc_samples=1,
        selfplay_action_source="scalar_q_argmax",
        wavefront_num_lanes_per_root=args.lanes_per_root,
        wavefront_max_depth=max(16, args.simulations + args.board_size * args.board_size + 2),
        wavefront_final_action_mode="argmax_q_mean",
    )

    modes = ["state_batch", "split_list"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for repeat in range(args.repeats):
            elapsed = _run_once(
                mode=mode,
                env=env,
                root_state_batch=root_state_batch,
                config=config,
                rng_key=jax.random.PRNGKey(repeat),
            )
            total = args.batch * args.simulations
            print(
                f"mode={mode} repeat={repeat + 1} elapsed_s={elapsed:.6f} "
                f"completed_evals_per_s={total / elapsed:.1f} "
                f"batch={args.batch} simulations={args.simulations} "
                f"board_size={args.board_size} prefill_steps={args.prefill_steps} "
                f"lanes_per_root={args.lanes_per_root}",
                flush=True,
            )


def _run_once(*, mode: str, env, root_state_batch, config, rng_key) -> float:
    def leaf_evaluator(obs):
        batch = obs.shape[0]
        num_actions = env.num_actions
        return (
            jnp.zeros((batch, num_actions), dtype=jnp.float32),
            jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1)),
            jnp.ones((batch, num_actions, 3), dtype=jnp.float32),
        )

    start = time.perf_counter()
    if mode == "state_batch":
        output = run_posterior_tree_search_state_batch(
            env=env,
            root_state_batch=root_state_batch,
            leaf_evaluator=leaf_evaluator,
            rng_key=rng_key,
            config=config,
        )
    elif mode == "split_list":
        output = run_posterior_tree_search(
            env=env,
            root_states=split_batched_state(root_state_batch),
            leaf_evaluator=leaf_evaluator,
            rng_key=rng_key,
            config=config,
        )
    elif mode == "object_tree":
        object_config = SimpleNamespace(**vars(config))
        object_config.search_policy = "posterior_tree"
        output = run_posterior_tree_search(
            env=env,
            root_states=split_batched_state(root_state_batch),
            leaf_evaluator=leaf_evaluator,
            rng_key=rng_key,
            config=object_config,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")
    jax.block_until_ready(output.action)
    return time.perf_counter() - start


def _prefill_hex_states(env, state_batch, *, batch: int, board_size: int, steps: int):
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


if __name__ == "__main__":
    main()
