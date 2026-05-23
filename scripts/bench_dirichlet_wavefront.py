from __future__ import annotations

import argparse
import time
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from scacchi.dirichlet_tree.arena_search import BatchedPosteriorArenaSearch
from scacchi.dirichlet_tree.search import search_config_from_any


class ToyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    terminated: jax.Array
    rewards: jax.Array


class ToyEnv:
    num_actions = 4

    def step(self, state: ToyState, action: jax.Array) -> ToyState:
        del action
        return ToyState(
            observation=state.observation + 1.0,
            legal_action_mask=jnp.array([True, True, True, True]),
            current_player=1 - state.current_player,
            terminated=jnp.array(False),
            rewards=jnp.array([0.0, 0.0], dtype=jnp.float32),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=int, default=256)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--policy-mc-samples", type=int, default=8)
    args = parser.parse_args()

    root_state_batch = _make_root_state_batch(args.roots, args.max_depth + args.simulations + 8)

    def leaf_evaluator(obs):
        batch = obs.shape[0]
        return (
            jnp.zeros((batch, 4), dtype=jnp.float32),
            jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1)),
            jnp.ones((batch, 4, 3), dtype=jnp.float32),
        )

    config = SimpleNamespace(
        num_simulations=args.simulations,
        wavefront_max_depth=args.max_depth,
        search_eval_batch_size=args.roots,
        c_leaf=1.0,
        c_terminal=8.0,
        c_state=0.1,
        c_value_search=1.0,
        policy_mc_samples=args.policy_mc_samples,
        wavefront_num_lanes_per_root=1,
        wavefront_final_action_mode="argmax_q_mean",
    )
    total = args.roots * args.simulations
    for repeat in range(args.repeats):
        search = BatchedPosteriorArenaSearch(env=ToyEnv(), rng_key=jax.random.PRNGKey(repeat))
        start = time.perf_counter()
        output = search.search_state_batch(
            root_state_batch,
            leaf_evaluator,
            search_config_from_any(config, num_roots=args.roots),
        )
        jax.block_until_ready(output.action)
        elapsed = time.perf_counter() - start
        print(f"repeat={repeat + 1} completed_evals_per_s={total / elapsed:.1f}")
        print(f"elapsed_s={elapsed:.6f}")
        print(f"roots={args.roots}")
        print(f"simulations={args.simulations}")


def _make_root_state_batch(roots: int, spacing: int) -> ToyState:
    observations = (jnp.arange(roots, dtype=jnp.float32) * float(spacing)).reshape((roots, 1))
    return ToyState(
        observation=observations,
        legal_action_mask=jnp.tile(jnp.array([[True, True, True, True]]), (roots, 1)),
        current_player=jnp.zeros((roots,), dtype=jnp.int32),
        terminated=jnp.zeros((roots,), dtype=jnp.bool_),
        rewards=jnp.zeros((roots, 2), dtype=jnp.float32),
    )


if __name__ == "__main__":
    main()
