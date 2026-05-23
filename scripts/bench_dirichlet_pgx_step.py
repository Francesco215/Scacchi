from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from scacchi.envs import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--board-size", type=int, default=5)
    args = parser.parse_args()

    env = make_env("hex", args.board_size)
    states = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(0), args.batch))
    actions = jnp.zeros((args.batch,), dtype=jnp.int32)
    step_fn = jax.jit(jax.vmap(env.step))
    jax.block_until_ready(step_fn(states, actions))
    start = time.perf_counter()
    for _ in range(args.iters):
        states = step_fn(states, actions)
    jax.block_until_ready(states)
    elapsed = time.perf_counter() - start
    print(f"steps_per_s={args.batch * args.iters / elapsed:.1f}")
    print(f"batch={args.batch}")


if __name__ == "__main__":
    main()
