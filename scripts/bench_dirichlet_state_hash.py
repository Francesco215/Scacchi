from __future__ import annotations

import argparse
import time

import jax

from scacchi.dirichlet_tree.state_hash import canonical_state_key
from scacchi.envs import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--board-size", type=int, default=5)
    args = parser.parse_args()

    env = make_env("hex", args.board_size)
    states = jax.jit(jax.vmap(env.init))(jax.random.split(jax.random.PRNGKey(0), args.batch))
    key_fn = jax.jit(jax.vmap(canonical_state_key))
    jax.block_until_ready(key_fn(states))
    start = time.perf_counter()
    for _ in range(args.iters):
        keys = key_fn(states)
    jax.block_until_ready(keys)
    elapsed = time.perf_counter() - start
    print(f"states_per_s={args.batch * args.iters / elapsed:.1f}")
    print(f"batch={args.batch}")


if __name__ == "__main__":
    main()
