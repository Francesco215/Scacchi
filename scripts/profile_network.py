"""Evaluate the network alone on a batch of positions, for profiling.

Benchmark only. Same evaluator as the search (``play_search.make_evaluator``),
on observations of positions after ``--plies`` random moves. The timed calls
sit between cudaProfilerStart/Stop, so nsys/ncu with a profiler-API capture
range see only them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL,+CONVOLUTION --xla_gpu_graph_min_graph_size=1",
)

from flax import nnx
import jax

from profile_search import cudart, load_config, random_positions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="hex7")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    from scacchi.checkpoint import from_pretrained
    from scacchi.envs import make_env
    from scacchi.network import build_model
    from scacchi.play_search import make_evaluator

    config = load_config(args.config_name, args.batch, [])
    env = make_env(config.env.id, config.env.board_size)
    if args.checkpoint:
        model = from_pretrained(args.checkpoint, env, rngs=nnx.Rngs(0))
    else:
        model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(0))
    states = jax.jit(random_positions, static_argnums=(0, 1, 2))(env, args.batch, args.plies, jax.random.PRNGKey(3))
    observation = states.observation

    @nnx.jit
    def run(model, observation):
        out = make_evaluator(model)(observation)
        return out.alpha_v, out.alpha_q, out.logits

    jax.block_until_ready(run(model, observation))
    jax.block_until_ready(run(model, observation))
    lib = cudart()
    times = []
    lib.cudaProfilerStart()
    for _ in range(args.repeats):
        started = time.perf_counter()
        jax.block_until_ready(run(model, observation))
        times.append(time.perf_counter() - started)
    lib.cudaProfilerStop()
    print(json.dumps({"batch": args.batch, "median_ms": round(1e3 * statistics.median(times), 3), "min_ms": round(1e3 * min(times), 3)}))


if __name__ == "__main__":
    main()
