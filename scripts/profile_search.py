"""Run one binned search (or a whole self-play rollout / evaluation) for profiling.

Benchmark only. Wraps the timed region in cudaProfilerStart/Stop so that

    nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node \
        uv run python scripts/profile_search.py --what search ...

records exactly the repeats and nothing from compilation or warm-up.

``--what search``    the binned search on a batch of positions reached by
                     ``--plies`` uniformly random legal moves from the start.
``--what selfplay``  ``make_selfplay`` (one full rollout).
``--what eval``      ``make_mcts_evaluate`` (one full evaluation).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL,+CONVOLUTION --xla_gpu_graph_min_graph_size=1",
)

from flax import nnx
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp


def cudart():
    import nvidia.cu13 as wheel

    root = Path(next(iter(wheel.__path__)))
    for candidate in sorted(root.glob("lib/libcudart.so*")):
        return ctypes.CDLL(str(candidate))
    raise RuntimeError(f"libcudart not found under {root}")


def load_config(config_name: str, batch: int, extra: list[str]):
    from scacchi.types import load_config as build

    overrides = ["+algorithm=binned", f"selfplay.batch_size={batch}", "logging.wandb.enabled=false", *extra]
    with initialize_config_dir(config_dir=str(REPO / "scacchi" / "configs"), version_base=None):
        return build(compose(config_name=config_name, overrides=overrides))


def random_positions(env, batch: int, plies: int, key):
    """States after ``plies`` uniformly random legal moves (finished games stay put)."""
    init_key, key = jax.random.split(key)
    state = jax.vmap(env.init)(jax.random.split(init_key, batch))

    def body(state, step_key):
        logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
        action = jax.random.categorical(step_key, logits).astype(jnp.int32)
        stepped = jax.vmap(env.step)(state, action)
        keep = state.terminated
        return jax.tree.map(lambda old, new: jnp.where(keep.reshape(keep.shape + (1,) * (new.ndim - 1)), old, new), state, stepped), None

    if plies:
        state, _ = jax.lax.scan(body, state, jax.random.split(key, plies))
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="hex7")
    parser.add_argument("--what", choices=["search", "selfplay", "eval"], default="search")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backup", choices=["xla", "cuda", "fused"], default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from scacchi.checkpoint import from_pretrained
    from scacchi.distributed import make_batch_parallel
    from scacchi.envs import make_env
    from scacchi.network import build_model
    from scacchi.play_search import make_search, make_evaluator

    extra = [f"selfplay.search.binned_thompson.backup_kernel={args.backup}", *args.override]
    if args.what == "eval":
        extra.append(f"eval.batch_size={args.batch}")
    config = load_config(args.config_name, args.batch, extra)
    env = make_env(config.env.id, config.env.board_size)
    if args.checkpoint:
        model = from_pretrained(args.checkpoint, env, rngs=nnx.Rngs(0))
    else:
        model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(0))
    parallel = make_batch_parallel(config)

    if args.what == "search":
        search = make_search(env, make_evaluator(model), config.selfplay.search, config.training.losses.q_supervision)
        states = jax.jit(random_positions, static_argnums=(0, 1, 2))(env, args.batch, args.plies, jax.random.PRNGKey(7))
        run_jit = nnx.jit(lambda model, state, key: search(state, key))

        def run(index):
            return run_jit(model, states, jax.random.PRNGKey(index))
    elif args.what == "selfplay":
        from scacchi.play import make_selfplay

        selfplay = make_selfplay(env, config, parallel=parallel)

        def run(index):
            with parallel.mesh_context():
                return selfplay(model, jax.random.PRNGKey(index))
    else:
        from scacchi.evaluations import load_eval_baseline, make_mcts_evaluate

        baseline = load_eval_baseline(config, env, parallel)
        evaluate = make_mcts_evaluate(env, config, baseline, parallel=parallel)

        def run(index):
            with parallel.mesh_context():
                return evaluate(jax.random.PRNGKey(200 + index), model)

    started = time.perf_counter()
    jax.block_until_ready(run(0))
    first = time.perf_counter() - started
    jax.block_until_ready(run(1))  # second warm-up: allocator and autotuning settle

    lib = cudart()
    times = []
    lib.cudaProfilerStart()
    for index in range(args.repeats):
        started = time.perf_counter()
        jax.block_until_ready(run(index + 2))
        times.append(time.perf_counter() - started)
    lib.cudaProfilerStop()
    result = {
        "what": args.what, "batch": args.batch, "plies": args.plies, "backup": args.backup,
        "first_s": first, "median_s": statistics.median(times), "min_s": min(times), "times_s": times,
    }
    print(json.dumps(result))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
