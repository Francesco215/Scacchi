"""End-to-end check of backup_kernel=fused against backup_kernel=cuda.

Runs the trainer's own functions (``make_selfplay``, the training iteration,
``make_mcts_evaluate``) under both settings with the same keys and model, then
compares every output array bitwise and reports wall times (median of
``--repeats`` after one compile run). Benchmark and check only.
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
import numpy as np

from profile_search import load_config  # noqa: E402


def timed(fn, repeats):
    started = time.perf_counter()
    out = jax.block_until_ready(fn(0))
    first = time.perf_counter() - started
    runs = []
    for index in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(fn(index + 1))
        runs.append(time.perf_counter() - started)
    return out, {"first_s": round(first, 3), "median_s": round(statistics.median(runs), 4), "runs_s": [round(r, 4) for r in runs]}


def leaves_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    bad = []
    for i, (x, y) in enumerate(zip(la, lb)):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append((i, "shape/dtype", x.shape, y.shape))
        elif x.dtype.kind == "f":
            if not np.array_equal(x.view(np.uint8), y.view(np.uint8)):
                bad.append((i, "values", float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))))
        elif not np.array_equal(x, y):
            bad.append((i, "values", int(np.sum(x != y))))
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="hex7")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--kernels", nargs="+", default=["cuda", "fused"],
                        help="variants: a backup_kernel name, optionally label=kernel;key=value;...")
    parser.add_argument("--what", nargs="+", default=["selfplay", "iteration", "eval"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from scacchi.checkpoint import from_pretrained
    from scacchi.distributed import make_batch_parallel
    from scacchi.envs import make_env
    from scacchi.evaluations import load_eval_baseline, make_mcts_evaluate
    from scacchi.network import build_model
    from scacchi.pipeline import build_optimizer, make_training_iteration
    from scacchi.play import make_selfplay

    results, outputs = {}, {}
    for variant in args.kernels:
        kernel, *extra = variant.split(";")
        label = variant
        if "=" in kernel:
            label, kernel = kernel.split("=", 1)
            extra = [f"selfplay.search.binned_thompson.{item}" for item in extra]
        config = load_config(args.config_name, args.batch, [
            f"selfplay.search.binned_thompson.backup_kernel={kernel}", f"eval.batch_size={args.eval_batch}", *extra, *args.override])
        kernel = label
        env = make_env(config.env.id, config.env.board_size)
        if args.checkpoint:
            model = from_pretrained(args.checkpoint, env, rngs=nnx.Rngs(0))
        else:
            model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(0))
        parallel = make_batch_parallel(config)
        entry = {}

        def in_mesh(fn):
            def run(index):
                with parallel.mesh_context():
                    return fn(index)
            return run

        if "selfplay" in args.what:
            selfplay = make_selfplay(env, config, parallel=parallel)
            out, entry["selfplay"] = timed(in_mesh(lambda i: selfplay(model, jax.random.PRNGKey(i))), args.repeats)
            outputs.setdefault("selfplay", {})[kernel] = jax.device_get(out)
        if "iteration" in args.what:
            trained = nnx.clone(model)
            optimizer = build_optimizer(trained, config)
            iteration = make_training_iteration(env, config, parallel=parallel)
            out, entry["selfplay_plus_training"] = timed(in_mesh(lambda i: iteration(trained, optimizer, jax.random.PRNGKey(100 + i))), args.repeats)
        if "eval" in args.what:
            baseline = load_eval_baseline(config, env, parallel)
            evaluate = make_mcts_evaluate(env, config, baseline, parallel=parallel)
            out, entry["eval"] = timed(in_mesh(lambda i: evaluate(jax.random.PRNGKey(200 + i), model)), args.repeats)
            outputs.setdefault("eval", {})[kernel] = jax.device_get(out)
        results[kernel] = entry
        print(json.dumps({kernel: entry}), flush=True)

    first = args.kernels[0].split(';')[0].split('=')[0]
    for what, by_kernel in outputs.items():
        for kernel, out in by_kernel.items():
            if kernel == first:
                continue
            bad = leaves_equal(by_kernel[first], out)
            results.setdefault("compare", {})[f"{what}:{first}_vs_{kernel}"] = "identical" if not bad else bad[:10]
    print(json.dumps(results.get("compare", {})), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
