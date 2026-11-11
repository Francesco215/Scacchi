"""Compare the CUDA network (native_binned.network) with the flax network.

Same model, same observations (positions after random moves); reports the
largest differences of the logits and the beliefs, how often the belief's
most likely atom differs, and the time per call of both (median, jitted,
after warm-up). Benchmark and check only.
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
import jax.numpy as jnp
import numpy as np

from check_fused_search import build  # noqa: E402
from profile_search import random_positions  # noqa: E402


def timed(fn, *args, repeats=20):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - started)
    return 1e3 * statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="hex7")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--plies", type=int, nargs="+", default=[0, 10, 25])
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    from scacchi.native_binned.network import make_native_evaluator, supports
    from scacchi.play_search import make_evaluator

    config, env, model = build(args.config_name, args.batch, args.checkpoint, None)
    assert supports(model), "model is not the supported AZBinnedNet shape"

    @nnx.jit
    def xla(model, obs):
        out = make_evaluator(model)(obs)
        return out.logits, out.alpha_v, out.alpha_q

    @nnx.jit
    def native(model, obs):
        out = make_native_evaluator(model)(obs)
        return out.logits, out.alpha_v, out.alpha_q

    for plies in args.plies:
        states = jax.jit(random_positions, static_argnums=(0, 1, 2))(env, args.batch, plies, jax.random.PRNGKey(plies))
        obs = states.observation
        a = [np.asarray(x, np.float64) for x in xla(model, obs)]
        b = [np.asarray(x, np.float64) for x in native(model, obs)]
        legal = np.asarray(states.legal_action_mask)
        report = {
            "plies": plies,
            "logits_max_abs": float(np.max(np.abs(a[0] - b[0])[legal])),
            "logits_scale": float(np.max(np.abs(a[0])[legal])),
            "value_probs_max_abs": float(np.max(np.abs(a[1] - b[1]))),
            "q_probs_max_abs": float(np.max(np.abs(a[2] - b[2]))),
            "value_mode_differs": float(np.mean(np.argmax(a[1], -1) != np.argmax(b[1], -1))),
            "q_mode_differs": float(np.mean(np.argmax(a[2], -1) != np.argmax(b[2], -1))),
            "value_sum_err": float(np.max(np.abs(b[1].sum(-1) - 1))),
            "xla_ms": round(timed(xla, model, obs), 3),
            "native_ms": round(timed(native, model, obs), 3),
        }
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
