"""Accuracy of the INT8 search network against the bf16 network, next to FP8,
and network_kernel=int8 against a JAX emulation of the same quantization.

The emulations are the flax network's layers in float32 with the roundings put
in by hand, as in ``check_fp8_network.py``.

INT8: the inputs of the 12 residual convolutions (after BatchNorm and ReLU,
so >= 0) are rounded to unsigned 8-bit integers with one scale per channel,
and the weights to signed 8-bit integers with one scale per output channel
after the input scales are folded in. The sums are then exact integers. The
input scale of a channel comes from its BatchNorm: the layer's output before
the ReLU has mean ``bias`` and standard deviation ``|scale|`` on the training
distribution, so the range [0, bias + clip * |scale|] is mapped to 0..255.
``--clip`` sets that number of standard deviations.

Reports, per ply, the mean and largest |logit difference| to the bf16 network
on legal moves, the largest belief difference, and how often the most likely
atom of a belief changes; then the same for the kernel against the emulation
at the packing's own clip (they should agree to f32/bf16 rounding).

Measured on Hex 7 (binned_7, 1024 positions, plies 0/10/25): mean |logit
difference| to bf16 0.37/0.12/0.06 for FP8 and 0.04/0.04/0.03 for INT8; Q
belief mode differs in 6.1/4.1/1.2% of moves for FP8 and 0/1.2/0.6% for INT8.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from check_fp8_network import HI, bf, bn, conv, reference as fp8_reference  # noqa: E402
from check_fused_search import build  # noqa: E402
from profile_search import random_positions  # noqa: E402

from scacchi.native_binned.network import INT8_CLIP, int8_input_step, make_native_evaluator  # noqa: E402


def int8_conv(z, kernel, step):
    q = jnp.clip(jnp.round(z / step), 0.0, 255.0)
    folded = kernel.astype(jnp.float32) * step[None, None, :, None]
    wstep = jnp.maximum(jnp.max(jnp.abs(folded), axis=(0, 1, 2)), 1e-30) / 127.0
    qw = jnp.clip(jnp.round(folded / wstep), -127.0, 127.0)
    saturated = jnp.mean(z / step > 255.5)
    return conv(q, qw) * wstep, saturated


def int8_reference(model, obs, clip):
    relu = jax.nn.relu
    x = bf(bf(conv(obs.astype(jnp.float32), bf(model.conv.kernel[...]))) + bf(model.conv.bias[...]))
    saturated = []
    for block in model.blocks:
        y, s = int8_conv(relu(bn(block.bn1, x)), block.conv1.kernel[...], int8_input_step(block.bn1, clip))
        saturated.append(s)
        y = bf(bf(y) + bf(block.conv1.bias[...]))
        y, s = int8_conv(relu(bn(block.bn2, y)), block.conv2.kernel[...], int8_input_step(block.bn2, clip))
        saturated.append(s)
        y = bf(bf(y) + bf(block.conv2.bias[...]))
        x = bf(y + x)
    h = relu(bn(model.bn, x))

    def head_conv(c, norm):
        return relu(bn(norm, bf(bf(conv(h, bf(c.kernel[...]))) + bf(c.bias[...]))))

    def dense_bf16(layer, v):
        return bf(bf(jnp.dot(v, bf(layer.kernel[...]), precision=HI)) + bf(layer.bias[...]))

    batch = obs.shape[0]
    logits = dense_bf16(model.policy_linear, head_conv(model.policy_conv, model.policy_bn).reshape(batch, -1))
    v = relu(dense_bf16(model.value_linear, head_conv(model.value_conv, model.value_bn).reshape(batch, -1)))
    v = jnp.dot(v, model.value_dir_out.kernel[...], precision=HI) + model.value_dir_out.bias[...]
    q = relu(dense_bf16(model.q_dir_linear, head_conv(model.q_dir_conv, model.q_dir_bn).reshape(batch, -1)))
    q = jnp.dot(q, model.q_dir_out.kernel[...], precision=HI) + model.q_dir_out.bias[...]
    return logits, jax.nn.softmax(v, -1), jax.nn.softmax(q.reshape(batch, logits.shape[-1], -1), -1), jnp.max(jnp.stack(saturated))


def compare(a, b, legal):
    a = [np.asarray(x, np.float64) for x in a[:3]]
    b = [np.asarray(x, np.float64) for x in b[:3]]
    return {
        "logit_mean": round(float(np.mean(np.abs(a[0] - b[0])[legal])), 4),
        "logit_max": round(float(np.max(np.abs(a[0] - b[0])[legal])), 3),
        "v_probs_max": round(float(np.max(np.abs(a[1] - b[1]))), 4),
        "q_probs_mean": round(float(np.mean(np.abs(a[2] - b[2])[legal])), 5),
        "q_mode_differs": round(float(np.mean((np.argmax(a[2], -1) != np.argmax(b[2], -1))[legal])), 4),
        "v_mode_differs": round(float(np.mean(np.argmax(a[1], -1) != np.argmax(b[1], -1))), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="hex7")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--plies", type=int, nargs="+", default=[0, 10, 25])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--clip", type=float, nargs="+", default=[INT8_CLIP])
    args = parser.parse_args()

    from scacchi.play_search import make_evaluator

    config, env, model = build(args.config_name, args.batch, args.checkpoint, None)

    @nnx.jit
    def exact(model, obs):
        out = make_evaluator(model)(obs)
        return out.logits, out.alpha_v, out.alpha_q

    @nnx.jit
    def kernel(model, obs):
        out = make_native_evaluator(model, "int8")(obs)
        return out.logits, out.alpha_v, out.alpha_q

    fp8 = nnx.jit(lambda model, obs: fp8_reference(model, obs))
    int8 = nnx.jit(lambda model, obs, clip: int8_reference(model, obs, clip))
    for plies in args.plies:
        states = jax.jit(random_positions, static_argnums=(0, 1, 2))(env, args.batch, plies, jax.random.PRNGKey(plies))
        legal = np.asarray(states.legal_action_mask)
        truth = exact(model, states.observation)
        print(json.dumps({"plies": plies, "kind": "fp8", **compare(truth, fp8(model, states.observation), legal)}), flush=True)
        for clip in args.clip:
            out = int8(model, states.observation, clip)
            print(json.dumps({"plies": plies, "kind": f"int8 clip {clip:g}", **compare(truth, out, legal),
                              "saturated_worst_layer": round(float(out[3]), 5)}), flush=True)
        emulation = int8(model, states.observation, INT8_CLIP)
        print(json.dumps({"plies": plies, "kind": "kernel vs emulation", **compare(emulation, kernel(model, states.observation), legal)}),
              flush=True)


if __name__ == "__main__":
    main()
