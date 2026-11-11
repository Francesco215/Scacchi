"""Check network_kernel=fp8 against a JAX emulation of the same quantization.

The reference runs the flax network's layers in float32 with the FP8 kernel's
roundings put in by hand: trunk weights to e4m3 with one scale per output
channel, convolution inputs (after BatchNorm and ReLU) to e4m3, and the bf16
roundings of the flax module elsewhere. Kernel and reference should then agree
to f32/bf16 rounding; the gap between either and the bf16 network is the FP8
approximation itself. Measured on Hex 7 (binned_7, 1024 positions): kernel vs
reference mean |logit difference| <= 0.0035, FP8 vs bf16 0.06-0.38.
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
# XLA drops f32 -> e4m3/bf16 -> f32 round trips by default ("excess precision"),
# which would silently remove the reference's roundings.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from check_fused_search import build  # noqa: E402
from profile_search import random_positions  # noqa: E402

HI = jax.lax.Precision.HIGHEST


def bf(x):
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def e4m3(x):
    return jnp.clip(x, -448.0, 448.0).astype(jnp.float8_e4m3fn).astype(jnp.float32)


def bn(module, x):
    mul = jax.lax.rsqrt(module.var[...] + module.epsilon) * module.scale[...]
    return bf((x - module.mean[...]) * mul + module.bias[...])


def conv(x, kernel):
    return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"), precision=HI)


def reference(model, obs, quantize=True):
    from scacchi.native_binned.network import _quantize_e4m3

    q8 = e4m3 if quantize else (lambda v: v)

    def weights(k):
        if not quantize:
            return bf(k)
        q, s = _quantize_e4m3(jnp.transpose(k, (3, 0, 1, 2)).reshape(k.shape[3], -1)[None])
        kq = jax.lax.bitcast_convert_type(q[0], jnp.float8_e4m3fn).astype(jnp.float32) * s[0][:, None]
        return jnp.transpose(kq.reshape(k.shape[3], 3, 3, k.shape[2]), (1, 2, 3, 0))

    relu = jax.nn.relu
    x = bf(bf(conv(obs.astype(jnp.float32), bf(model.conv.kernel[...]))) + bf(model.conv.bias[...]))
    for block in model.blocks:
        z = q8(relu(bn(block.bn1, x)))
        y = bf(bf(conv(z, weights(block.conv1.kernel[...]))) + bf(block.conv1.bias[...]))
        z = q8(relu(bn(block.bn2, y)))
        y = bf(bf(conv(z, weights(block.conv2.kernel[...]))) + bf(block.conv2.bias[...]))
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
    return logits, jax.nn.softmax(v, -1), jax.nn.softmax(q.reshape(batch, logits.shape[-1], -1), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="hex7")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--plies", type=int, nargs="+", default=[0, 10, 25])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--no-quant", action="store_true", help="check the reference against the bf16 kernel")
    args = parser.parse_args()

    from scacchi.native_binned.network import make_native_evaluator

    config, env, model = build(args.config_name, args.batch, args.checkpoint, None)
    ref = nnx.jit(lambda model, obs: reference(model, obs, quantize=not args.no_quant))

    @nnx.jit
    def kernel(model, obs):
        out = make_native_evaluator(model, fp8=not args.no_quant)(obs)
        return out.logits, out.alpha_v, out.alpha_q

    for plies in args.plies:
        states = jax.jit(random_positions, static_argnums=(0, 1, 2))(env, args.batch, plies, jax.random.PRNGKey(plies))
        a = [np.asarray(x, np.float64) for x in ref(model, states.observation)]
        b = [np.asarray(x, np.float64) for x in kernel(model, states.observation)]
        legal = np.asarray(states.legal_action_mask)
        print(json.dumps({
            "plies": plies,
            "logits_max_abs": float(np.max(np.abs(a[0] - b[0])[legal])),
            "value_probs_max_abs": float(np.max(np.abs(a[1] - b[1]))),
            "q_probs_max_abs": float(np.max(np.abs(a[2] - b[2]))),
            "q_mode_differs": float(np.mean(np.argmax(a[2], -1) != np.argmax(b[2], -1))),
        }), flush=True)


if __name__ == "__main__":
    main()
