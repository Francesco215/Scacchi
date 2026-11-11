"""The binned network's forward pass as CUDA kernels (``network.cu``).

``make_native_evaluator(model)`` behaves like ``play_search.make_evaluator``
for an ``AZBinnedNet`` with 128 channels and 6 ResNet-v2 blocks on Hex, but
runs the 12 residual convolutions on the tensor cores with BatchNorm, ReLU and
the residual add fused into each convolution's epilogue, and the three heads in
one kernel. Same precision as the flax module (bf16 layers, f32 accumulation,
f32 belief layers); the f32 sums are added in another order than cuDNN's, so
results match the XLA network to bf16 rounding, not bitwise. A position's
result does not depend on the batch it is evaluated in.

``kernel="fp8"`` runs the 12 residual convolutions on FP8 (e4m3) inputs with
f32 accumulation: weights get one scale per output channel (largest magnitude
mapped to 448, e4m3's largest value) and the convolution inputs (after
BatchNorm and ReLU) are rounded to e4m3 without a scale, saturating at 448.

``kernel="int8"`` runs them on 8-bit integers with exact int32 sums
(``conv_int8.cuh``; on an RTX 5090 the integer tensor-core instruction runs
twice as often as the FP8 one). A convolution's inputs are >= 0 (after
BatchNorm and ReLU) and are rounded to 0..255 with one step per channel; the
weights are rounded to -127..127 with one step per output channel after the
input steps are folded in. The input step of a channel comes from its
BatchNorm: before the ReLU the layer's output has mean ``bias`` and standard
deviation ``|scale|`` on the training distribution, so [0, bias +
INT8_CLIP * |scale|] is mapped to 0..255 and larger values saturate. On the
trained Hex 7 network this is 2 to 9 times closer to the bf16 network than FP8
(``scripts/check_int8_network.py``).

In both, stem, heads, the residual stream and every epilogue stay as above.
They approximate the bf16 network and are meant for search only; training
keeps bf16.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

CH = 128
BLOCKS = 6


KERNELS = {"cuda": 0, "fp8": 1, "int8": 2}
# Standard deviations of a BatchNorm output, above its mean, that the int8 grid of a convolution input covers.
INT8_CLIP = 6.0


class PackedNetwork(NamedTuple):
    stem_w: jax.Array   # [128, 36] bf16
    trunk_w: jax.Array  # [12, 128, 1152] bf16, e4m3 bytes (uint8) for fp8, or int8
    wscale: jax.Array   # [12, 128] f32 weight scale per output channel (ones for bf16)
    qinv: jax.Array     # [12, 128] f32 1 / input step of each convolution (int8; ones otherwise)
    vec: jax.Array      # [52, 128] f32
    head_cw: jax.Array  # [5, 128] bf16
    head_cv: jax.Array  # [4, 5] f32
    pol_w: jax.Array    # [2C, A] bf16
    pol_b: jax.Array    # [A] f32
    val_w: jax.Array    # [C, 128] bf16
    val_b: jax.Array    # [128] f32
    vout_w: jax.Array   # [128, K] f32
    vout_b: jax.Array   # [K] f32
    q_w: jax.Array      # [2C, 128] bf16
    q_b: jax.Array      # [128] f32
    qout_w: jax.Array   # [128, A K] f32
    qout_b: jax.Array   # [A K] f32


def supports(model) -> bool:
    from ..network import AZBinnedNet

    return (isinstance(model, AZBinnedNet) and model.resnet_v2 and model.num_channels == CH
            and model.num_blocks == BLOCKS and model.dtype == jnp.bfloat16 and model.bins is not None
            and model.bins.count <= 16)


def _bf16(x):
    """Values rounded to bf16, kept as float32 (biases the bf16 layers add)."""
    return jnp.asarray(x).astype(jnp.bfloat16).astype(jnp.float32)


def _bn(module):
    # flax nnx BatchNorm with running averages: (x - mean) * (rsqrt(var + eps) * scale) + bias.
    mul = jax.lax.rsqrt(module.var[...] + module.epsilon) * module.scale[...]
    return [module.mean[...], mul, module.bias[...]]


def _conv3x3(kernel):
    # HWIO [3, 3, cin, cout] -> [cout, (kh * 3 + kw) * cin + c]
    kernel = jnp.asarray(kernel).astype(jnp.bfloat16)
    return jnp.transpose(kernel, (3, 0, 1, 2)).reshape(kernel.shape[3], -1)


E4M3_MAX = 448.0


def _quantize_e4m3(kernels):
    """[convs, 128, 1152] weights -> e4m3 bytes and per-output-channel scales."""
    amax = jnp.max(jnp.abs(kernels), axis=-1)
    scale = jnp.where(amax > 0, amax / E4M3_MAX, 1.0).astype(jnp.float32)
    q = (kernels / scale[..., None]).astype(jnp.float8_e4m3fn)
    return jax.lax.bitcast_convert_type(q, jnp.uint8), scale


def int8_input_step(norm, clip: float = INT8_CLIP):
    """Step of the 0..255 grid for relu(norm(.)), per channel."""
    top = norm.bias[...].astype(jnp.float32) + clip * jnp.abs(norm.scale[...].astype(jnp.float32))
    return jnp.maximum(top, 1e-6) / 255.0


def _quantize_int8(kernels, steps):
    """[convs, 128, 1152] weights and [convs, 128] input steps -> int8 weights and per-output-channel steps."""
    folded = kernels * jnp.tile(steps, (1, 9))[:, None, :]  # column tap * 128 + c takes step[c]
    wstep = (jnp.maximum(jnp.max(jnp.abs(folded), axis=-1), 1e-30) / 127.0).astype(jnp.float32)
    q = jnp.clip(jnp.round(folded / wstep[..., None]), -127.0, 127.0).astype(jnp.int8)
    return q, wstep


def pack(model, kernel: str = "cuda") -> PackedNetwork:
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {sorted(KERNELS)}")
    rows = [_bf16(model.conv.bias[...])]
    trunk = []
    steps = []
    for block in model.blocks:
        rows += _bn(block.bn1) + [_bf16(block.conv1.bias[...])] + _bn(block.bn2) + [_bf16(block.conv2.bias[...])]
        trunk += [block.conv1.kernel[...], block.conv2.kernel[...]]
        steps += [int8_input_step(block.bn1), int8_input_step(block.bn2)]
    rows += _bn(model.bn)
    head_kernels = [model.policy_conv.kernel[...], model.value_conv.kernel[...], model.q_dir_conv.kernel[...]]
    head_cw = jnp.concatenate([k.reshape(k.shape[-2], k.shape[-1]).T for k in head_kernels], axis=0).astype(jnp.bfloat16)
    head_bn = [_bn(model.policy_bn), _bn(model.value_bn), _bn(model.q_dir_bn)]
    head_cv = jnp.stack([
        jnp.concatenate([_bf16(model.policy_conv.bias[...]), _bf16(model.value_conv.bias[...]), _bf16(model.q_dir_conv.bias[...])]),
        *[jnp.concatenate([bn[i] for bn in head_bn]) for i in range(3)],
    ])
    # HWIO [3, 3, cin, cout] -> [cout, (kh * 3 + kw) * cin + c], float32
    trunk = jnp.stack([jnp.transpose(k, (3, 0, 1, 2)).reshape(k.shape[3], -1) for k in trunk]).astype(jnp.float32)
    qinv = jnp.ones(trunk.shape[:2], jnp.float32)
    if kernel == "fp8":
        trunk_w, wscale = _quantize_e4m3(trunk)
    elif kernel == "int8":
        steps = jnp.stack(steps)
        trunk_w, wscale = _quantize_int8(trunk, steps)
        qinv = (1.0 / steps).astype(jnp.float32)
    else:
        trunk_w, wscale = trunk.astype(jnp.bfloat16), jnp.ones(trunk.shape[:2], jnp.float32)
    return PackedNetwork(
        stem_w=_conv3x3(model.conv.kernel[...]),
        trunk_w=trunk_w,
        wscale=wscale,
        qinv=qinv,
        vec=jnp.stack(rows).astype(jnp.float32),
        head_cw=head_cw,
        head_cv=head_cv.astype(jnp.float32),
        pol_w=model.policy_linear.kernel[...].astype(jnp.bfloat16),
        pol_b=_bf16(model.policy_linear.bias[...]),
        val_w=model.value_linear.kernel[...].astype(jnp.bfloat16),
        val_b=_bf16(model.value_linear.bias[...]),
        vout_w=model.value_dir_out.kernel[...].astype(jnp.float32),
        vout_b=model.value_dir_out.bias[...].astype(jnp.float32),
        q_w=model.q_dir_linear.kernel[...].astype(jnp.bfloat16),
        q_b=_bf16(model.q_dir_linear.bias[...]),
        qout_w=model.q_dir_out.kernel[...].astype(jnp.float32),
        qout_b=model.q_dir_out.bias[...].astype(jnp.float32),
    )


def apply(packed: PackedNetwork, observation: jax.Array, num_actions: int, atoms: int):
    """Logits [B, A] f32, value beliefs [B, K] and Q beliefs [B, A, K] (softmax)."""
    from ._library import NETWORK_TARGET, register_network

    register_network()
    batch, side = observation.shape[0], observation.shape[1]
    rows = batch * side * side
    results = [
        jax.ShapeDtypeStruct((batch, num_actions), jnp.float32),
        jax.ShapeDtypeStruct((batch, atoms), jnp.float32),
        jax.ShapeDtypeStruct((batch, num_actions, atoms), jnp.float32),
        *[jax.ShapeDtypeStruct((rows, CH), jnp.bfloat16)] * 3,
    ]
    kernel = {jnp.dtype(jnp.bfloat16): 0, jnp.dtype(jnp.uint8): 1, jnp.dtype(jnp.int8): 2}[packed.trunk_w.dtype]
    out = jax.ffi.ffi_call(NETWORK_TARGET, results)(observation.astype(jnp.bool_), *packed, kernel=np.int64(kernel))
    return out[0], out[1], out[2]


def make_native_evaluator(model, kernel: str = "cuda"):
    """``play_search.make_evaluator`` for ``AZBinnedNet`` with the CUDA forward pass (cuda, fp8 or int8)."""
    from .. import binned
    from ..play_search import EvaluatorOutput

    packed = pack(model, kernel)
    support = jnp.asarray(binned.support(model.bins))
    num_actions = packed.pol_w.shape[1]
    atoms = packed.vout_w.shape[1]

    def evaluator(observation):
        logits, value_probs, action_probs = apply(packed, observation, num_actions, atoms)
        return EvaluatorOutput(
            logits=logits,
            value=2.0 * binned.mean(value_probs, support) - 1.0,
            alpha_v=value_probs,
            alpha_q=action_probs,
            support=support,
        )

    return evaluator
