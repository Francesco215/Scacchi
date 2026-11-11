"""The binned network's forward pass as CUDA kernels (``network.cu``).

``make_native_evaluator(model)`` behaves like ``play_search.make_evaluator``
for an ``AZBinnedNet`` with 128 channels and 6 ResNet-v2 blocks on Hex, but
runs the 12 residual convolutions on the tensor cores with BatchNorm, ReLU and
the residual add fused into each convolution's epilogue, and the three heads in
one kernel. Same precision as the flax module (bf16 layers, f32 accumulation,
f32 belief layers); the f32 sums are added in another order than cuDNN's, so
results match the XLA network to bf16 rounding, not bitwise. A position's
result does not depend on the batch it is evaluated in.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

CH = 128
BLOCKS = 6


class PackedNetwork(NamedTuple):
    stem_w: jax.Array   # [128, 36] bf16
    trunk_w: jax.Array  # [12, 128, 1152] bf16
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


def pack(model) -> PackedNetwork:
    rows = [_bf16(model.conv.bias[...])]
    trunk = []
    for block in model.blocks:
        rows += _bn(block.bn1) + [_bf16(block.conv1.bias[...])] + _bn(block.bn2) + [_bf16(block.conv2.bias[...])]
        trunk += [_conv3x3(block.conv1.kernel[...]), _conv3x3(block.conv2.kernel[...])]
    rows += _bn(model.bn)
    head_kernels = [model.policy_conv.kernel[...], model.value_conv.kernel[...], model.q_dir_conv.kernel[...]]
    head_cw = jnp.concatenate([k.reshape(k.shape[-2], k.shape[-1]).T for k in head_kernels], axis=0).astype(jnp.bfloat16)
    head_bn = [_bn(model.policy_bn), _bn(model.value_bn), _bn(model.q_dir_bn)]
    head_cv = jnp.stack([
        jnp.concatenate([_bf16(model.policy_conv.bias[...]), _bf16(model.value_conv.bias[...]), _bf16(model.q_dir_conv.bias[...])]),
        *[jnp.concatenate([bn[i] for bn in head_bn]) for i in range(3)],
    ])
    return PackedNetwork(
        stem_w=_conv3x3(model.conv.kernel[...]),
        trunk_w=jnp.stack(trunk),
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
    out = jax.ffi.ffi_call(NETWORK_TARGET, results)(observation.astype(jnp.bool_), *packed)
    return out[0], out[1], out[2]


def make_native_evaluator(model):
    """``play_search.make_evaluator`` for ``AZBinnedNet`` with the CUDA forward pass."""
    from .. import binned
    from ..play_search import EvaluatorOutput

    packed = pack(model)
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
