"""The binned network's forward pass as CUDA kernels (``network.cu``)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

CH = 128
BLOCKS = 6
CACHE_SLOTS = 65536

KERNELS = {"cuda": 0, "fp8": 1, "int8": 2, "int8_14": 3}
INT8_CLIP = 6.0


class PackedNetwork(NamedTuple):
    stem_w: jax.Array   # [128, 36] bf16
    trunk_w: jax.Array  # [12, 128, 1152] bf16, e4m3 bytes (uint8) for fp8, or int8
    wscale: jax.Array   # [12, 128] f32
    qinv: jax.Array     # [12, 128] f32
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


class PositionCache(NamedTuple):
    keys: jax.Array   # [CACHE_SLOTS, 10] uint32 (40 bytes/entry = CacheKey)
    locks: jax.Array  # [CACHE_SLOTS] uint32
    v: jax.Array      # [CACHE_SLOTS, K] float32
    q: jax.Array      # [CACHE_SLOTS, A, K] float32


def empty_position_cache(num_actions: int, atoms: int, enabled: bool = True) -> PositionCache:
    slots = CACHE_SLOTS if enabled else 1
    return PositionCache(
        keys=jnp.zeros((slots, 10), jnp.uint32),
        locks=jnp.zeros((slots,), jnp.uint32),
        v=jnp.zeros((slots, atoms), jnp.float32),
        q=jnp.zeros((slots, num_actions, atoms), jnp.float32),
    )


def supports(model) -> bool:
    from ..network import AZBinnedNet

    return (isinstance(model, AZBinnedNet) and model.resnet_v2 and model.num_channels == CH
            and model.num_blocks == BLOCKS and model.dtype == jnp.bfloat16 and model.bins is not None
            and model.bins.count <= 16)


def _bf16(x):
    return jnp.asarray(x).astype(jnp.bfloat16).astype(jnp.float32)


def _bn(module):
    mul = jax.lax.rsqrt(module.var[...] + module.epsilon) * module.scale[...]
    return [module.mean[...], mul, module.bias[...]]


def _conv3x3(kernel):
    kernel = jnp.asarray(kernel).astype(jnp.bfloat16)
    return jnp.transpose(kernel, (3, 0, 1, 2)).reshape(kernel.shape[3], -1)


E4M3_MAX = 448.0


def _quantize_e4m3(kernels):
    amax = jnp.max(jnp.abs(kernels), axis=-1)
    scale = jnp.where(amax > 0, amax / E4M3_MAX, 1.0).astype(jnp.float32)
    q = (kernels / scale[..., None]).astype(jnp.float8_e4m3fn)
    return jax.lax.bitcast_convert_type(q, jnp.uint8), scale


def int8_input_step(norm, clip: float = INT8_CLIP):
    top = norm.bias[...].astype(jnp.float32) + clip * jnp.abs(norm.scale[...].astype(jnp.float32))
    return jnp.maximum(top, 1e-6) / 255.0


def _quantize_int8(kernels, steps):
    folded = kernels * jnp.tile(steps, (1, 9))[:, None, :]
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
    trunk = jnp.stack([jnp.transpose(k, (3, 0, 1, 2)).reshape(k.shape[3], -1) for k in trunk]).astype(jnp.float32)
    qinv = jnp.ones(trunk.shape[:2], jnp.float32)
    if kernel == "fp8":
        trunk_w, wscale = _quantize_e4m3(trunk)
    elif kernel in ("int8", "int8_14"):
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


def apply(packed: PackedNetwork, observation: jax.Array, num_actions: int, atoms: int, *, kernel_override: int | None = None):
    """Logits [B, A] f32, value beliefs [B, K] and Q beliefs [B, A, K] (softmax)."""
    from ._library import NETWORK_TARGET, register_network

    register_network()
    batch, side = observation.shape[0], observation.shape[1]
    kernel = kernel_override if kernel_override is not None else {
        jnp.dtype(jnp.bfloat16): 0, jnp.dtype(jnp.uint8): 1, jnp.dtype(jnp.int8): 2
    }[packed.trunk_w.dtype]
    scratch_rows = 1 if (kernel == 2 and 3 <= side <= 11 and batch < 170) else (batch * side * side)
    results = [
        jax.ShapeDtypeStruct((batch, num_actions), jnp.float32),
        jax.ShapeDtypeStruct((batch, atoms), jnp.float32),
        jax.ShapeDtypeStruct((batch, num_actions, atoms), jnp.float32),
        *[jax.ShapeDtypeStruct((scratch_rows, CH), jnp.bfloat16)] * 3,
    ]
    out = jax.ffi.ffi_call(NETWORK_TARGET, results)(observation.astype(jnp.bool_), *packed, kernel=np.int64(kernel))
    return out[0], out[1], out[2]


def apply_compact(
    packed: PackedNetwork,
    obs_compact: jax.Array,
    d_boards: jax.Array,
    miss_keys: jax.Array,
    cache: PositionCache,
    num_actions: int,
    atoms: int,
    *,
    use_cache: bool = True,
    small_hint: bool = False,
) -> tuple[jax.Array, jax.Array, PositionCache]:
    """Evaluate only `*d_boards` compacted boards with INT8 + GPU position cache."""
    from ._library import NETWORK_COMPACT_TARGET, register_network_compact

    register_network_compact()
    batch, side = obs_compact.shape[0], obs_compact.shape[1]
    scratch_rows = 1 if (small_hint or batch < 170) else (batch * side * side)
    results = [
        jax.ShapeDtypeStruct((batch, atoms), jnp.float32),
        jax.ShapeDtypeStruct((batch, num_actions, atoms), jnp.float32),
        jax.ShapeDtypeStruct(cache.keys.shape, cache.keys.dtype),
        jax.ShapeDtypeStruct(cache.locks.shape, cache.locks.dtype),
        jax.ShapeDtypeStruct(cache.v.shape, cache.v.dtype),
        jax.ShapeDtypeStruct(cache.q.shape, cache.q.dtype),
        *[jax.ShapeDtypeStruct((scratch_rows, CH), jnp.bfloat16)] * 3,
    ]
    # Cache arrays are inputs 20, 21, 22, 23 and outputs 2, 3, 4, 5 -> alias in-place!
    aliases = {20: 2, 21: 3, 22: 4, 23: 5}
    call = jax.ffi.ffi_call(NETWORK_COMPACT_TARGET, results, input_output_aliases=aliases)
    out = call(
        obs_compact.astype(jnp.bool_),
        d_boards.astype(jnp.int32),
        miss_keys,
        *packed,
        cache.keys,
        cache.locks,
        cache.v,
        cache.q,
        use_cache=np.int64(1 if use_cache else 0),
        small_hint=np.int64(1 if small_hint else 0),
    )
    return out[0], out[1], PositionCache(keys=out[2], locks=out[3], v=out[4], q=out[5])


def make_native_evaluator(model, kernel: str = "cuda"):
    """``play_search.make_evaluator`` for ``AZBinnedNet`` with the CUDA forward pass."""
    from .. import binned
    from ..play_search import EvaluatorOutput

    packed = pack(model, kernel)
    support = jnp.asarray(binned.support(model.bins))
    num_actions = packed.pol_w.shape[1]
    atoms = packed.vout_w.shape[1]
    override = 3 if kernel == "int8_14" else None

    def evaluator(observation):
        logits, value_probs, action_probs = apply(packed, observation, num_actions, atoms, kernel_override=override)
        return EvaluatorOutput(
            logits=logits,
            value=2.0 * binned.mean(value_probs, support) - 1.0,
            alpha_v=value_probs,
            alpha_q=action_probs,
            support=support,
        )

    evaluator.packed = packed  # type: ignore[attr-defined]
    evaluator.kernel = kernel  # type: ignore[attr-defined]
    evaluator.support = support  # type: ignore[attr-defined]
    return evaluator
