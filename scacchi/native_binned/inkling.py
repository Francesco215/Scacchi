"""Pack and invoke the fused CUDA / INT8 kernel for AZInklingBinnedNet (`inkling_network.cu`)."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .network import PositionCache

INKLING_CH = 192
INKLING_HEADS = 6
INKLING_KV_HEADS = 2
INKLING_HEAD_DIM = 32
INKLING_QK_CLIP = 2.0 * math.sqrt(INKLING_HEAD_DIM)


class PackedInkling(NamedTuple):
    # Static metadata carried alongside arrays (not JAX buffers; see _buffers())
    # Keep ONLY JAX arrays in NamedTuple so it is a valid JAX pytree!
    stem_w: jax.Array         # [192, 27] bf16
    stem_b: jax.Array         # [192] f32
    emb_s: jax.Array          # [192] f32
    emb_b: jax.Array          # [192] f32
    attn_read_q: jax.Array    # [L, 3, 192] f32
    mlp_read_q: jax.Array     # [L, 192] f32
    attn_norm_sb: jax.Array   # [L, 6, 192] f32
    mlp_norm_sb: jax.Array    # [L, 2, 192] f32
    w_qkvg: jax.Array         # [L, 512, 192] bf16 or int8
    s_qkvg: jax.Array         # [L, 512] f32
    w_out: jax.Array          # [L, 192, 192] bf16 or int8
    s_out: jax.Array          # [L, 192] f32
    w_gu: jax.Array           # [L, 2*FFN, 192] paired 128-row tiles, bf16 or int8
    s_gu: jax.Array           # [L, 2*FFN] f32
    w_down: jax.Array         # [L, 192, FFN] bf16 or int8
    s_down: jax.Array         # [L, 192] f32
    q_scale: jax.Array        # [L, 6, 32] f32
    sink_k: jax.Array         # [L, 2, 2, 32] f32
    rope_freqs: jax.Array     # [L, 2, 16, 2] f32
    dw_kv: jax.Array          # [L, 2, 9, 64] f32
    dw_post: jax.Array        # [L, 2, 9, 192] f32
    final_read_q: jax.Array   # [192] f32
    final_norm_sb: jax.Array  # [2, 192] f32
    conv1pgv: jax.Array       # [192, 192] bf16: [conv1p(48); conv1g(48); conv1v(48); zeros(48)]
    head_norm_sb: jax.Array   # [6, 48] f32
    linear_g_w: jax.Array     # [48, 96] f32
    pol_w: jax.Array          # [48] f32
    pol_b: jax.Array          # [1] f32
    q_w: jax.Array            # [K, 48] f32
    q_b: jax.Array            # [K] f32
    pass_pol_w: jax.Array     # [96] f32
    pass_pol_b: jax.Array     # [1] f32
    pass_q_w: jax.Array       # [K, 96] f32
    pass_q_b: jax.Array       # [K] f32
    val_l1_w: jax.Array       # [128, 96] f32
    val_l1_b: jax.Array       # [128] f32
    val_out_w: jax.Array      # [K, 128] f32
    val_out_b: jax.Array      # [K] f32
    # Metadata encoded in 0D/1D static-shape int32 array? Better: store layers_per_block as 1-elem i32?
    # No, pass layers_per_block explicitly or store in meta_i32 (wait: FFI takes int64 attr, so keep a helper).


def supports_inkling(model) -> bool:
    from ..network import AZInklingBinnedNet

    if not isinstance(model, AZInklingBinnedNet):
        return False
    if model.passes != 1 or model.num_channels != INKLING_CH or model.dtype != jnp.bfloat16:
        return False
    if model.bins is None or model.bins.count > 16 or not model.canonicalize_hex_color:
        return False
    if not (3 <= model.height <= 11 and model.height == model.width):
        return False
    if len(model.blocks) == 0:
        return False
    blk0 = model.blocks[0]
    ffn = int(blk0.w_down.kernel[...].shape[0])
    return (
        blk0.num_heads == INKLING_HEADS
        and blk0.kv_heads == INKLING_KV_HEADS
        and blk0.head_dim == INKLING_HEAD_DIM
        and (ffn % 64) == 0
        and ffn <= 768
    )


def _quant_row_s8(w: jax.Array, use_int8: bool) -> tuple[jax.Array, jax.Array]:
    """Quantize [..., N, K] float weights per output row to symmetric int8 + f32 scale."""
    w32 = jnp.asarray(w, dtype=jnp.float32)
    if not use_int8:
        return w32.astype(jnp.bfloat16), jnp.ones(w32.shape[:-1], dtype=jnp.float32)
    amax = jnp.max(jnp.abs(w32), axis=-1)
    scale = jnp.maximum(amax, 1e-6) / 127.0
    q = jnp.clip(jnp.round(w32 / scale[..., None]), -127.0, 127.0).astype(jnp.int8)
    return q, scale.astype(jnp.float32)


def _linear_nk(linear) -> jax.Array:
    # nnx.Linear kernel is [K, N] -> transpose to [N, K]
    return jnp.asarray(linear.kernel[...], dtype=jnp.float32).T


def _conv1x1_nk(conv) -> jax.Array:
    # nnx.Conv 1x1 kernel is [1, 1, K, N] -> [N, K]
    k = jnp.asarray(conv.kernel[...], dtype=jnp.float32)
    return k.reshape((k.shape[-2], k.shape[-1])).T


def _clip_heads_np(x: jax.Array, thr: float) -> jax.Array:
    xf = jnp.asarray(x, dtype=jnp.float32)
    norm = jnp.sqrt(jnp.maximum(jnp.sum(xf * xf, axis=-1, keepdims=True), 1e-24))
    return xf * jnp.minimum(1.0, thr / (norm + 1e-6))


def pack_inkling(model, kernel: str = "int8") -> tuple[PackedInkling, int]:
    """Pack an unlooped AZInklingBinnedNet into (PackedInkling, layers_per_block)."""
    if kernel not in ("cuda", "int8"):
        raise ValueError(f"inkling native kernel must be 'cuda' or 'int8', got {kernel!r}")
    if not supports_inkling(model):
        raise ValueError("model is not supported by the native Inkling CUDA kernel")
    use_int8 = kernel == "int8"

    # Stem: [3, 3, 3, 192] -> [192, 27] bf16
    sk = jnp.asarray(model.conv.kernel[...], dtype=jnp.bfloat16)
    stem_w = jnp.transpose(sk, (3, 0, 1, 2)).reshape((INKLING_CH, 27))
    stem_b = jnp.asarray(model.conv.bias[...], dtype=jnp.float32)
    emb_s = jnp.asarray(model.embedding_norm.scale[...], dtype=jnp.float32)
    emb_b = jnp.asarray(model.embedding_norm.bias[...], dtype=jnp.float32)

    attn_read_qs, mlp_read_qs = [], []
    attn_norm_sbs, mlp_norm_sbs = [], []
    w_qkvgs, w_outs, w_gus, w_downs = [], [], [], []
    q_scales, sink_ks, rope_freqss, dw_kvs, dw_posts = [], [], [], [], []

    for blk in model.blocks:
        # attn_read_q padded to [3, 192]
        if blk.attn_read_q is None:
            rq = jnp.zeros((3, INKLING_CH), dtype=jnp.float32)
        else:
            raw_rq = jnp.asarray(blk.attn_read_q[...], dtype=jnp.float32)
            if raw_rq.shape[0] == 1:
                rq = jnp.concatenate([raw_rq, jnp.zeros((2, INKLING_CH), dtype=jnp.float32)], axis=0)
            else:
                rq = raw_rq
        attn_read_qs.append(rq)
        mlp_read_qs.append(jnp.asarray(blk.mlp_read_q[...], dtype=jnp.float32)[0])

        # attn_norm_sb padded to [6, 192]
        if blk.full:
            ansb = jnp.stack([
                blk.q_norm.scale[...], blk.q_norm.bias[...],
                blk.k_norm.scale[...], blk.k_norm.bias[...],
                blk.v_norm.scale[...], blk.v_norm.bias[...],
            ], axis=0).astype(jnp.float32)
        else:
            ansb = jnp.stack([
                blk.attn_norm.scale[...], blk.attn_norm.bias[...],
                jnp.zeros((INKLING_CH,), jnp.float32), jnp.zeros((INKLING_CH,), jnp.float32),
                jnp.zeros((INKLING_CH,), jnp.float32), jnp.zeros((INKLING_CH,), jnp.float32),
            ], axis=0).astype(jnp.float32)
        attn_norm_sbs.append(ansb)
        mlp_norm_sbs.append(jnp.stack([blk.mlp_norm.scale[...], blk.mlp_norm.bias[...]], axis=0).astype(jnp.float32))

        # [Q(192); OutGate(192); K(64); V(64)] -> [512, 192]
        w_qkvg = jnp.concatenate([
            _linear_nk(blk.q_proj),
            _linear_nk(blk.out_gate),
            _linear_nk(blk.k_proj),
            _linear_nk(blk.v_proj),
        ], axis=0)
        w_qkvgs.append(w_qkvg)
        w_outs.append(_linear_nk(blk.out_proj))

        # Paired 128-row tiles for fused Capped-SwiGLU: [FFN/64, 2, 64, 192] -> [2*FFN, 192]
        wg = _linear_nk(blk.w_gate)  # [FFN, 192]
        wu = _linear_nk(blk.w_up)    # [FFN, 192]
        ffn = wg.shape[0]
        w_gu_paired = jnp.concatenate([
            wg.reshape((ffn // 64, 64, INKLING_CH)),
            wu.reshape((ffn // 64, 64, INKLING_CH)),
        ], axis=1).reshape((2 * ffn, INKLING_CH))
        w_gus.append(w_gu_paired)
        w_downs.append(_linear_nk(blk.w_down))

        q_scales.append(jnp.asarray(blk.q_scale[...], dtype=jnp.float32))
        sink_ks.append(_clip_heads_np(blk.sink_k[...][0], INKLING_QK_CLIP))  # [2, 2, 32]
        rope_freqss.append(jnp.asarray(blk.rope_freqs[...], dtype=jnp.float32))
        # dw_weight: [3, 3, 1, C] -> [9, C]
        dw_k = jnp.asarray(blk.k_conv.dw_weight[...], dtype=jnp.float32).reshape((9, 64))
        dw_v = jnp.asarray(blk.v_conv.dw_weight[...], dtype=jnp.float32).reshape((9, 64))
        dw_kvs.append(jnp.stack([dw_k, dw_v], axis=0))
        dw_a = jnp.asarray(blk.after_attn.dw_weight[...], dtype=jnp.float32).reshape((9, INKLING_CH))
        dw_m = jnp.asarray(blk.after_mlp.dw_weight[...], dtype=jnp.float32).reshape((9, INKLING_CH))
        dw_posts.append(jnp.stack([dw_a, dw_m], axis=0))

    w_qkvg_q, s_qkvg = _quant_row_s8(jnp.stack(w_qkvgs, axis=0), use_int8)
    w_out_q, s_out = _quant_row_s8(jnp.stack(w_outs, axis=0), use_int8)
    w_gu_q, s_gu = _quant_row_s8(jnp.stack(w_gus, axis=0), use_int8)
    w_down_q, s_down = _quant_row_s8(jnp.stack(w_downs, axis=0), use_int8)

    pq = model.policy_q_head
    vh = model.value_head
    K = int(model.bins.count)
    head_norm_sb = jnp.stack([
        pq.norm_g.scale[...], pq.norm_g.bias[...],
        pq.norm_p.scale[...], pq.norm_p.bias[...],
        vh.norm1.scale[...], vh.norm1.bias[...],
    ], axis=0).astype(jnp.float32)

    if pq.has_pass:
        pass_pol_w = _linear_nk(pq.pass_policy)[0]
        pass_pol_b = jnp.asarray(pq.pass_policy.bias[...], dtype=jnp.float32)
        pass_q_w = _linear_nk(pq.pass_q)
        pass_q_b = jnp.asarray(pq.pass_q.bias[...], dtype=jnp.float32)
    else:
        pass_pol_w = jnp.zeros((96,), dtype=jnp.float32)
        pass_pol_b = jnp.zeros((1,), dtype=jnp.float32)
        pass_q_w = jnp.zeros((K, 96), dtype=jnp.float32)
        pass_q_b = jnp.zeros((K,), dtype=jnp.float32)

    packed = PackedInkling(
        stem_w=stem_w,
        stem_b=stem_b,
        emb_s=emb_s,
        emb_b=emb_b,
        attn_read_q=jnp.stack(attn_read_qs, axis=0),
        mlp_read_q=jnp.stack(mlp_read_qs, axis=0),
        attn_norm_sb=jnp.stack(attn_norm_sbs, axis=0),
        mlp_norm_sb=jnp.stack(mlp_norm_sbs, axis=0),
        w_qkvg=w_qkvg_q,
        s_qkvg=s_qkvg,
        w_out=w_out_q,
        s_out=s_out,
        w_gu=w_gu_q,
        s_gu=s_gu,
        w_down=w_down_q,
        s_down=s_down,
        q_scale=jnp.stack(q_scales, axis=0),
        sink_k=jnp.stack(sink_ks, axis=0),
        rope_freqs=jnp.stack(rope_freqss, axis=0),
        dw_kv=jnp.stack(dw_kvs, axis=0),
        dw_post=jnp.stack(dw_posts, axis=0),
        final_read_q=jnp.asarray(model.final_read_q[...], dtype=jnp.float32)[0],
        final_norm_sb=jnp.stack([model.final_norm.scale[...], model.final_norm.bias[...]], axis=0).astype(jnp.float32),
        conv1pgv=jnp.concatenate([
            _conv1x1_nk(pq.conv1p),
            _conv1x1_nk(pq.conv1g),
            _conv1x1_nk(vh.conv1),
            jnp.zeros((48, INKLING_CH), dtype=jnp.float32),
        ], axis=0).astype(jnp.bfloat16),
        head_norm_sb=head_norm_sb,
        linear_g_w=_linear_nk(pq.linear_g),
        pol_w=_conv1x1_nk(pq.policy_conv)[0],
        pol_b=jnp.asarray(pq.policy_conv.bias[...], dtype=jnp.float32),
        q_w=_conv1x1_nk(pq.q_conv),
        q_b=jnp.asarray(pq.q_conv.bias[...], dtype=jnp.float32),
        pass_pol_w=pass_pol_w,
        pass_pol_b=pass_pol_b,
        pass_q_w=pass_q_w,
        pass_q_b=pass_q_b,
        val_l1_w=_linear_nk(vh.linear1),
        val_l1_b=jnp.asarray(vh.linear1.bias[...], dtype=jnp.float32),
        val_out_w=_linear_nk(vh.out),
        val_out_b=jnp.asarray(vh.out.bias[...], dtype=jnp.float32),
    )
    return packed, int(model.layers_per_block)


def _scratch_specs(packed: PackedInkling, batch: int, cells: int, layers_per_block: int) -> list[jax.ShapeDtypeStruct]:
    L = int(packed.w_out.shape[0])
    ffn = int(packed.w_down.shape[2])
    max_bank = max(1, (L + layers_per_block - 1) // layers_per_block)
    max_M = max(1, batch * cells)
    return [
        jax.ShapeDtypeStruct((max_M, INKLING_CH), jnp.bfloat16),             # sc_partial
        jax.ShapeDtypeStruct((max_bank, max_M, INKLING_CH), jnp.bfloat16),   # sc_bank
        jax.ShapeDtypeStruct((max_M, max(512, ffn)), jnp.bfloat16),          # sc_qkvg / FFN
        jax.ShapeDtypeStruct((4, max_M, INKLING_CH), jnp.bfloat16),          # sc_act (3 streams + disjoint w_down)
        jax.ShapeDtypeStruct((3, max_M), jnp.float32),                       # sc_scales
    ]


def apply_inkling(
    packed: PackedInkling,
    observation: jax.Array,
    num_actions: int,
    atoms: int,
    *,
    layers_per_block: int = 2,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Full Inkling forward pass returning (logits [B, A], vprobs [B, K], qprobs [B, A, K])."""
    from ._library import INKLING_NETWORK_TARGET, register_inkling_network

    register_inkling_network()
    batch, side = int(observation.shape[0]), int(observation.shape[1])
    cells = side * side
    use_int8 = 1 if packed.w_out.dtype == jnp.int8 else 0
    results = [
        jax.ShapeDtypeStruct((batch, num_actions), jnp.float32),
        jax.ShapeDtypeStruct((batch, atoms), jnp.float32),
        jax.ShapeDtypeStruct((batch, num_actions, atoms), jnp.float32),
        *_scratch_specs(packed, batch, cells, layers_per_block),
    ]
    out = jax.ffi.ffi_call(INKLING_NETWORK_TARGET, results)(
        observation.astype(jnp.bool_),
        *packed,
        use_int8=np.int64(use_int8),
        layers_per_block=np.int64(layers_per_block),
    )
    return out[0], out[1], out[2]


def apply_inkling_compact(
    packed: PackedInkling,
    obs_compact: jax.Array,
    d_boards: jax.Array,
    miss_keys: jax.Array,
    cache: PositionCache,
    num_actions: int,
    atoms: int,
    *,
    layers_per_block: int = 2,
    use_cache: bool = True,
) -> tuple[jax.Array, jax.Array, PositionCache]:
    """Compact dynamic-*d_boards Inkling forward pass with lock-free PositionCache writeback."""
    from ._library import INKLING_COMPACT_TARGET, register_inkling_compact

    register_inkling_compact()
    batch, side = int(obs_compact.shape[0]), int(obs_compact.shape[1])
    cells = side * side
    use_int8 = 1 if packed.w_out.dtype == jnp.int8 else 0
    results = [
        jax.ShapeDtypeStruct((batch, atoms), jnp.float32),
        jax.ShapeDtypeStruct((batch, num_actions, atoms), jnp.float32),
        jax.ShapeDtypeStruct(cache.keys.shape, cache.keys.dtype),
        jax.ShapeDtypeStruct(cache.locks.shape, cache.locks.dtype),
        jax.ShapeDtypeStruct(cache.v.shape, cache.v.dtype),
        jax.ShapeDtypeStruct(cache.q.shape, cache.q.dtype),
        *_scratch_specs(packed, batch, cells, layers_per_block),
    ]
    # Args: 0=obs, 1=d_boards, 2=miss_keys, 3..40=*packed (38 buffers), 41..44=cache, 45=wave -> alias 41..44 to 2..5!
    aliases = {41: 2, 42: 3, 43: 4, 44: 5}
    call = jax.ffi.ffi_call(INKLING_COMPACT_TARGET, results, input_output_aliases=aliases)
    out = call(
        obs_compact.astype(jnp.bool_),
        d_boards.astype(jnp.int32),
        miss_keys,
        *packed,
        cache.keys,
        cache.locks,
        cache.v,
        cache.q,
        cache.wave,
        use_int8=np.int64(use_int8),
        layers_per_block=np.int64(layers_per_block),
        use_cache=np.int64(1 if use_cache else 0),
    )
    return out[0], out[1], PositionCache(keys=out[2], locks=out[3], v=out[4], q=out[5], wave=cache.wave)


def make_inkling_evaluator(model, kernel: str = "int8"):
    """`play_search.make_evaluator` for `AZInklingBinnedNet` with the fused CUDA kernel."""
    from .. import binned
    from ..play_search import EvaluatorOutput

    packed, layers_per_block = pack_inkling(model, kernel)
    support = jnp.asarray(binned.support(model.bins))
    num_actions = int(model.num_actions)
    atoms = int(model.bins.count)

    def evaluator(observation):
        logits, value_probs, action_probs = apply_inkling(
            packed, observation, num_actions, atoms, layers_per_block=layers_per_block
        )
        return EvaluatorOutput(
            logits=logits,
            value=2.0 * binned.mean(value_probs, support) - 1.0,
            alpha_v=value_probs,
            alpha_q=action_probs,
            support=support,
        )

    evaluator.packed = packed  # type: ignore[attr-defined]
    evaluator.layers_per_block = layers_per_block  # type: ignore[attr-defined]
    evaluator.kernel = kernel  # type: ignore[attr-defined]
    evaluator.support = support  # type: ignore[attr-defined]
    return evaluator
