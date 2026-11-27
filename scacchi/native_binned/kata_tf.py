"""Pack and invoke the SMEM-resident INT8 + FP8-FlashAttention megakernel for flat AZKataTransformerBinnedNet."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .network import PositionCache

KATA_TF_CH = 192
KATA_TF_HEADS = 6
KATA_TF_HEAD_DIM = 32


class PackedKataTf(NamedTuple):
    # Keep ONLY JAX arrays in NamedTuple so it is a valid JAX pytree (31 buffers).
    stem_w: jax.Array          # [192, 27] bf16
    stem_b: jax.Array          # [192] f32
    attn_norm_sb: jax.Array    # [L, 2, 192] f32
    ffn_norm_sb: jax.Array     # [L, 2, 192] f32
    qk_norm_sb: jax.Array      # [L, 4, 32] f32 ([q_s, q_b, k_s, k_b])
    rope_freqs: jax.Array      # [L, 6, 16, 2] f32
    w_qkv: jax.Array           # [L, 3, 192, 192] int8
    s_qkv: jax.Array           # [L, 3, 192] f32
    w_out: jax.Array           # [L, 192, 192] int8
    s_out: jax.Array           # [L, 192] f32
    w_gu: jax.Array            # [L, 2*FFN, 192] paired 128-row [gate64 | up64] tiles, int8
    s_gu: jax.Array            # [L, 2*FFN] f32
    w_down: jax.Array          # [L, 192, FFN] int8
    s_down: jax.Array          # [L, 192] f32
    qkv_fp8_scales: jax.Array  # [L, 14] f32 ([q_sc, k_sc, v_sc[6], inv_v_sc[6]])
    final_norm_sb: jax.Array   # [2, 192] f32
    conv1pgv: jax.Array        # [192, 192] bf16: [conv1p(48); conv1g(48); conv1v(48); zeros(48)]
    head_norm_sb: jax.Array    # [6, 48] f32
    linear_g_w: jax.Array      # [96, 48] f32 (transposed for coalesced GPU loads!)
    pol_w: jax.Array           # [48] f32
    pol_b: jax.Array           # [1] f32
    q_w: jax.Array             # [K, 48] f32
    q_b: jax.Array             # [K] f32
    pass_pol_w: jax.Array      # [96] f32
    pass_pol_b: jax.Array      # [1] f32
    pass_q_w: jax.Array        # [96, K] f32 (transposed for coalesced GPU loads!)
    pass_q_b: jax.Array        # [K] f32
    val_l1_w: jax.Array        # [96, 128] f32 (transposed for coalesced GPU loads!)
    val_l1_b: jax.Array        # [128] f32
    val_out_w: jax.Array       # [128, K] f32 (transposed for coalesced GPU loads!)
    val_out_b: jax.Array       # [K] f32


def supports_kata_tf(model) -> bool:
    from ..network import AZKataTransformerBinnedNet

    if not isinstance(model, AZKataTransformerBinnedNet):
        return False
    if model.num_channels != KATA_TF_CH or model.dtype != jnp.bfloat16:
        return False
    if model.bins is None or model.bins.count > 16 or not model.canonicalize_hex_color:
        return False
    if not (3 <= model.height <= 11 and model.height == model.width):
        return False
    if len(model.blocks) == 0:
        return False
    for blk in model.blocks:
        if blk.use_bottleneck or len(blk.attns) != 1 or len(blk.ffns) != 1:
            return False
        attn = blk.attns[0]
        ffn = blk.ffns[0]
        if (
            attn.channels != KATA_TF_CH
            or attn.num_heads != KATA_TF_HEADS
            or attn.head_dim != KATA_TF_HEAD_DIM
            or not attn.qk_norm
        ):
            return False
        ffn_ch = int(ffn.w_down.kernel[...].shape[0])
        if (ffn_ch % 64) != 0 or ffn_ch > 768:
            return False
    return True


def _quant_row_s8(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    w32 = jnp.asarray(w, dtype=jnp.float32)
    amax = jnp.max(jnp.abs(w32), axis=-1)
    scale = jnp.maximum(amax, 1e-6) / 127.0
    q = jnp.clip(jnp.round(w32 / scale[..., None]), -127.0, 127.0).astype(jnp.int8)
    return q, scale.astype(jnp.float32)


def _linear_nk(linear) -> jax.Array:
    return jnp.asarray(linear.kernel[...], dtype=jnp.float32).T


def _conv1x1_nk(conv) -> jax.Array:
    k = jnp.asarray(conv.kernel[...], dtype=jnp.float32)
    return k.reshape((k.shape[-2], k.shape[-1])).T


def pack_kata_tf(model, kernel: str = "int8") -> PackedKataTf:
    """Pack a flat AZKataTransformerBinnedNet into INT8 + FP8-FlashAttention GPU buffers."""
    if kernel != "int8":
        raise ValueError(f"pack_kata_tf only supports kernel='int8', got {kernel!r}")
    if not supports_kata_tf(model):
        raise ValueError("Model is not supported by the native KataTransformer INT8 kernel")

    stem_w = jnp.asarray(model.conv.kernel[...], dtype=jnp.float32).reshape((27, KATA_TF_CH)).T.astype(jnp.bfloat16)
    stem_b = jnp.asarray(model.conv.bias[...], dtype=jnp.float32)

    attn_norm_list, ffn_norm_list, qk_norm_list, rope_list = [], [], [], []
    wqkv_list, sqkv_list, wo_list, so_list = [], [], [], []
    wgu_list, sgu_list, wd_list, sd_list = [], [], [], []
    fp8_sc_list = []

    for blk in model.blocks:
        attn = blk.attns[0]
        ffn = blk.ffns[0]

        ans = jnp.asarray(attn.norm.scale[...], dtype=jnp.float32)
        anb = jnp.asarray(attn.norm.bias[...], dtype=jnp.float32)
        attn_norm_list.append(jnp.stack([ans, anb], axis=0))

        fns = jnp.asarray(ffn.norm.scale[...], dtype=jnp.float32)
        fnb = jnp.asarray(ffn.norm.bias[...], dtype=jnp.float32)
        ffn_norm_list.append(jnp.stack([fns, fnb], axis=0))

        qns = jnp.asarray(attn.q_norm.scale[...], dtype=jnp.float32)
        qnb = jnp.asarray(attn.q_norm.bias[...], dtype=jnp.float32)
        kns = jnp.asarray(attn.k_norm.scale[...], dtype=jnp.float32)
        knb = jnp.asarray(attn.k_norm.bias[...], dtype=jnp.float32)
        qk_norm_list.append(jnp.stack([qns, qnb, kns, knb], axis=0))
        rope_list.append(jnp.asarray(attn.rope_freqs[...], dtype=jnp.float32))

        wq_f = _linear_nk(attn.q_proj)
        wk_f = _linear_nk(attn.k_proj)
        wv_f = _linear_nk(attn.v_proj)
        qkv_f = jnp.stack([wq_f, wk_f, wv_f], axis=0)  # [3, 192, 192]
        wqkv_q, sqkv_q = _quant_row_s8(qkv_f)
        wqkv_list.append(wqkv_q)
        sqkv_list.append(sqkv_q)

        wo_q, so_q = _quant_row_s8(_linear_nk(attn.out_proj))
        wo_list.append(wo_q)
        so_list.append(so_q)

        wg_f = _linear_nk(ffn.w_gate)  # [FFN, 192]
        wu_f = _linear_nk(ffn.w_up)    # [FFN, 192]
        ffn_ch = int(wg_f.shape[0])
        n_chunks = ffn_ch // 64
        gu_paired = jnp.concatenate(
            [wg_f.reshape((n_chunks, 64, KATA_TF_CH)), wu_f.reshape((n_chunks, 64, KATA_TF_CH))],
            axis=1,
        ).reshape((2 * ffn_ch, KATA_TF_CH))
        wgu_q, sgu_q = _quant_row_s8(gu_paired)
        wgu_list.append(wgu_q)
        sgu_list.append(sgu_q)

        wd_q, sd_q = _quant_row_s8(_linear_nk(ffn.w_down))
        wd_list.append(wd_q)
        sd_list.append(sd_q)

        # Analytical scale-invariant FP8 E4M3 scales with ~14-sigma headroom (< 448.0):
        # Q/K are unit-RMS * scale + bias; V is xn @ W_v^T where xn has RMS ~ rms(attn.norm.scale) + rms(bias).
        q_rms_est = jnp.sqrt(jnp.mean(qns * qns)) + jnp.max(jnp.abs(qnb))
        k_rms_est = jnp.sqrt(jnp.mean(kns * kns)) + jnp.max(jnp.abs(knb))
        q_sc = 32.0 / jnp.maximum(q_rms_est, 1.0)
        k_sc = 32.0 / jnp.maximum(k_rms_est, 1.0)

        x_rms_vec = jnp.sqrt(ans * ans + anb * anb)  # [192]
        wv_scaled = wv_f * x_rms_vec[None, :]        # [192, 192]
        wv_heads = wv_scaled.reshape((KATA_TF_HEADS, KATA_TF_HEAD_DIM, KATA_TF_CH))
        # Per-head RMS of V_h = sqrt(mean_d sum_c (W_v[h, d, c] * x_rms[c])^2)
        v_rms_h = jnp.sqrt(jnp.mean(jnp.sum(wv_heads * wv_heads, axis=-1), axis=-1))  # [6]
        v_sc_h = 32.0 / jnp.maximum(v_rms_h, 1e-4)
        inv_v_sc_h = 1.0 / v_sc_h
        fp8_sc_list.append(jnp.concatenate([jnp.stack([q_sc, k_sc]), v_sc_h, inv_v_sc_h], axis=0))

    pq = model.policy_q_head
    vh = model.value_head
    final_norm_sb = jnp.stack(
        [
            jnp.asarray(model.final_norm.scale[...], dtype=jnp.float32),
            jnp.asarray(model.final_norm.bias[...], dtype=jnp.float32),
        ],
        axis=0,
    )
    head_norm_sb = jnp.stack(
        [
            jnp.asarray(pq.norm_g.scale[...], dtype=jnp.float32),
            jnp.asarray(pq.norm_g.bias[...], dtype=jnp.float32),
            jnp.asarray(pq.norm_p.scale[...], dtype=jnp.float32),
            jnp.asarray(pq.norm_p.bias[...], dtype=jnp.float32),
            jnp.asarray(vh.norm1.scale[...], dtype=jnp.float32),
            jnp.asarray(vh.norm1.bias[...], dtype=jnp.float32),
        ],
        axis=0,
    )
    atoms = int(model.bins.count)
    if pq.has_pass and pq.pass_policy is not None and pq.pass_q is not None:
        pass_pol_w = jnp.asarray(pq.pass_policy.kernel[...], dtype=jnp.float32)[:, 0]
        pass_pol_b = jnp.asarray(pq.pass_policy.bias[...], dtype=jnp.float32)
        # Keep [96, K] directly (nnx.Linear kernel is ALREADY [96, K] -> coalesced across k!)
        pass_q_w = jnp.asarray(pq.pass_q.kernel[...], dtype=jnp.float32)
        pass_q_b = jnp.asarray(pq.pass_q.bias[...], dtype=jnp.float32)
    else:
        pass_pol_w = jnp.zeros((96,), dtype=jnp.float32)
        pass_pol_b = jnp.zeros((1,), dtype=jnp.float32)
        pass_q_w = jnp.zeros((96, atoms), dtype=jnp.float32)
        pass_q_b = jnp.zeros((atoms,), dtype=jnp.float32)

    return PackedKataTf(
        stem_w=stem_w,
        stem_b=stem_b,
        attn_norm_sb=jnp.stack(attn_norm_list, axis=0),
        ffn_norm_sb=jnp.stack(ffn_norm_list, axis=0),
        qk_norm_sb=jnp.stack(qk_norm_list, axis=0),
        rope_freqs=jnp.stack(rope_list, axis=0),
        w_qkv=jnp.stack(wqkv_list, axis=0),
        s_qkv=jnp.stack(sqkv_list, axis=0),
        w_out=jnp.stack(wo_list, axis=0),
        s_out=jnp.stack(so_list, axis=0),
        w_gu=jnp.stack(wgu_list, axis=0),
        s_gu=jnp.stack(sgu_list, axis=0),
        w_down=jnp.stack(wd_list, axis=0),
        s_down=jnp.stack(sd_list, axis=0),
        qkv_fp8_scales=jnp.stack(fp8_sc_list, axis=0),
        final_norm_sb=final_norm_sb,
        conv1pgv=jnp.concatenate(
            [
                _conv1x1_nk(pq.conv1p),
                _conv1x1_nk(pq.conv1g),
                _conv1x1_nk(vh.conv1),
                jnp.zeros((48, KATA_TF_CH), dtype=jnp.float32),
            ],
            axis=0,
        ).astype(jnp.bfloat16),
        head_norm_sb=head_norm_sb,
        # Transposed head weights ([in, out] = raw nnx.Linear kernels!) for 100% coalesced GPU reads!
        linear_g_w=jnp.asarray(pq.linear_g.kernel[...], dtype=jnp.float32),  # [96, 48]
        pol_w=_conv1x1_nk(pq.policy_conv)[0],
        pol_b=jnp.asarray(pq.policy_conv.bias[...], dtype=jnp.float32),
        q_w=_conv1x1_nk(pq.q_conv),
        q_b=jnp.asarray(pq.q_conv.bias[...], dtype=jnp.float32),
        pass_pol_w=pass_pol_w,
        pass_pol_b=pass_pol_b,
        pass_q_w=pass_q_w,                                                   # [96, K]
        pass_q_b=pass_q_b,
        val_l1_w=jnp.asarray(vh.linear1.kernel[...], dtype=jnp.float32),     # [96, 128]
        val_l1_b=jnp.asarray(vh.linear1.bias[...], dtype=jnp.float32),
        val_out_w=jnp.asarray(vh.out.kernel[...], dtype=jnp.float32),        # [128, K]
        val_out_b=jnp.asarray(vh.out.bias[...], dtype=jnp.float32),
    )


def _scratch_specs(batch: int, cells: int) -> list[jax.ShapeDtypeStruct]:
    # Hex 3..8 (cells <= 64) is 100% SMEM-resident and needs zero L2 scratch!
    if cells <= 64:
        return [
            jax.ShapeDtypeStruct((1,), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.int8),
            jax.ShapeDtypeStruct((1,), jnp.float32),
        ]
    from ._library import device_sm_count

    max_ctas = min(max(1, batch), device_sm_count() * 2)
    rows = max_ctas * cells
    return [
        jax.ShapeDtypeStruct((rows, KATA_TF_CH), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, KATA_TF_CH), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, KATA_TF_CH), jnp.int8),
        jax.ShapeDtypeStruct((rows,), jnp.float32),
    ]


def apply_kata_tf(
    packed: PackedKataTf,
    observation: jax.Array,
    num_actions: int,
    atoms: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Full KataTransformer INT8 forward pass returning (logits [B, A], vprobs [B, K], qprobs [B, A, K])."""
    from ._library import KATA_TF_NETWORK_TARGET, register_kata_tf_network

    register_kata_tf_network()
    batch, side = int(observation.shape[0]), int(observation.shape[1])
    cells = side * side
    results = [
        jax.ShapeDtypeStruct((batch, num_actions), jnp.float32),
        jax.ShapeDtypeStruct((batch, atoms), jnp.float32),
        jax.ShapeDtypeStruct((batch, num_actions, atoms), jnp.float32),
        *_scratch_specs(batch, cells),
    ]
    out = jax.ffi.ffi_call(KATA_TF_NETWORK_TARGET, results)(
        observation.astype(jnp.bool_),
        *packed,
    )
    return out[0], out[1], out[2]


def apply_kata_tf_compact(
    packed: PackedKataTf,
    obs_compact: jax.Array,
    d_boards: jax.Array,
    miss_keys: jax.Array,
    cache: PositionCache,
    num_actions: int,
    atoms: int,
    *,
    use_cache: bool = True,
) -> tuple[jax.Array, jax.Array, PositionCache]:
    """Compact dynamic-*d_boards KataTransformer INT8 forward pass with lock-free PositionCache writeback."""
    from ._library import KATA_TF_COMPACT_TARGET, register_kata_tf_compact

    register_kata_tf_compact()
    batch, side = int(obs_compact.shape[0]), int(obs_compact.shape[1])
    cells = side * side
    results = [
        jax.ShapeDtypeStruct((batch, atoms), jnp.float32),
        jax.ShapeDtypeStruct((batch, num_actions, atoms), jnp.float32),
        jax.ShapeDtypeStruct(cache.keys.shape, cache.keys.dtype),
        jax.ShapeDtypeStruct(cache.locks.shape, cache.locks.dtype),
        jax.ShapeDtypeStruct(cache.v.shape, cache.v.dtype),
        jax.ShapeDtypeStruct(cache.q.shape, cache.q.dtype),
        *_scratch_specs(batch, cells),
    ]
    # Args: 0=obs, 1=d_boards, 2=miss_keys, 3..33=*packed (31 buffers), 34..37=cache, 38=wave -> alias 34..37 to 2..5!
    aliases = {34: 2, 35: 3, 36: 4, 37: 5}
    call = jax.ffi.ffi_call(KATA_TF_COMPACT_TARGET, results, input_output_aliases=aliases)
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
        use_cache=np.int64(1 if use_cache else 0),
    )
    return out[0], out[1], PositionCache(keys=out[2], locks=out[3], v=out[4], q=out[5], wave=cache.wave)


def make_kata_tf_evaluator(model, kernel: str = "int8"):
    """`play_search.make_evaluator` for flat `AZKataTransformerBinnedNet` with the INT8 megakernel."""
    from .. import binned
    from ..play_search import EvaluatorOutput

    packed = pack_kata_tf(model, kernel)
    support = jnp.asarray(binned.support(model.bins))
    num_actions = int(model.num_actions)
    atoms = int(model.bins.count)

    def evaluator(observation):
        logits, value_probs, action_probs = apply_kata_tf(packed, observation, num_actions, atoms)
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
