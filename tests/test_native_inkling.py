"""Tests for the fused CUDA/INT8 AZInklingBinnedNet kernel, FP8 linear, EVPI stopping, and 512K PositionCache."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi import binned
from scacchi.envs import make_env
from scacchi.native_binned._library import device_sm_count, find_nvcc
from scacchi.native_binned.inkling import (
    apply_inkling,
    apply_inkling_compact,
    make_inkling_evaluator,
    pack_inkling,
    supports_inkling,
)
from scacchi.native_binned.network import CACHE_SLOTS, empty_position_cache
from scacchi.native_binned.search import empty_fast_tree_carry, fused_fast_search
from scacchi.network import AZInklingBinnedNet, _fp8_linear
from scacchi.play_search import make_search_player
from scacchi.types import (
    ActionCommitmentConfig,
    BinConfig,
    BinnedThompsonSearchConfig,
    SearchConfig,
    SearchKind,
)


def _kernel_available() -> bool:
    if jax.default_backend() != "gpu":
        return False
    try:
        find_nvcc()
    except RuntimeError:
        return False
    return True


needs_kernel = pytest.mark.skipif(not _kernel_available(), reason="needs a CUDA GPU and nvcc")


def _make_inkling(side: int, *, num_blocks: int = 4, ffn: int = 512, seed: int = 0, fp8: bool = False):
    env = make_env("hex", side)
    model = AZInklingBinnedNet(
        env.num_actions,
        env.observation_shape,
        num_channels=192,
        num_blocks=num_blocks,
        layers_per_block=2,
        passes=1,
        ffn_channels=ffn,
        num_heads=6,
        kv_heads=2,
        bins=BinConfig(count=9),
        dtype=jnp.bfloat16,
        fp8_training=fp8,
        canonicalize_hex_color=True,
        rngs=nnx.Rngs(seed),
    )
    # Perturb zero-init heads so logits, V, and Q are non-trivial.
    k = jax.random.PRNGKey(seed + 101)
    model.policy_q_head.policy_conv.kernel[...] = (
        jax.random.normal(k, model.policy_q_head.policy_conv.kernel[...].shape) * 0.2
    )
    model.policy_q_head.q_conv.kernel[...] = (
        jax.random.normal(jax.random.fold_in(k, 1), model.policy_q_head.q_conv.kernel[...].shape) * 0.2
    )
    model.value_head.out.kernel[...] = (
        jax.random.normal(jax.random.fold_in(k, 2), model.value_head.out.kernel[...].shape) * 0.2
    )
    return env, model


def _random_obs(batch: int, side: int, seed: int = 42) -> jax.Array:
    rng = np.random.default_rng(seed)
    obs = np.zeros((batch, side, side, 4), dtype=np.uint8)
    for b in range(batch):
        m = rng.integers(0, 3, size=(side, side))
        obs[b, :, :, 0] = m == 1
        obs[b, :, :, 1] = m == 2
        obs[b, :, :, 2] = b % 2  # mix Black & White to test D2 canonicalization!
    return jnp.asarray(obs, dtype=jnp.bool_)


def test_fp8_linear_matches_bf16_forward_and_gradients():
    key = jax.random.PRNGKey(13)
    kx, kw, kdy = jax.random.split(key, 3)
    x = jax.random.normal(kx, (64, 64, 192), dtype=jnp.bfloat16)
    w = (0.05 * jax.random.normal(kw, (192, 512))).astype(jnp.bfloat16)
    dy = jax.random.normal(kdy, (64, 64, 512), dtype=jnp.bfloat16)

    y_ref, vjp_ref = jax.vjp(lambda a, b: a @ b, x, w)
    dx_ref, dw_ref = vjp_ref(dy)

    y_fp8, vjp_fp8 = jax.vjp(_fp8_linear, x, w)
    dx_fp8, dw_fp8 = vjp_fp8(dy)

    def cos(a, b):
        af, bf = a.astype(jnp.float32).reshape(-1), b.astype(jnp.float32).reshape(-1)
        return float(jnp.dot(af, bf) / (jnp.linalg.norm(af) * jnp.linalg.norm(bf)))

    assert cos(y_ref, y_fp8) > 0.999
    assert cos(dx_ref, dx_fp8) > 0.999
    assert cos(dw_ref, dw_fp8) > 0.999


def test_inkling_fp8_training_step_finite():
    env, model = _make_inkling(8, seed=7, fp8=True)
    obs = _random_obs(8, 8, seed=19)

    def loss_fn(m):
        out = m(obs, train=True)
        return jnp.mean(out.logits ** 2) + jnp.mean(out.value_probs ** 2) + jnp.mean(out.action_probs ** 2)

    val, grads = nnx.value_and_grad(loss_fn)(model)
    assert np.isfinite(float(val))
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


@needs_kernel
@pytest.mark.parametrize("side", [5, 8, 10])
@pytest.mark.parametrize("mode", ["cuda", "int8"])
def test_inkling_cuda_and_int8_match_jax(side: int, mode: str):
    env, model = _make_inkling(side, seed=side)
    assert supports_inkling(model)
    obs = _random_obs(16, side, seed=side + 7)
    ref = model(obs, train=False)

    packed, lpb = pack_inkling(model, mode)
    logits, vprobs, qprobs = apply_inkling(packed, obs, env.num_actions, 9, layers_per_block=lpb)

    atol_l = 0.05 if mode == "cuda" else 0.10
    atol_p = 0.015 if mode == "cuda" else 0.03
    assert float(jnp.max(jnp.abs(logits - ref.logits))) < atol_l
    assert float(jnp.max(jnp.abs(vprobs - ref.value_probs))) < atol_p
    assert float(jnp.max(jnp.abs(qprobs - ref.action_probs))) < atol_p
    assert np.allclose(np.asarray(jnp.sum(vprobs, axis=-1)), 1.0, atol=1e-5)
    assert np.allclose(np.asarray(jnp.sum(qprobs, axis=-1)), 1.0, atol=1e-5)


@needs_kernel
def test_inkling_compact_and_fast_search_with_evpi_and_cache():
    assert device_sm_count() >= 1
    env, model = _make_inkling(8, seed=3)
    packed, lpb = pack_inkling(model, "int8")
    obs = _random_obs(32, 8, seed=99)
    _, full_v, full_q = apply_inkling(packed, obs, env.num_actions, 9, layers_per_block=lpb)

    cache = empty_position_cache(env.num_actions, 9, enabled=True)
    assert cache.keys.shape[0] == CACHE_SLOTS == 524288
    assert cache.v.dtype == jnp.bfloat16
    assert cache.q.dtype == jnp.bfloat16

    d_active = jnp.asarray([13], dtype=jnp.int32)
    miss_keys = jnp.zeros((32, 10), dtype=jnp.uint32)
    comp_v, comp_q, cache = apply_inkling_compact(
        packed, obs, d_active, miss_keys, cache, env.num_actions, 9, layers_per_block=lpb, use_cache=True
    )
    assert np.allclose(np.asarray(comp_v[:13]), np.asarray(full_v[:13]), atol=1e-5)
    assert np.allclose(np.asarray(comp_q[:13]), np.asarray(full_q[:13]), atol=1e-5)

    # Full fused_fast_search across 2 consecutive plies with subtree reuse, 512K cache, and EVPI stopping!
    pts = jnp.asarray(binned.support(model.bins), dtype=jnp.float32)
    batch = 32
    state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), batch))
    carry = empty_fast_tree_carry(batch, 16, env.num_actions, 9, 64)
    s0, pol0, act0, carry, cache = fused_fast_search(
        packed, state, jax.random.PRNGKey(1), points=pts, num_simulations=16,
        tree_carry=carry, cache=cache, subtree_reuse=True, use_cache=True,
        early_stop_prob=0.96, early_stop_evpi=0.005, layers_per_block=lpb,
    )
    assert bool(jnp.all(jnp.isfinite(pol0)))
    assert np.allclose(np.asarray(jnp.sum(pol0, axis=-1)), 1.0, atol=1e-5)
    filled0 = int(jnp.sum(cache.locks >= 2))
    assert filled0 > 0

    next_state = jax.vmap(env.step)(state, act0)
    carry = carry._replace(prev_actions=act0, was_reset=next_state.terminated)
    s1, pol1, act1, carry, cache = fused_fast_search(
        packed, next_state, jax.random.PRNGKey(2), points=pts, num_simulations=16,
        tree_carry=carry, cache=cache, subtree_reuse=True, use_cache=True,
        early_stop_prob=0.96, early_stop_evpi=0.005, layers_per_block=lpb,
    )
    assert bool(jnp.all(jnp.isfinite(pol1)))
    assert bool(jnp.all(jnp.isfinite(s1.value_alpha)))
    assert int(jnp.sum(cache.locks >= 2)) >= filled0

    # Also verify make_search_player routes AZInklingBinnedNet to make_inkling_evaluator + fast_search.
    search_cfg = SearchConfig(
        kind=SearchKind.binned_thompson,
        binned_thompson=BinnedThompsonSearchConfig(
            num_simulations=8, search_kernel="fused", network_kernel="int8",
            fast_search=True, early_stop_evpi=0.005,
        ),
    )
    player = make_search_player(env, model, search_cfg, ActionCommitmentConfig())
    out = player(state, jax.random.PRNGKey(5))
    assert out.action.shape == (batch,)
    assert bool(jnp.all(state.legal_action_mask[jnp.arange(batch), out.action]))


def test_inkling_int8_large_batch_determinism_and_no_nan():
    """Regression test: large multi-wave INT8 forward (B=2048) has zero cross-CTA aliasing races."""
    if jax.default_backend() != "gpu":
        pytest.skip("GPU required")

    env, model = _make_inkling(8, seed=11)
    packed, lpb = pack_inkling(model, "int8")
    obs = _random_obs(2048, 8, seed=123)
    ref_l, ref_v, ref_q = apply_inkling(packed, obs, env.num_actions, 9, layers_per_block=lpb)
    assert bool(jnp.all(jnp.isfinite(ref_l)))
    assert bool(jnp.all(jnp.isfinite(ref_v)))
    assert bool(jnp.all(jnp.isfinite(ref_q)))
    for _ in range(5):
        l, v, q = apply_inkling(packed, obs, env.num_actions, 9, layers_per_block=lpb)
        assert bool(jnp.all(jnp.isfinite(l)))
        assert bool(jnp.all(jnp.isfinite(v)))
        assert bool(jnp.all(jnp.isfinite(q)))
        assert np.array_equal(np.asarray(v), np.asarray(ref_v))
        assert np.array_equal(np.asarray(q), np.asarray(ref_q))
