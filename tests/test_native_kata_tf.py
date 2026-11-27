"""Tests for the SMEM-resident INT8 + FP8-FlashAttention flat AZKataTransformerBinnedNet megakernel."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi import binned
from scacchi.envs import make_env
from scacchi.native_binned._library import find_nvcc
from scacchi.native_binned.kata_tf import (
    apply_kata_tf,
    apply_kata_tf_compact,
    make_kata_tf_evaluator,
    pack_kata_tf,
    supports_kata_tf,
)
from scacchi.native_binned.network import CACHE_SLOTS, empty_position_cache
from scacchi.native_binned.search import empty_fast_tree_carry, fused_fast_search
from scacchi.network import AZKataTransformerBinnedNet
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


def _make_kata_tf(side: int, *, num_blocks: int = 4, ffn: int = 512, seed: int = 0):
    env = make_env("hex", side)
    model = AZKataTransformerBinnedNet(
        env.num_actions,
        env.observation_shape,
        num_channels=192,
        num_blocks=num_blocks,
        ffn_channels=ffn,
        num_heads=6,
        inner_blocks=1,
        qk_norm=True,
        bins=BinConfig(count=9),
        dtype=jnp.bfloat16,
        canonicalize_hex_color=True,
        rngs=nnx.Rngs(seed),
    )
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
    m = rng.integers(0, 3, size=(batch, side, side))
    obs = np.zeros((batch, side, side, 4), dtype=np.uint8)
    obs[:, :, :, 0] = m == 1
    obs[:, :, :, 1] = m == 2
    obs[:, :, :, 2] = (np.arange(batch) % 2)[:, None, None]
    return jnp.asarray(obs, dtype=jnp.bool_)


@needs_kernel
@pytest.mark.parametrize("side", [5, 8, 10])
def test_kata_tf_int8_matches_jax(side: int):
    env, model = _make_kata_tf(side, seed=side)
    assert supports_kata_tf(model)
    obs = _random_obs(32, side, seed=side + 7)
    ref = model(obs, train=False)

    packed = pack_kata_tf(model, "int8")
    logits, vprobs, qprobs = apply_kata_tf(packed, obs, env.num_actions, 9)

    assert float(jnp.max(jnp.abs(logits - ref.logits))) < 0.10
    assert float(jnp.max(jnp.abs(vprobs - ref.value_probs))) < 0.03
    assert float(jnp.max(jnp.abs(qprobs - ref.action_probs))) < 0.03
    assert np.allclose(np.asarray(jnp.sum(vprobs, axis=-1)), 1.0, atol=1e-5)
    assert np.allclose(np.asarray(jnp.sum(qprobs, axis=-1)), 1.0, atol=1e-5)


@needs_kernel
@pytest.mark.parametrize("side", [8, 10])
def test_kata_tf_compact_and_fast_search(side: int):
    env, model = _make_kata_tf(side, seed=side + 3)
    packed = pack_kata_tf(model, "int8")
    obs = _random_obs(32, side, seed=99)
    _, full_v, full_q = apply_kata_tf(packed, obs, env.num_actions, 9)

    cache = empty_position_cache(env.num_actions, 9, enabled=True)
    assert cache.keys.shape[0] == CACHE_SLOTS == 524288
    d_active = jnp.asarray([13], dtype=jnp.int32)
    miss_keys = jnp.zeros((32, 10), dtype=jnp.uint32)
    comp_v, comp_q, cache = apply_kata_tf_compact(
        packed, obs, d_active, miss_keys, cache, env.num_actions, 9, use_cache=True
    )
    assert np.allclose(np.asarray(comp_v[:13]), np.asarray(full_v[:13]), atol=1e-5)
    assert np.allclose(np.asarray(comp_q[:13]), np.asarray(full_q[:13]), atol=1e-5)

    pts = jnp.asarray(binned.support(model.bins), dtype=jnp.float32)
    batch = 32
    keys = jax.random.split(jax.random.PRNGKey(11), batch)
    state = jax.vmap(env.init)(keys)
    tc = empty_fast_tree_carry(batch, 12, env.num_actions, 9, side * side)
    summary, policy, action, tc, cache = fused_fast_search(
        packed,
        state,
        jax.random.PRNGKey(12),
        points=pts,
        num_simulations=12,
        tree_carry=tc,
        cache=cache,
        subtree_reuse=True,
        use_cache=True,
        early_stop_prob=0.96,
        early_stop_evpi=0.005,
        min_stop_nodes=6,
    )
    assert bool(jnp.all(jnp.isfinite(policy)))
    assert bool(jnp.all(jnp.isfinite(summary.value_alpha)))
    filled0 = int(jnp.sum(cache.locks >= 2))
    assert filled0 > 0

    state2 = jax.vmap(env.step)(state, action)
    tc = tc._replace(prev_actions=action, was_reset=state2.terminated)
    summary2, policy2, _, _, cache = fused_fast_search(
        packed,
        state2,
        jax.random.PRNGKey(13),
        points=pts,
        num_simulations=12,
        tree_carry=tc,
        cache=cache,
        subtree_reuse=True,
        use_cache=True,
        early_stop_prob=0.96,
        early_stop_evpi=0.005,
        min_stop_nodes=6,
    )
    assert bool(jnp.all(jnp.isfinite(policy2)))
    assert bool(jnp.all(jnp.isfinite(summary2.value_alpha)))
    assert int(jnp.sum(cache.locks >= 2)) >= filled0

    player = make_search_player(
        env,
        model,
        SearchConfig(
            kind=SearchKind.binned_thompson,
            binned_thompson=BinnedThompsonSearchConfig(
                num_simulations=8,
                network_kernel="int8",
                search_kernel="fused",
                fast_search=True,
                position_cache=True,
                early_stop_evpi=0.005,
            ),
        ),
        ActionCommitmentConfig(),
    )
    pout = player(state, jax.random.PRNGKey(14))
    assert pout.action.shape == (batch,)
    assert pout.posterior is not None and bool(jnp.all(jnp.isfinite(pout.posterior.prediction.policy)))


@needs_kernel
@pytest.mark.parametrize("side", [8, 10])
def test_kata_tf_large_batch_determinism_and_no_nan(side: int):
    env, model = _make_kata_tf(side, seed=side + 5)
    packed = pack_kata_tf(model, "int8")
    obs = _random_obs(2048, side, seed=123)
    fn = jax.jit(lambda x: apply_kata_tf(packed, x, env.num_actions, 9))
    ref_l, ref_v, ref_q = fn(obs)
    jax.block_until_ready((ref_l, ref_v, ref_q))
    assert bool(jnp.all(jnp.isfinite(ref_l)))
    assert bool(jnp.all(jnp.isfinite(ref_v)))
    assert bool(jnp.all(jnp.isfinite(ref_q)))
    for _ in range(4):
        l, v, q = fn(obs)
        jax.block_until_ready((l, v, q))
        np.testing.assert_array_equal(np.asarray(l), np.asarray(ref_l))
        np.testing.assert_array_equal(np.asarray(v), np.asarray(ref_v))
        np.testing.assert_array_equal(np.asarray(q), np.asarray(ref_q))
