"""Tests for the fast Hex binned-search, Z_2 symmetry, FP8 conv, and stratified eval paths."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi import binned
from scacchi.envs import make_env
from scacchi.loss import make_compute_input_for_lossfn, train, train_binned_fast
from scacchi.native_binned._library import find_nvcc
from scacchi.native_binned.network import empty_position_cache, pack
from scacchi.native_binned.search import empty_fast_tree_carry, fused_fast_search
from scacchi.network import AZBinnedNet, _fp8_conv3x3
from scacchi.pipeline import _apply_hex_rot180, build_optimizer
from scacchi.play import PlayerOutput, _init_dephased_env_state, play_eval
from scacchi.types import BinConfig, Config, EnvConfig, ModelConfig, Network, SelfplayConfig, TrainingConfig


def _kernel_available() -> bool:
    if jax.default_backend() != "gpu":
        return False
    try:
        find_nvcc()
    except RuntimeError:
        return False
    return True


needs_kernel = pytest.mark.skipif(not _kernel_available(), reason="needs a CUDA GPU and nvcc")


def test_hex_rot180_isomorphism_on_env_and_sample():
    env = make_env("hex", 5)
    cells = 25
    st_a = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 16))
    st_b = st_a
    for step in range(9):
        logits = jnp.where(st_a.legal_action_mask, 0.0, -1e9)
        act = jax.random.categorical(jax.random.PRNGKey(100 + step), logits).astype(jnp.int32)
        act_rot = (cells - 1) - act
        st_a = jax.vmap(env.step)(st_a, act)
        st_b = jax.vmap(env.step)(st_b, act_rot)

    assert bool(jnp.array_equal(st_a.observation[:, ::-1, ::-1, :], st_b.observation))
    assert bool(jnp.array_equal(jnp.flip(st_a.legal_action_mask[:, :cells], axis=1), st_b.legal_action_mask[:, :cells]))
    assert bool(jnp.array_equal(st_a.terminated, st_b.terminated))
    assert bool(jnp.array_equal(st_a.rewards, st_b.rewards))


def test_fp8_conv3x3_matches_bf16_forward_and_gradients():
    key = jax.random.PRNGKey(7)
    kx, kw, kdy = jax.random.split(key, 3)
    x = jax.random.normal(kx, (64, 8, 8, 128), dtype=jnp.bfloat16)
    w = (0.05 * jax.random.normal(kw, (3, 3, 128, 128))).astype(jnp.bfloat16)
    dy = jax.random.normal(kdy, (64, 8, 8, 128), dtype=jnp.bfloat16)

    def ref_fn(x_, w_):
        return jax.lax.conv_general_dilated(
            x_, w_, window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )

    y_ref, vjp_ref = jax.vjp(ref_fn, x, w)
    dx_ref, dw_ref = vjp_ref(dy)

    y_fp8, vjp_fp8 = jax.vjp(_fp8_conv3x3, x, w)
    dx_fp8, dw_fp8 = vjp_fp8(dy)

    def cos(a, b):
        af, bf = a.astype(jnp.float32).reshape(-1), b.astype(jnp.float32).reshape(-1)
        return float(jnp.dot(af, bf) / (jnp.linalg.norm(af) * jnp.linalg.norm(bf)))

    assert cos(y_ref, y_fp8) > 0.999
    assert cos(dx_ref, dx_fp8) > 0.999
    assert cos(dw_ref, dw_fp8) > 0.999


def test_stratified_hex_eval_balances_colors_and_terminates():
    env = make_env("hex", 4)

    def first_legal_player(state, key):
        del key
        action = jnp.argmax(state.legal_action_mask.astype(jnp.int32), axis=-1).astype(jnp.int32)
        return PlayerOutput(action=action, posterior=None)

    metrics = play_eval(
        env,
        first_legal_player,
        first_legal_player,
        jax.random.PRNGKey(3),
        batch_size=64,
        stratified_hex=True,
    )
    # Two identical deterministic players with exact 32 Black / 32 White split must score exactly 50%!
    assert float(metrics.win_rate) == 0.5
    assert float(metrics.lose_rate) == 0.5
    assert float(metrics.draw_rate) == 0.0


def test_dephased_env_init_spreads_step_counts():
    env = make_env("hex", 6)
    state = _init_dephased_env_state(env, jax.random.PRNGKey(11), 128)
    steps = np.asarray(state._x.step_count)
    assert steps.min() == 0
    assert steps.max() >= 8
    assert len(np.unique(steps)) >= 8


@needs_kernel
def test_fused_fast_search_subtree_reuse_and_cache():
    env = make_env("hex", 5)
    model = AZBinnedNet(
        env.num_actions,
        env.observation_shape,
        num_channels=128,
        num_blocks=6,
        dtype=jnp.bfloat16,
        bins=BinConfig(count=9),
        rngs=nnx.Rngs(2),
    )
    packed = pack(model, "int8")
    pts = jnp.asarray(binned.support(model.bins), dtype=jnp.float32)
    batch = 32
    state = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), batch))
    carry = empty_fast_tree_carry(batch, 16, env.num_actions, 9, 25)
    cache = empty_position_cache(env.num_actions, 9, enabled=True)

    summary0, pol0, act0, carry, cache = fused_fast_search(
        packed, state, jax.random.PRNGKey(1), points=pts, num_simulations=16,
        tree_carry=carry, cache=cache, subtree_reuse=True, use_cache=True,
    )
    assert bool(jnp.all(jnp.isfinite(pol0)))
    assert np.allclose(np.asarray(jnp.sum(pol0, axis=-1)), 1.0, atol=1e-5)

    next_state = jax.vmap(env.step)(state, act0)
    carry = carry._replace(prev_actions=act0, was_reset=next_state.terminated)
    summary1, pol1, act1, carry, cache = fused_fast_search(
        packed, next_state, jax.random.PRNGKey(2), points=pts, num_simulations=16,
        tree_carry=carry, cache=cache, subtree_reuse=True, use_cache=True,
    )
    assert bool(jnp.all(jnp.isfinite(pol1)))
    assert bool(jnp.all(jnp.isfinite(summary1.value_alpha)))
    assert int(jnp.sum(cache.locks == 2)) > 0
