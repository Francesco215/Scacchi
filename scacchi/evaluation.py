"""PGX baseline evaluation."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import pgx
from flax import nnx
from jaxtyping import Array, PRNGKeyArray

from scacchi.config import EvalConfig
from scacchi.search import mask_illegal_logits

BaselineModel = Callable[[Array], tuple[Array, Array]]


def _where_done(done: Array, done_value: Array, active_value: Array) -> Array:
    mask = done.reshape((done.shape[0],) + (1,) * (active_value.ndim - 1))
    return jnp.where(mask, done_value, active_value)


def evaluate_baseline(
    *,
    env: pgx.Env,
    model: nnx.Module,
    baseline: BaselineModel,
    rng_key: PRNGKeyArray,
    cfg: EvalConfig,
) -> Array:
    """Sample games against a PGX baseline and return model-player rewards."""

    batch_size = cfg.batch_size
    rng_key, init_key, dummy_key = jax.random.split(rng_key, 3)
    state = jax.vmap(env.init)(jax.random.split(init_key, batch_size))
    dummy_state = jax.vmap(env.init)(jax.random.split(dummy_key, batch_size))
    model_player = jnp.arange(batch_size, dtype=jnp.int32) % 2
    batch_ix = jnp.arange(batch_size)

    def step_fn(carry, key):
        state, returns, done = carry
        state = jax.tree_util.tree_map(
            lambda dummy, active: _where_done(done, dummy, active), dummy_state, state
        )
        model_logits, _ = model(state.observation, train=False)
        baseline_logits, _ = baseline(state.observation)
        logits = jnp.where(
            (state.current_player == model_player)[:, None],
            mask_illegal_logits(model_logits, state.legal_action_mask),
            mask_illegal_logits(baseline_logits, state.legal_action_mask),
        )
        state = jax.vmap(env.step)(state, jax.random.categorical(key, logits, axis=-1))
        returns += jnp.where(done, 0.0, state.rewards[batch_ix, model_player])
        return (state, returns, done | state.terminated | state.truncated), None

    _, returns, _ = jax.lax.scan(
        step_fn,
        (state, jnp.zeros(batch_size, dtype=jnp.float32), jnp.zeros(batch_size, dtype=jnp.bool_)),
        jax.random.split(rng_key, cfg.max_num_steps),
    )[0]
    return returns


def baseline_log(prefix: str, returns: Array) -> dict[str, float]:
    games = returns.size
    return {
        f"{prefix}/avg_R": float(returns.mean()),
        f"{prefix}/win_rate": float(jnp.sum(returns > 0) / games),
        f"{prefix}/draw_rate": float(jnp.sum(returns == 0) / games),
        f"{prefix}/loss_rate": float(jnp.sum(returns < 0) / games),
    }
