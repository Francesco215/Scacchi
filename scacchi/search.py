"""Gumbel AlphaZero search over PGX dynamics."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import mctx
import pgx
from flax import nnx

from scacchi.config import SearchConfig


def mask_illegal_logits(logits, legal_action_mask):
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)


def run_search(
    *,
    env: pgx.Env,
    model: nnx.Module,
    rng_key,
    state: pgx.State,
    cfg: SearchConfig,
):
    """Run Gumbel AlphaZero search from a batched PGX state."""

    step = jax.vmap(env.step)

    def recurrent_fn(_, rng_key, action, state):
        del rng_key
        actor = state.current_player
        state = step(state, action)
        logits, value = model(state.observation, train=False)
        value = jnp.where(state.terminated, 0.0, value)
        return (
            mctx.RecurrentFnOutput(
                reward=state.rewards[jnp.arange(action.shape[0]), actor],
                discount=jnp.where(state.terminated, 0.0, -jnp.ones_like(value)),
                prior_logits=mask_illegal_logits(logits, state.legal_action_mask),
                value=value,
            ),
            state,
        )

    logits, value = model(state.observation, train=False)
    return mctx.gumbel_muzero_policy(
        params=(),
        rng_key=rng_key,
        root=mctx.RootFnOutput(
            prior_logits=mask_illegal_logits(logits, state.legal_action_mask),
            value=value,
            embedding=state,
        ),
        recurrent_fn=recurrent_fn,
        num_simulations=cfg.num_simulations,
        invalid_actions=~state.legal_action_mask,
        max_depth=cfg.max_depth,
        qtransform=partial(
            mctx.qtransform_completed_by_mix_value,
            value_scale=cfg.qtransform_value_scale,
            rescale_values=cfg.qtransform_rescale_values,
        ),
        max_num_considered_actions=cfg.max_num_considered_actions,
        gumbel_scale=cfg.gumbel_scale,
    )
