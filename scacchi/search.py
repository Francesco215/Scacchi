"""MCTX search adapter over PGX state embeddings."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import mctx
import pgx
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from scacchi.config import SearchConfig


def mask_illegal_logits(
    logits: Float[Array, "batch action"], legal_action_mask: Bool[Array, "batch action"]
) -> Float[Array, "batch action"]:
    """Set illegal actions to the smallest finite logit."""

    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(legal_action_mask, logits, min_logit)





def make_recurrent_fn(env: pgx.Env, model: nnx.Module):
    """Create an MCTX recurrent function backed by PGX dynamics."""

    step = jax.vmap(env.step)

    def recurrent_fn(
        unused_params: Any,
        rng_key: PRNGKeyArray,
        action: Int[Array, "batch"],
        state: pgx.State,
    ) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
        del unused_params
        del rng_key
        actor = state.current_player
        next_state = step(state, action)
        logits, value = model(next_state.observation, train=False)
        logits = mask_illegal_logits(logits, next_state.legal_action_mask)
        reward = next_state.rewards[jnp.arange(action.shape[0]), actor]
        value = jnp.where(next_state.terminated, 0.0, value)
        discount = jnp.where(next_state.terminated, 0.0, -jnp.ones_like(value))
        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return output, next_state

    return recurrent_fn


def run_search(
    *,
    env: pgx.Env,
    model: nnx.Module,
    rng_key: PRNGKeyArray,
    state: pgx.State,
    cfg: SearchConfig,
) -> mctx.PolicyOutput[Any]:
    """Run Gumbel AlphaZero search from a batched PGX state."""

    logits, value = model(state.observation, train=False)
    logits = mask_illegal_logits(logits, state.legal_action_mask)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
    recurrent_fn = make_recurrent_fn(env, model)
    return mctx.gumbel_muzero_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=cfg.num_simulations,
        invalid_actions=~state.legal_action_mask,
        max_depth=cfg.max_depth,
        qtransform=partial(mctx.qtransform_completed_by_mix_value, use_mixed_value=True),
        max_num_considered_actions=cfg.max_num_considered_actions,
        gumbel_scale=cfg.gumbel_scale,
    )
