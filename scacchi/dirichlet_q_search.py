from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp

from . import dirichlet_mctx
from .dirichlet_mctx.categorical import NO_OUTCOME


def _required_output(value: jax.Array | None, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"evaluator output is missing {name}")
    return value


flip_outcome = dirichlet_mctx.flip_outcome
outcome_utility = dirichlet_mctx.outcome_utility
outcome_mean = dirichlet_mctx.outcome_mean
posterior_best_policy_target = dirichlet_mctx.posterior_best_policy_target


def terminal_outcome_from_reward(
    reward: jax.Array,
    num_outcomes: int,
) -> jax.Array:
    """Return the exact categorical outcome index for the rewarded player."""

    rounded_reward = jnp.round(reward).astype(jnp.int32)
    if num_outcomes == 2:
        outcome = (rounded_reward + 1) // 2
    elif num_outcomes == 3:
        outcome = rounded_reward + 1
    else:
        raise ValueError(f"unsupported outcome count: {num_outcomes}")
    return outcome.astype(jnp.int8)


def make_dirichlet_expand_fn(env: Any, evaluator: Any):
    """Build the native environment-step plus leaf-evaluation function."""

    def expand_fn(_, rng_key: jax.Array, action: jax.Array, env_state: Any):
        del rng_key
        parent_player = env_state.current_player
        child_state = jax.vmap(env.step)(env_state, action)
        prediction = evaluator(child_state.observation)
        alpha_v = _required_output(prediction.alpha_v, "alpha_v")
        alpha_q = _required_output(prediction.alpha_q, "alpha_q")
        reward = child_state.rewards[
            jnp.arange(child_state.rewards.shape[0]),
            parent_player,
        ]
        parent_terminal_outcome = terminal_outcome_from_reward(
            reward,
            alpha_v.shape[-1],
        )
        child_terminal_outcome = jnp.where(
            parent_player == child_state.current_player,
            parent_terminal_outcome,
            (alpha_v.shape[-1] - 1 - parent_terminal_outcome).astype(jnp.int8),
        )
        terminal_outcome = jnp.where(
            child_state.terminated,
            child_terminal_outcome,
            jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
        )
        step = dirichlet_mctx.RecurrentFnOutput(
            value=alpha_v,
            action_values=alpha_q,
            invalid_actions=~child_state.legal_action_mask,
            terminal_outcome=terminal_outcome,
            to_play=child_state.current_player,
        )
        return step, child_state

    return expand_fn


def posterior_best_action(
    policy_target: jax.Array,
    legal_action_mask: jax.Array,
) -> jax.Array:
    return jnp.argmax(
        jnp.where(legal_action_mask, policy_target, -jnp.inf),
        axis=-1,
    ).astype(jnp.int32)


def posterior_sample_action(
    rng_key: chex.PRNGKey,
    policy_target: jax.Array,
    legal_action_mask: jax.Array,
) -> jax.Array:
    logits = jnp.log(jnp.clip(policy_target, 1e-8, 1.0))
    logits = jnp.where(
        legal_action_mask,
        logits,
        jnp.finfo(logits.dtype).min,
    )
    return jax.random.categorical(rng_key, logits).astype(jnp.int32)


def q_loss_weight_from_mode(
    mode: str,
    q_search_count: jax.Array,
    posterior_policy_target: jax.Array,
) -> jax.Array:
    if mode == "evidence_mass":
        return jnp.sum(q_search_count, axis=-1) + jnp.zeros_like(
            posterior_policy_target
        )
    if mode == "policy":
        return posterior_policy_target
    raise ValueError(f"unknown q_loss_weight_mode: {mode!r}")
