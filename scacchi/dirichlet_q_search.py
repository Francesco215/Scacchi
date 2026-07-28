from __future__ import annotations

import math
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from . import dirichlet_mctx
from .dirichlet_mctx.outcomes import NO_OUTCOME, align_categorical_outcome
from .types import QActionSet, QPairReduction


class QSupervision(NamedTuple):
    selected: Bool[Array, "*batch action"]
    pair_weight: Float[Array, "*batch action"]


def _required_output(value: jax.Array | None, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"evaluator output is missing {name}")
    return value


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
        child_terminal_outcome = align_categorical_outcome(parent_terminal_outcome, parent_player, child_state.current_player, alpha_v.shape[-1])
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
    temperature: float = 1.0,
) -> jax.Array:
    """Sample from a power-temperature transform of a root policy."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "posterior sample temperature must be finite and > 0; "
            f"got {temperature}."
        )

    positive_legal = legal_action_mask & (policy_target > 0.0)
    eligible = jnp.where(
        jnp.any(positive_legal, axis=-1, keepdims=True),
        positive_legal,
        legal_action_mask,
    )
    logits = jnp.log(jnp.clip(policy_target, 1e-8, 1.0))
    logits = logits / jnp.asarray(temperature, dtype=logits.dtype)
    logits = jnp.where(
        eligible,
        logits,
        jnp.finfo(logits.dtype).min,
    )
    return jax.random.categorical(rng_key, logits).astype(jnp.int32)


def build_q_supervision(
    action_set: QActionSet | str,
    reduction: QPairReduction | str,
    q_search_count: jax.Array,
    posterior_policy_target: jax.Array,
    solved_action: jax.Array,
    legal: jax.Array,
) -> QSupervision:
    """Select Q pairs and, only for legacy runs, retain source weighting."""

    positive_evidence = jnp.sum(q_search_count, axis=-1) > 0
    positive_policy = posterior_policy_target > 0
    return _build_q_supervision(
        action_set,
        reduction,
        positive_evidence=positive_evidence,
        positive_policy=positive_policy,
        solved_action=solved_action,
        legal=legal,
        evidence_mass=jnp.sum(q_search_count, axis=-1)
        + jnp.zeros_like(posterior_policy_target),
        posterior_policy_target=posterior_policy_target,
    )


def _build_q_supervision(
    action_set: QActionSet | str,
    reduction: QPairReduction | str,
    *,
    positive_evidence: jax.Array,
    positive_policy: jax.Array,
    solved_action: jax.Array,
    legal: jax.Array,
    evidence_mass: jax.Array,
    posterior_policy_target: jax.Array,
) -> QSupervision:
    if action_set == QActionSet.positive_search_evidence_or_solved:
        source_selected = positive_evidence
        source_weight = evidence_mass
    elif action_set == QActionSet.positive_posterior_policy_or_solved:
        source_selected = positive_policy
        source_weight = posterior_policy_target
    else:
        raise ValueError(f"unknown Q action set: {action_set!r}")

    selected = legal & (source_selected | solved_action)
    if reduction == QPairReduction.mean_over_selected_state_action_pairs:
        pair_weight = selected.astype(posterior_policy_target.dtype)
    elif reduction == QPairReduction.legacy_normalized_source_weighted_mean:
        # Historical weighted runs used their source magnitude for unresolved
        # actions and exactly unit weight for a solved action whose source was
        # smaller. Keep that behavior isolated to this deprecated reduction.
        pair_weight = jnp.where(
            legal & solved_action,
            jnp.maximum(
                source_weight,
                jnp.ones((), dtype=source_weight.dtype),
            ),
            source_weight,
        )
        pair_weight = jnp.where(selected, pair_weight, 0.0)
    else:
        raise ValueError(f"unknown Q pair reduction: {reduction!r}")
    return QSupervision(selected=selected, pair_weight=pair_weight)
