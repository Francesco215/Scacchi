from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp

from . import dirichlet_mctx
from .dirichlet_mctx.outcomes import NO_OUTCOME, align_categorical_outcome


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
    """Sample a floored power-temperature transform of a root policy.

    On legal actions this samples
    ``q_temperature(a) ∝ clip(q(a), 1e-8, 1) ** (1 / temperature)``.
    The floor is the pre-existing numerical support convention.  Keeping the
    exact temperature-one branch preserves the original seeded action path.
    Illegal actions remain masked exactly as before.
    """

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "posterior sample temperature must be finite and > 0; "
            f"got {temperature}."
        )

    logits = jnp.log(jnp.clip(policy_target, 1e-8, 1.0))
    if temperature != 1.0:
        logits = logits / jnp.asarray(temperature, dtype=logits.dtype)
    logits = jnp.where(
        legal_action_mask,
        logits,
        jnp.finfo(logits.dtype).min,
    )
    return jax.random.categorical(rng_key, logits).astype(jnp.int32)


class PosteriorPluralityResult(NamedTuple):
    """Paired plurality readouts of one shared categorical vote histogram."""

    action: jax.Array
    lowest_index_action: jax.Array
    uniform_tie_action: jax.Array
    vote_counts: jax.Array
    max_count_tie_multiplicity: jax.Array
    resampling_eligible: jax.Array


# Domain-separate the tie lottery without changing the historical vote key.
# The input key must remain the categorical-vote key so that the legacy
# posterior_plurality seeded path stays exact and paired experiments can
# compare the two tie rules on literally the same histogram.
_POSTERIOR_PLURALITY_UNIFORM_TIE_FOLD_IN = 0x544945


def posterior_plurality_result(
    rng_key: chex.PRNGKey,
    policy_target: jax.Array,
    legal_action_mask: jax.Array,
    num_samples: int = 32,
    *,
    tie_break: Literal["lowest", "uniform"] = "lowest",
) -> PosteriorPluralityResult:
    """Return paired lowest-index and uniform-tie plurality readouts.

    Both readouts use categorical votes drawn directly with ``rng_key``.
    Uniform tie-breaking alone uses a domain-separated
    ``jax.random.fold_in`` key, so changing ``tie_break`` never changes the
    votes, their counts, or a decision with a unique maximum count.
    """

    if tie_break not in {"lowest", "uniform"}:
        raise ValueError(
            "posterior plurality tie_break must be 'lowest' or 'uniform'; "
            f"got {tie_break!r}."
        )
    if num_samples < 1:
        raise ValueError(
            "posterior plurality num_samples must be >= 1; "
            f"got {num_samples}."
        )

    policy = jnp.asarray(policy_target)
    legal = jnp.asarray(legal_action_mask)
    positive_legal = legal & jnp.isfinite(policy) & (policy > 0)
    probability = jnp.where(positive_legal, policy, 0.0)
    total = jnp.sum(probability, axis=-1, keepdims=True)
    legal_count = jnp.sum(legal, axis=-1, keepdims=True)
    legal_fallback = legal.astype(policy.dtype) / jnp.maximum(legal_count, 1)
    no_legal_fallback = jax.nn.one_hot(
        jnp.zeros(policy.shape[:-1], dtype=jnp.int32),
        policy.shape[-1],
        dtype=policy.dtype,
    )
    fallback = jnp.where(legal_count > 0, legal_fallback, no_legal_fallback)
    probability = jnp.where(
        total > 0,
        probability / jnp.maximum(total, jnp.finfo(policy.dtype).tiny),
        fallback,
    )

    safe_probability = jnp.where(probability > 0, probability, 1.0)
    logits = jnp.where(probability > 0, jnp.log(safe_probability), -jnp.inf)
    vote_shape = (int(num_samples), *policy.shape[:-1])
    votes = jax.random.categorical(
        rng_key,
        logits,
        axis=-1,
        shape=vote_shape,
    )
    counts = jnp.sum(
        jax.nn.one_hot(
            votes,
            policy.shape[-1],
            dtype=jnp.int32,
        ),
        axis=0,
    )
    lowest_index_action = jnp.argmax(counts, axis=-1).astype(jnp.int32)
    max_count = jnp.max(counts, axis=-1, keepdims=True)
    tied_for_max = counts == max_count
    tie_multiplicity = jnp.sum(tied_for_max, axis=-1).astype(jnp.int32)
    tie_logits = jnp.where(tied_for_max, 0.0, -jnp.inf)
    tie_key = jax.random.fold_in(
        rng_key,
        _POSTERIOR_PLURALITY_UNIFORM_TIE_FOLD_IN,
    )
    uniform_tie_action = jax.random.categorical(
        tie_key,
        tie_logits,
    ).astype(jnp.int32)

    # Native solved-root commitment is an exact one-hot draw. Preserve it
    # directly instead of introducing even a numerical chance of changing it.
    resampling_eligible = jnp.sum(probability > 0, axis=-1) > 1
    deterministic_action = jnp.argmax(probability, axis=-1).astype(jnp.int32)
    lowest_index_action = jnp.where(
        resampling_eligible,
        lowest_index_action,
        deterministic_action,
    )
    uniform_tie_action = jnp.where(
        resampling_eligible,
        uniform_tie_action,
        deterministic_action,
    )
    action = (
        uniform_tie_action
        if tie_break == "uniform"
        else lowest_index_action
    )
    return PosteriorPluralityResult(
        action=action,
        lowest_index_action=lowest_index_action,
        uniform_tie_action=uniform_tie_action,
        vote_counts=counts,
        max_count_tie_multiplicity=tie_multiplicity,
        resampling_eligible=resampling_eligible,
    )


def posterior_plurality_action(
    rng_key: chex.PRNGKey,
    policy_target: jax.Array,
    legal_action_mask: jax.Array,
    num_samples: int = 32,
    *,
    tie_break: Literal["lowest", "uniform"] = "lowest",
) -> jax.Array:
    """Commit the plurality of categorical votes drawn from a root policy.

    This is a root-only readout.  Conditional on ``policy_target`` being the
    exact posterior-best action distribution, ``num_samples=M`` has the same
    action law as taking the lowest-index argmax of an M-draw native
    winner-MC histogram.  An approximate policy such as prefix-CDF adds only
    its estimator error; the random realizations are not bitwise coupled.

    Exact one-hot policies bypass resampling.  In particular, a solved
    categorical root keeps the native action already sampled by the search,
    including its exact distance-aware tie semantics.
    """

    return posterior_plurality_result(
        rng_key,
        policy_target,
        legal_action_mask,
        num_samples=num_samples,
        tie_break=tie_break,
    ).action


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
