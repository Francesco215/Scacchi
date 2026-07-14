"""Action-selection rules used by Dirichlet Thompson tree search."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .tree import Tree


def flip_outcome(outcome: jax.Array) -> jax.Array:
    return outcome[..., ::-1]


def align_outcome(
    outcome: jax.Array,
    source_player: jax.Array,
    target_player: jax.Array,
) -> jax.Array:
    return jnp.where(
        (source_player == target_player)[..., None],
        outcome,
        flip_outcome(outcome),
    )


def outcome_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def outcome_utility(outcome: jax.Array) -> jax.Array:
    return outcome[..., -1] - outcome[..., 0]


def masked_argmax(scores: jax.Array, invalid_actions: jax.Array) -> jax.Array:
    return jnp.argmax(jnp.where(invalid_actions, -jnp.inf, scores), axis=-1).astype(
        jnp.int32
    )


def thompson_action_selection(
    rng_key: chex.PRNGKey,
    tree: Tree,
    node_index: chex.Array,
) -> jax.Array:
    """Sample the selected node's action posteriors and choose their best."""

    alpha = tree.action_posteriors.alpha[node_index]
    sampled_outcome = jax.random.dirichlet(rng_key, alpha)
    return masked_argmax(
        outcome_utility(sampled_outcome),
        tree.invalid_actions[node_index],
    )
