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


def thompson_root_action_selection(
    rng_key: chex.PRNGKey,
    tree: Tree,
    node_index: chex.Array,
) -> jax.Array:
    del node_index
    sampled_outcome = jax.random.dirichlet(rng_key, tree.posterior.alpha)
    return masked_argmax(
        outcome_utility(sampled_outcome),
        tree.root_invalid_actions,
    )


def policy_prior_interior_action_selection(
    rng_key: chex.PRNGKey,
    tree: Tree,
    node_index: chex.Array,
    depth: chex.Array,
) -> jax.Array:
    del rng_key, depth
    logits = tree.children_prior_logits[node_index]
    valid = jnp.isfinite(logits)
    finite_logits = jnp.where(valid, logits, jnp.finfo(logits.dtype).min)
    prior_probs = jax.nn.softmax(finite_logits)
    visit_counts = tree.children_visits[node_index]
    visit_frequency = visit_counts / (1 + jnp.sum(visit_counts, keepdims=True))
    scores = jnp.where(valid, prior_probs - visit_frequency, -jnp.inf)
    return jnp.argmax(scores, axis=-1).astype(jnp.int32)

