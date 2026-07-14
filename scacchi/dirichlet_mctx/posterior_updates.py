"""Replaceable bottom-up posterior repair rules."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .action_selection import align_outcome, thompson_policy
from .tree import NodePosterior, PosteriorUpdateContext


DEFAULT_KAPPA_N = 4.0
DEFAULT_POLICY_SAMPLES = 32
DEFAULT_POLICY_SAMPLE_CHUNK_SIZE = 4


def mix_value_prior(
    value_prior: jax.Array,
    effective_action_alpha: jax.Array,
    search_policy: jax.Array,
    n_down: jax.Array,
    *,
    kappa_n: float = DEFAULT_KAPPA_N,
) -> jax.Array:
    """Mix a node prior with its policy-weighted current edge posteriors."""

    if kappa_n <= 0:
        raise ValueError(f"kappa_n must be > 0, got {kappa_n}")
    weighted_alpha = jnp.sum(
        search_policy[..., None] * effective_action_alpha,
        axis=-2,
    )
    dtype = value_prior.dtype
    n_down = n_down.astype(dtype)
    kappa = jnp.asarray(kappa_n, dtype=dtype)
    gamma = n_down / (kappa + n_down)
    return (
        (1.0 - gamma[..., None]) * value_prior
        + gamma[..., None] * weighted_alpha
    )


def update_posterior(
    rng_key: jax.Array,
    context: PosteriorUpdateContext,
    *,
    kappa_n: float = DEFAULT_KAPPA_N,
    policy_samples: int = DEFAULT_POLICY_SAMPLES,
    policy_sample_chunk_size: int = DEFAULT_POLICY_SAMPLE_CHUNK_SIZE,
) -> NodePosterior:
    """Repair one path node using the Tic-Tac-Toe message-passing rule.

    This function owns the posterior mathematics. Search only gathers the
    current node and all child summaries, invokes the function bottom-up, and
    stores its complete return value. A different rule with the same two-arg
    contract can therefore replace this implementation directly.
    """

    if kappa_n <= 0:
        raise ValueError(f"kappa_n must be > 0, got {kappa_n}")
    if policy_samples < 1:
        raise ValueError(f"policy_samples must be >= 1, got {policy_samples}")
    if policy_sample_chunk_size < 1:
        raise ValueError(
            "policy_sample_chunk_size must be >= 1, got "
            f"{policy_sample_chunk_size}"
        )

    node = context.node
    children = context.children
    leaf = context.leaf
    old = node.posterior
    active = context.active
    action_alpha = old.action_alpha
    action_count = old.action_count

    # backupPath's direct final-edge message is part of the replaceable rule.
    # It is applied only at the deepest node; ancestors receive leaf.active=0.
    batch = jnp.arange(action_count.shape[0])
    leaf_action = jnp.where(leaf.active, leaf.action, 0)
    direct_alpha = align_outcome(
        leaf.value_alpha,
        leaf.to_play,
        node.to_play,
    )
    old_direct_alpha = action_alpha[batch, leaf_action]
    old_direct_count = action_count[batch, leaf_action]
    action_alpha = action_alpha.at[batch, leaf_action].set(
        jnp.where(leaf.active[..., None], direct_alpha, old_direct_alpha)
    )
    action_count = action_count.at[batch, leaf_action].set(
        old_direct_count + leaf.active.astype(action_count.dtype)
    )

    child_value = align_outcome(
        children.value_alpha,
        children.to_play,
        node.to_play[:, None],
    )
    refresh = (
        active[:, None]
        & children.visited
        & ~children.terminal
        & (children.count > 0)
    )
    action_alpha = jnp.where(refresh[..., None], child_value, action_alpha)
    action_count = jnp.where(refresh, 1 + children.count, action_count)

    # Rebuild the effective edge posterior after the direct and child repairs.
    child_prior = align_outcome(
        children.value_prior,
        children.to_play,
        node.to_play[:, None],
    )
    fallback = jnp.where(
        children.visited[..., None],
        child_prior,
        action_alpha,
    )
    effective_alpha = jnp.where(
        (action_count > 0)[..., None],
        action_alpha,
        fallback,
    )
    legal = ~node.invalid_actions
    # This is computeStateSearchPosterior from the demo: pi_search is a fresh
    # population from the node's *current* post-repair action posteriors.  It
    # is deliberately neither a visit policy nor a historical running mean.
    search_policy = thompson_policy(
        rng_key,
        effective_alpha,
        node.invalid_actions,
        policy_samples,
        chunk_size=policy_sample_chunk_size,
    )
    n_down = jnp.sum(jnp.where(legal, action_count, 0), axis=-1)
    repaired_value = mix_value_prior(
        node.value_prior,
        effective_alpha,
        search_policy,
        n_down,
        kappa_n=kappa_n,
    )
    has_message = n_down > 0
    value_alpha = jnp.where(
        (active & has_message)[..., None],
        repaired_value,
        old.value_alpha,
    )
    return NodePosterior(
        action_alpha=action_alpha,
        action_count=action_count,
        value_alpha=value_alpha,
    )
