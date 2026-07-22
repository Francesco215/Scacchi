"""Replaceable bottom-up posterior repair rules."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32

from . import base
from .action_selection import posterior_best_policy
from .outcomes import NO_OUTCOME, align_outcome
from .tree import PosteriorUpdate, PosteriorUpdateContext


DEFAULT_KAPPA = 4.0
DEFAULT_POLICY_SAMPLES = 32
DEFAULT_POLICY_SAMPLE_CHUNK_SIZE = 4


def _validate_kappa(kappa: float) -> None:
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError(f"kappa must be finite and > 0, got {kappa}")


def mix_value_prior(value_prior: Float[Array, "*batch outcome"], effective_action_alpha: Float[Array, "*batch action outcome"], search_policy: Float[Array, "*batch action"], n_down: Int32[Array, "*batch"], *, kappa: float = DEFAULT_KAPPA) -> Float[Array, "*batch outcome"]:
    """Mix a node prior with its policy-weighted current edge posteriors."""

    _validate_kappa(kappa)
    weighted_alpha = jnp.sum(search_policy[..., None] * effective_action_alpha, axis=-2)
    dtype = value_prior.dtype
    n_down = n_down.astype(dtype)
    # Compute the normalized weights before multiplying either posterior.
    # The ratio form for kappa >= 1 also avoids overflowing the search dtype
    # when a very large (but finite) Python float is supplied.
    if kappa >= 1.0:
        relative_count = n_down * jnp.asarray(1.0 / kappa, dtype=dtype)
        denominator = 1.0 + relative_count
        prior_weight = 1.0 / denominator
        descendant_weight = relative_count / denominator
    else:
        kappa_array = jnp.asarray(kappa, dtype=dtype)
        denominator = kappa_array + n_down
        prior_weight = kappa_array / denominator
        descendant_weight = n_down / denominator
    return (
        prior_weight[..., None] * value_prior
        + descendant_weight[..., None] * weighted_alpha
    )


def update_posterior(rng_key: base.PRNGKey, context: PosteriorUpdateContext, *, kappa: float = DEFAULT_KAPPA, policy_samples: int = DEFAULT_POLICY_SAMPLES, policy_sample_chunk_size: int = DEFAULT_POLICY_SAMPLE_CHUNK_SIZE) -> PosteriorUpdate:
    """Repair one path node using the Tic-Tac-Toe message-passing rule.

    This function owns the posterior mathematics. Search only gathers the
    current node and all child summaries, invokes the function bottom-up, and
    stores its complete return value. A different rule with the same two-arg
    contract can therefore replace this implementation directly.
    """

    _validate_kappa(kappa)
    if policy_samples < 1:
        raise ValueError(f"policy_samples must be >= 1, got {policy_samples}")
    if policy_sample_chunk_size < 1:
        raise ValueError(f"policy_sample_chunk_size must be >= 1, got {policy_sample_chunk_size}")

    node = context.node
    children = context.children
    leaf = context.leaf
    active = context.active
    edge_alpha = node.edge_alpha
    edge_payload = node.edge_payload
    edge_outcome = node.edge_categorical_outcome
    unresolved = edge_outcome == int(NO_OUTCOME)

    # The direct final-edge message is applied only at the deepest node.
    # Terminal expansion has already folded support into the parent and
    # replaced this payload with distance, so categorical edges are untouched.
    batch = jnp.arange(edge_payload.shape[0])
    leaf_action = jnp.where(leaf.active, leaf.action, 0)
    direct_alpha = align_outcome(leaf.value_alpha, leaf.to_play, node.to_play)
    old_direct_alpha = edge_alpha[batch, leaf_action]
    old_direct_count = edge_payload[batch, leaf_action]
    direct_is_dirichlet = leaf.active & unresolved[batch, leaf_action]
    edge_alpha = edge_alpha.at[batch, leaf_action].set(jnp.where(direct_is_dirichlet[..., None], direct_alpha, old_direct_alpha))
    edge_payload = edge_payload.at[batch, leaf_action].set(old_direct_count + direct_is_dirichlet.astype(edge_payload.dtype))

    child_target_player = jnp.broadcast_to(node.to_play[:, None], children.to_play.shape)
    child_value = align_outcome(children.value_alpha, children.to_play, child_target_player)
    refresh = (
        active[:, None]
        & children.visited
        & (children.categorical_outcome == int(NO_OUTCOME))
        & (children.node_payload > 0)
        & unresolved
    )
    edge_alpha = jnp.where(refresh[..., None], child_value, edge_alpha)
    edge_payload = jnp.where(refresh, 1 + children.node_payload, edge_payload)

    # Rebuild the effective edge posterior after the direct and child repairs.
    child_prior = align_outcome(children.value_prior, children.to_play, child_target_player)
    fallback = jnp.where(children.visited[..., None], child_prior, edge_alpha)
    unresolved_count = jnp.where(unresolved, edge_payload, 0)
    use_stored = ~unresolved | (unresolved_count > 0)
    effective_alpha = jnp.where(use_stored[..., None], edge_alpha, fallback)
    legal = ~node.invalid_actions
    # This is computeStateSearchPosterior from the demo: pi_search is a fresh
    # population from the node's *current* post-repair action posteriors.  It
    # is deliberately neither a visit policy nor a historical running mean.
    search_policy = posterior_best_policy(rng_key, effective_alpha, node.invalid_actions, policy_samples, chunk_size=policy_sample_chunk_size, categorical_outcome=edge_outcome)
    categorical = ~unresolved
    safe_categorical_outcome = jnp.where(categorical, edge_outcome, 0)
    categorical_mean = jax.nn.one_hot(safe_categorical_outcome, effective_alpha.shape[-1], dtype=effective_alpha.dtype)
    # A mixed unresolved node still needs a Dirichlet-shaped value cache.
    # Project exact edges to their categorical mean while preserving their
    # existing learned mass; selection and training continue to use sidecars.
    categorical_alpha = (
        jnp.sum(effective_alpha, axis=-1, keepdims=True) * categorical_mean
    )
    cache_alpha = jnp.where(categorical[..., None], categorical_alpha, effective_alpha)
    old_unresolved_count = jnp.where(unresolved, node.edge_payload, 0)
    count_delta = jnp.sum(jnp.where(legal, unresolved_count - old_unresolved_count, 0), axis=-1)
    n_down = node.node_payload + count_delta
    repaired_value = mix_value_prior(node.value_prior, cache_alpha, search_policy, n_down, kappa=kappa)
    has_message = n_down > 0
    value_alpha = jnp.where((active & has_message)[..., None], repaired_value, node.value_alpha)
    return PosteriorUpdate(
        edge_alpha=edge_alpha,
        edge_payload=edge_payload,
        value_alpha=value_alpha,
    )
