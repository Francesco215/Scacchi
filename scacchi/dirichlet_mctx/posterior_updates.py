"""Conjugate posterior updates for root actions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .tree import Posterior


def update_posterior(
    posterior: Posterior,
    *,
    action: jax.Array,
    action_value_prior: jax.Array,
    has_action_value_prior: jax.Array,
    evidence: jax.Array,
    evidence_weight: jax.Array,
    active: jax.Array,
) -> Posterior:
    """Replace a first-explored Q fallback and add one evidence item.

    Inputs are batched and already aligned to the root player's perspective.
    This is the Dirichlet-search analogue of an MCTX ``qtransform``: callers
    may supply another function with this signature to change update policy
    without changing traversal or tree storage.
    """

    batch = jnp.arange(posterior.base.shape[0])
    old_base = posterior.base[batch, action]
    old_explored = posterior.explored[batch, action]
    replace_base = active & has_action_value_prior & ~old_explored
    selected_base = jnp.where(
        replace_base[..., None],
        action_value_prior,
        old_base,
    )
    base = posterior.base.at[batch, action].set(selected_base)

    weighted_evidence = jnp.where(
        active[..., None],
        evidence_weight[..., None] * evidence,
        0.0,
    )
    evidence_sum = posterior.evidence.at[batch, action].add(weighted_evidence)
    explored = posterior.explored.at[batch, action].set(old_explored | active)
    return Posterior(base=base, evidence=evidence_sum, explored=explored)
