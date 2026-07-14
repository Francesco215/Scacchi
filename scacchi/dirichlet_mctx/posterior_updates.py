"""Default node-local posterior update for Dirichlet Thompson search."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .action_selection import align_outcome
from .tree import Posterior, PosteriorUpdateContext


def update_posterior(
    context: PosteriorUpdateContext,
) -> Posterior:
    """Apply the current incremental evidence rule to one path node.

    The richer context is intentional: callers can replace this function with
    a rule that recomputes the node from all child embeddings or child
    posteriors without changing traversal or backup.  This default preserves
    the existing behavior: replace the selected Q fallback with the evaluated
    child's V prior on first exploration, then add the backed-up leaf evidence.
    """

    posterior = context.node.posterior
    action = context.action
    active = context.active
    batch = jnp.arange(posterior.base.shape[0])
    old_base = posterior.base[batch, action]
    old_explored = posterior.explored[batch, action]
    child_value = context.children.nodes.value[batch, action]
    child_player = context.children.nodes.to_play[batch, action]
    action_value_prior = align_outcome(
        child_value,
        child_player,
        context.node.to_play,
    )
    replace_base = active & context.is_leaf_edge & ~old_explored
    selected_base = jnp.where(
        replace_base[..., None],
        action_value_prior,
        old_base,
    )
    base = posterior.base.at[batch, action].set(selected_base)

    weighted_evidence = jnp.where(
        active[..., None],
        context.evidence_weight[..., None] * context.outcome,
        0.0,
    )
    evidence_sum = posterior.evidence.at[batch, action].add(weighted_evidence)
    explored = posterior.explored.at[batch, action].set(old_explored | active)
    return Posterior(base=base, evidence=evidence_sum, explored=explored)
