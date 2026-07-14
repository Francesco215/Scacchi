"""Public policy wrappers built on the Dirichlet tree-search core."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from . import action_selection
from . import base
from . import posterior_updates
from .search import instantiate_tree_from_root, search


def posterior_best_policy_target(
    rng_key: chex.PRNGKey,
    alpha: jax.Array,
    legal_action_mask: jax.Array,
    num_samples: int,
    *,
    chunk_size: int | None = None,
) -> jax.Array:
    """Monte Carlo estimate of each action's posterior best probability."""
    return action_selection.thompson_policy(
        rng_key,
        alpha,
        ~legal_action_mask,
        num_samples,
        chunk_size=chunk_size,
    )


def dirichlet_thompson_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: jax.Array | None = None,
    posterior_update: base.PosteriorUpdateFn = posterior_updates.update_posterior,
    max_depth: int | None = None,
    num_search_blocks: int = 1,
    policy_samples: int = 32,
    policy_sample_chunk_size: int | None = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
) -> base.PolicyOutput:
    """Run Thompson tree search with an MCTX-shaped external API.

    ``num_search_blocks`` multiplies the simulation budget while retaining one
    persistent tree; message/count/cache repair is not valid across resets.
    """

    if num_search_blocks < 1:
        raise ValueError(
            f"num_search_blocks must be >= 1, got {num_search_blocks}"
        )
    if num_simulations < 0:
        raise ValueError(f"num_simulations must be >= 0, got {num_simulations}")
    if policy_samples < 0:
        raise ValueError(f"policy_samples must be >= 0, got {policy_samples}")
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits, dtype=bool)
    search_key, policy_key = jax.random.split(rng_key)

    total_simulations = num_simulations * num_search_blocks
    if total_simulations == 0:
        tree = instantiate_tree_from_root(root, 0, invalid_actions)
    else:
        tree = search(
            params=params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            action_selection_fn=action_selection.thompson_action_selection,
            posterior_update=posterior_update,
            num_simulations=total_simulations,
            max_depth=max_depth,
            invalid_actions=invalid_actions,
            loop_fn=loop_fn,
        )

    alpha = action_selection.root_action_alpha(tree)
    legal_action_mask = ~invalid_actions
    action_weights = posterior_best_policy_target(
        policy_key,
        alpha,
        legal_action_mask,
        max(1, policy_samples),
        chunk_size=policy_sample_chunk_size,
    )
    action = action_selection.masked_argmax(
        action_weights,
        invalid_actions,
    )

    return base.PolicyOutput(
        action=action,
        action_weights=action_weights,
        search_tree=tree,
    )
