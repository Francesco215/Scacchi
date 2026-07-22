"""Public policy wrappers built on the Dirichlet tree-search core."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int8

from . import action_selection
from . import base
from . import posterior_updates
from .categorical import NO_OUTCOME
from .search import search
from .tree import instantiate_tree_from_root
from .tree import Tree


def posterior_best_policy_target(rng_key: base.PRNGKey, alpha: Float[Array, "*batch action outcome"], legal_action_mask: Bool[Array, "*batch action"], num_samples: int, *, chunk_size: int | None = None, categorical_outcome: Int8[Array, "*batch action"] | None = None) -> Float[Array, "*batch action"]:
    """Monte Carlo estimate of each action's posterior best probability."""
    return action_selection.thompson_policy(rng_key, alpha, ~legal_action_mask, num_samples, chunk_size=chunk_size, categorical_outcome=categorical_outcome)


def dirichlet_thompson_policy(params: base.Params, rng_key: base.PRNGKey, *, root: base.RootFnOutput, recurrent_fn: base.RecurrentFn, num_simulations: int, invalid_actions: Bool[Array, "batch action"] | None = None, posterior_update: base.PosteriorUpdateFn = posterior_updates.update_posterior, max_depth: int | None = None, policy_samples: int = 32, policy_sample_chunk_size: int | None = None, loop_fn: base.LoopFn = jax.lax.fori_loop) -> base.PolicyOutput[Tree]:
    """Run Thompson tree search with an MCTX-shaped external API."""

    if num_simulations < 0:
        raise ValueError(f"num_simulations must be >= 0, got {num_simulations}")
    if policy_samples < 0:
        raise ValueError(f"policy_samples must be >= 0, got {policy_samples}")
    if invalid_actions is None:
        invalid_actions = ~jnp.isfinite(root.prior_logits)
    search_key, policy_key = jax.random.split(rng_key)

    if num_simulations == 0:
        tree = instantiate_tree_from_root(root, 0, invalid_actions)
    else:
        tree = search(
            params=params,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            action_selection_fn=action_selection.thompson_action_selection,
            posterior_update=posterior_update,
            num_simulations=num_simulations,
            max_depth=max_depth,
            invalid_actions=invalid_actions,
            loop_fn=loop_fn,
        )

    alpha = action_selection.root_action_alpha(tree)
    root_categorical_outcome = tree.node_categorical_outcome[:, tree.ROOT_INDEX]
    root_edge_categorical_outcome = tree.edge_categorical_outcome[
        :, tree.ROOT_INDEX
    ]
    legal_action_mask = ~invalid_actions
    categorical_root_action = action_selection.categorical_action(jax.random.fold_in(policy_key, 0), root_categorical_outcome, root_edge_categorical_outcome, tree.edge_payload[:, tree.ROOT_INDEX], invalid_actions, num_outcomes=alpha.shape[-1])
    categorical_policy = jax.nn.one_hot(categorical_root_action, alpha.shape[-2], dtype=alpha.dtype)
    root_is_categorical = root_categorical_outcome != int(NO_OUTCOME)

    def unresolved_policy(_: None) -> Float[Array, "batch action"]:
        sampled_policy = posterior_best_policy_target(policy_key, alpha, legal_action_mask, max(1, policy_samples), chunk_size=policy_sample_chunk_size, categorical_outcome=root_edge_categorical_outcome)
        return jnp.where(root_is_categorical[:, None], categorical_policy, sampled_policy)

    def categorical_policy_only(_: None) -> Float[Array, "batch action"]:
        return categorical_policy

    action_weights = jax.lax.cond(jnp.all(root_is_categorical), categorical_policy_only, unresolved_policy, operand=None)
    action = action_selection.masked_argmax(action_weights, invalid_actions)

    return base.PolicyOutput(
        action=action,
        action_weights=action_weights,
        search_tree=tree,
    )
