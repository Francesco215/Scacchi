"""Public policy wrappers built on the Dirichlet tree-search core."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from . import action_selection
from . import base
from . import posterior_updates
from .categorical import NO_OUTCOME
from .search import instantiate_tree_from_root, search


def posterior_best_policy_target(
    rng_key: chex.PRNGKey,
    alpha: jax.Array,
    legal_action_mask: jax.Array,
    num_samples: int,
    *,
    chunk_size: int | None = None,
    categorical_outcome: jax.Array | None = None,
) -> jax.Array:
    """Monte Carlo estimate of each action's posterior best probability."""
    return action_selection.thompson_policy(
        rng_key,
        alpha,
        ~legal_action_mask,
        num_samples,
        chunk_size=chunk_size,
        categorical_outcome=categorical_outcome,
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
    policy_samples: int = 32,
    policy_sample_chunk_size: int | None = None,
    categorical_draw_rule: str = "policy_prior",
    loop_fn: base.LoopFn = jax.lax.fori_loop,
) -> base.PolicyOutput:
    """Run Thompson tree search with an MCTX-shaped external API."""

    if num_simulations < 0:
        raise ValueError(f"num_simulations must be >= 0, got {num_simulations}")
    if policy_samples < 0:
        raise ValueError(f"policy_samples must be >= 0, got {policy_samples}")
    categorical_draw_rule = action_selection.validate_categorical_draw_rule(
        categorical_draw_rule
    )
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits, dtype=bool)
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
            categorical_draw_rule=categorical_draw_rule,
            loop_fn=loop_fn,
        )

    alpha = action_selection.root_action_alpha(tree)
    root_categorical_outcome = tree.node_categorical_outcome[:, tree.ROOT_INDEX]
    root_edge_categorical_outcome = tree.edge_categorical_outcome[
        :, tree.ROOT_INDEX
    ]
    legal_action_mask = ~invalid_actions
    categorical_root_action = action_selection.categorical_action(
        root_categorical_outcome,
        root_edge_categorical_outcome,
        tree.edge_categorical_distance[:, tree.ROOT_INDEX],
        tree.node_prior_logits[:, tree.ROOT_INDEX],
        invalid_actions,
        num_outcomes=alpha.shape[-1],
        draw_rule=categorical_draw_rule,
    )
    categorical_policy = jax.nn.one_hot(
        categorical_root_action,
        alpha.shape[-2],
        dtype=alpha.dtype,
    )
    root_is_categorical = root_categorical_outcome != int(NO_OUTCOME)

    def unresolved_policy(_: None) -> jax.Array:
        sampled_policy = posterior_best_policy_target(
            policy_key,
            alpha,
            legal_action_mask,
            max(1, policy_samples),
            chunk_size=policy_sample_chunk_size,
            categorical_outcome=root_edge_categorical_outcome,
        )
        return jnp.where(
            root_is_categorical[:, None],
            categorical_policy,
            sampled_policy,
        )

    action_weights = jax.lax.cond(
        jnp.all(root_is_categorical),
        lambda _: categorical_policy,
        unresolved_policy,
        operand=None,
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
