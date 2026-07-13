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

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if chunk_size is None:
        chunk_size = num_samples
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    chunk_size = min(chunk_size, num_samples)
    num_actions = alpha.shape[-2]

    def sample_chunk(total_hits, chunk):
        keys, valid_samples = chunk
        samples = jax.vmap(lambda key: jax.random.dirichlet(key, alpha))(keys)
        scores = action_selection.outcome_utility(samples)
        scores = jnp.where(legal_action_mask[None, ...], scores, -jnp.inf)
        best = jnp.argmax(scores, axis=-1)
        hits = jax.nn.one_hot(best, num_actions, dtype=alpha.dtype)
        weight = valid_samples.astype(alpha.dtype).reshape(
            (chunk_size,) + (1,) * (hits.ndim - 1)
        )
        return total_hits + jnp.sum(hits * weight, axis=0), None

    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    padded_count = num_chunks * chunk_size
    sample_key, padding_key = jax.random.split(rng_key)
    keys = jax.random.split(sample_key, num_samples)
    pad_count = padded_count - num_samples
    if pad_count:
        keys = jnp.concatenate(
            [keys, jax.random.split(padding_key, pad_count)],
            axis=0,
        )
    keys = keys.reshape((num_chunks, chunk_size) + keys.shape[1:])
    valid = (jnp.arange(padded_count) < num_samples).reshape(
        (num_chunks, chunk_size)
    )
    initial_hits = jnp.zeros(alpha.shape[:-1], dtype=alpha.dtype)
    total_hits, _ = jax.lax.scan(sample_chunk, initial_hits, (keys, valid))
    target = jnp.where(
        legal_action_mask,
        total_hits / jnp.asarray(num_samples, dtype=alpha.dtype),
        0.0,
    )
    total = jnp.sum(target, axis=-1, keepdims=True)
    legal_count = jnp.sum(legal_action_mask, axis=-1, keepdims=True)
    fallback = legal_action_mask.astype(alpha.dtype) / jnp.maximum(legal_count, 1)
    return jnp.where(total > 0, target / jnp.maximum(total, 1.0), fallback)


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
    """Run Thompson tree search with an MCTX-shaped external API."""

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
    search_key, policy_key, _ = jax.random.split(rng_key, 3)

    tree = instantiate_tree_from_root(
        root,
        num_simulations,
        invalid_actions,
    )
    if num_simulations > 0:
        def run_block(block_index: int, state):
            del block_index
            key, previous_tree = state
            key, block_key = jax.random.split(key)
            tree = search(
                params=params,
                rng_key=block_key,
                root=root,
                recurrent_fn=recurrent_fn,
                root_action_selection_fn=(
                    action_selection.thompson_root_action_selection
                ),
                interior_action_selection_fn=(
                    action_selection.policy_prior_interior_action_selection
                ),
                posterior_update=posterior_update,
                num_simulations=num_simulations,
                max_depth=max_depth,
                invalid_actions=invalid_actions,
                posterior=previous_tree.posterior,
                loop_fn=loop_fn,
            )
            return key, tree

        _, tree = jax.lax.fori_loop(
            0,
            num_search_blocks,
            run_block,
            (search_key, tree),
        )

    alpha = tree.posterior.alpha
    legal_action_mask = ~invalid_actions
    if num_simulations == 0:
        sampled = jax.random.dirichlet(search_key, alpha)
        action = action_selection.masked_argmax(
            action_selection.outcome_utility(sampled),
            invalid_actions,
        )
    else:
        action = action_selection.masked_argmax(
            action_selection.outcome_utility(action_selection.outcome_mean(alpha)),
            invalid_actions,
        )

    if policy_samples == 0:
        if num_simulations == 0:
            action_weights = jax.nn.one_hot(
                action,
                root.prior_logits.shape[-1],
                dtype=root.action_values.dtype,
            )
        else:
            action_weights = tree.summary().visit_probs
    else:
        action_weights = posterior_best_policy_target(
            policy_key,
            alpha,
            legal_action_mask,
            policy_samples,
            chunk_size=policy_sample_chunk_size,
        )

    return base.PolicyOutput(
        action=action,
        action_weights=action_weights,
        search_tree=tree,
    )
