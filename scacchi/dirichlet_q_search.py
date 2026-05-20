from __future__ import annotations

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
from mctx._src import action_selection
from mctx._src import base
from mctx._src import search as mctx_search
import pgx


NO_PARENT = -1


class NodeEmbedding(NamedTuple):
    state: pgx.State
    outcome_dist: jax.Array
    evidence_weight: jax.Array
    root_action: jax.Array
    depth_parity: jax.Array
    alpha_Q_prior: jax.Array


def flip_outcome(outcome_dist: jax.Array) -> jax.Array:
    return outcome_dist[..., ::-1]


def outcome_utility(outcome_dist: jax.Array) -> jax.Array:
    return outcome_dist[..., -1] - outcome_dist[..., 0]


def outcome_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def terminal_outcome_from_reward(reward: jax.Array, num_outcomes: int) -> jax.Array:
    rounded_reward = jnp.round(reward).astype(jnp.int32)
    if num_outcomes == 2:
        outcome_index = (rounded_reward + 1) // 2
    elif num_outcomes == 3:
        outcome_index = rounded_reward + 1
    else:
        raise ValueError(f"unsupported outcome count: {num_outcomes}")
    return jax.nn.one_hot(outcome_index, num_outcomes, dtype=reward.dtype)


def mask_invalid_scores(scores: jax.Array, legal_action_mask: jax.Array) -> jax.Array:
    return jnp.where(legal_action_mask, scores, -jnp.inf)


def q_evidence_sum_from_tree(tree: Any) -> jax.Array:
    embeddings = tree.embeddings
    outcome_dist = embeddings.outcome_dist
    evidence_weight = embeddings.evidence_weight
    root_action = embeddings.root_action
    depth_parity = embeddings.depth_parity

    aligned_outcome = jnp.where(
        depth_parity[..., None] == 1,
        flip_outcome(outcome_dist),
        outcome_dist,
    )
    valid = (root_action != NO_PARENT) & (tree.node_visits > 0)
    weight = jnp.where(valid, evidence_weight, 0.0)

    batch_size, num_nodes = tree.node_visits.shape
    num_actions = tree.children_index.shape[-1]
    num_outcomes = outcome_dist.shape[-1]
    batch_ix = jnp.broadcast_to(jnp.arange(batch_size)[:, None], (batch_size, num_nodes))
    safe_action = jnp.where(valid, root_action, 0)

    evidence_sum = jnp.zeros(
        (batch_size, num_actions, num_outcomes),
        dtype=outcome_dist.dtype,
    )
    return evidence_sum.at[batch_ix, safe_action].add(weight[..., None] * aligned_outcome)


@chex.dataclass(frozen=True)
class DirichletRootExtraData:
    action_alpha_prior: jax.Array


def _q_evidence_sum_from_unbatched_tree(tree: Any) -> jax.Array:
    embeddings = tree.embeddings
    aligned_outcome = jnp.where(
        embeddings.depth_parity[..., None] == 1,
        flip_outcome(embeddings.outcome_dist),
        embeddings.outcome_dist,
    )
    valid = (embeddings.root_action != NO_PARENT) & (tree.node_visits > 0)
    weight = jnp.where(valid, embeddings.evidence_weight, 0.0)
    safe_action = jnp.where(valid, embeddings.root_action, 0)

    evidence_sum = jnp.zeros(
        (tree.num_actions, embeddings.outcome_dist.shape[-1]),
        dtype=embeddings.outcome_dist.dtype,
    )
    return evidence_sum.at[safe_action].add(weight[..., None] * aligned_outcome)


def dirichlet_root_action_selection(
    rng_key: chex.PRNGKey,
    tree: Any,
    node_index: chex.Numeric,
) -> jax.Array:
    del node_index
    q_evidence = _q_evidence_sum_from_unbatched_tree(tree)
    alpha_post = tree.extra_data.action_alpha_prior + q_evidence
    phi = jax.random.dirichlet(rng_key, alpha_post)
    score = outcome_utility(phi)
    return action_selection.masked_argmax(score, tree.root_invalid_actions)


def policy_prior_interior_action_selection(
    rng_key: chex.PRNGKey,
    tree: Any,
    node_index: chex.Numeric,
    depth: chex.Numeric,
) -> jax.Array:
    del rng_key, depth
    visit_counts = tree.children_visits[node_index]
    prior_probs = jax.nn.softmax(tree.children_prior_logits[node_index])
    to_argmax = prior_probs - visit_counts / (1 + jnp.sum(visit_counts, keepdims=True))
    return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


def dirichlet_q_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    *,
    action_alpha_prior: jax.Array,
    num_simulations: int,
    invalid_actions: chex.Array,
    max_depth: int | None = None,
    loop_fn=jax.lax.fori_loop,
) -> base.PolicyOutput[DirichletRootExtraData]:
    root = root.replace(
        prior_logits=jnp.where(
            invalid_actions,
            jnp.finfo(root.prior_logits.dtype).min,
            root.prior_logits,
        )
    )
    search_tree = mctx_search.search(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        root_action_selection_fn=dirichlet_root_action_selection,
        interior_action_selection_fn=policy_prior_interior_action_selection,
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        extra_data=DirichletRootExtraData(action_alpha_prior=action_alpha_prior),
        loop_fn=loop_fn,
    )
    q_evidence = q_evidence_sum_from_tree(search_tree)
    alpha_post = action_alpha_prior + q_evidence
    score = outcome_utility(outcome_mean(alpha_post))
    action = action_selection.masked_argmax(score, invalid_actions)
    return base.PolicyOutput(
        action=action,
        action_weights=search_tree.summary().visit_probs,
        search_tree=search_tree,
    )


def posterior_best_policy_target(
    rng_key: chex.PRNGKey,
    alpha_Q_post: jax.Array,
    legal_action_mask: jax.Array,
    num_samples: int,
) -> jax.Array:
    phi = jax.random.dirichlet(
        rng_key,
        alpha_Q_post,
        shape=(num_samples, *alpha_Q_post.shape[:-1]),
    )
    scores = outcome_utility(phi)
    scores = mask_invalid_scores(scores, legal_action_mask[None, ...])
    best_action = jnp.argmax(scores, axis=-1)

    num_actions = alpha_Q_post.shape[-2]
    action_hits = jax.nn.one_hot(best_action, num_actions, dtype=alpha_Q_post.dtype)
    target = jnp.mean(action_hits, axis=0)
    target = jnp.where(legal_action_mask, target, 0.0)

    target_sum = jnp.sum(target, axis=-1, keepdims=True)
    legal_count = jnp.sum(legal_action_mask, axis=-1, keepdims=True)
    legal_fallback = legal_action_mask.astype(alpha_Q_post.dtype) / jnp.maximum(legal_count, 1)
    normalized = target / jnp.maximum(target_sum, 1.0)
    return jnp.where(target_sum > 0, normalized, legal_fallback)


def posterior_targets(
    alpha_V_prior: jax.Array,
    action_value_prior: jax.Array,
    q_evidence_sum: jax.Array,
    policy_target: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    beta_Q_target = action_value_prior + q_evidence_sum
    v_evidence_sum = jnp.sum(policy_target[..., None] * q_evidence_sum, axis=-2)
    beta_V_target = alpha_V_prior + v_evidence_sum
    return beta_Q_target, beta_V_target
