from __future__ import annotations

from typing import Any, NamedTuple, cast

import chex
import jax
import jax.numpy as jnp
from mctx._src import action_selection
from mctx._src import base
from mctx._src import search as mctx_search


NO_PARENT = -1


class NodeEmbedding(NamedTuple):
    state: Any
    outcome_dist: jax.Array
    alpha_V_prior: jax.Array
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
    action_value_prior: jax.Array
    explored_action_mask: jax.Array

    @property
    def action_alpha_prior(self) -> jax.Array:
        return self.action_value_prior


@chex.dataclass(frozen=True)
class DirichletQSearchOutput:
    action: chex.Array
    action_weights: chex.Array
    search_tree: Any
    q_evidence_sum: jax.Array
    alpha_search: jax.Array
    explored_action_mask: jax.Array


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


def _root_child_value_priors_from_unbatched_tree(tree: Any) -> tuple[jax.Array, jax.Array]:
    root_children = tree.children_index[0]
    child_exists = root_children != NO_PARENT
    safe_child = jnp.where(child_exists, root_children, 0)
    child_visits = tree.node_visits[safe_child]
    explored = child_exists & (child_visits > 0)
    child_prior = tree.embeddings.alpha_V_prior[safe_child]
    child_parity = tree.embeddings.depth_parity[safe_child]
    aligned_child_prior = jnp.where(
        child_parity[..., None] == 1,
        flip_outcome(child_prior),
        child_prior,
    )
    return aligned_child_prior, explored


def _root_action_value_priors_from_unbatched_tree(tree: Any) -> jax.Array:
    child_prior, explored = _root_child_value_priors_from_unbatched_tree(tree)
    newly_explored = explored & ~tree.extra_data.explored_action_mask
    return jnp.where(
        newly_explored[..., None],
        child_prior,
        tree.extra_data.action_value_prior,
    )


def root_explored_actions_from_tree(tree: Any) -> jax.Array:
    root_children = tree.children_index[:, 0, :]
    child_exists = root_children != NO_PARENT
    safe_child = jnp.where(child_exists, root_children, 0)
    child_visits = jnp.take_along_axis(tree.node_visits, safe_child, axis=1)
    return child_exists & (child_visits > 0)


def root_action_value_priors_from_tree(
    tree: Any,
    action_value_prior: jax.Array,
    explored_action_mask: jax.Array | None = None,
) -> jax.Array:
    if explored_action_mask is None:
        explored_action_mask = jnp.zeros(action_value_prior.shape[:-1], dtype=bool)
    root_children = tree.children_index[:, 0, :]
    child_exists = root_children != NO_PARENT
    safe_child = jnp.where(child_exists, root_children, 0)
    child_visits = jnp.take_along_axis(tree.node_visits, safe_child, axis=1)
    explored = child_exists & (child_visits > 0)
    child_prior_index = jnp.broadcast_to(
        safe_child[..., None],
        safe_child.shape + (tree.embeddings.alpha_V_prior.shape[-1],),
    )
    child_prior = jnp.take_along_axis(
        tree.embeddings.alpha_V_prior,
        child_prior_index,
        axis=1,
    )
    child_parity = jnp.take_along_axis(
        tree.embeddings.depth_parity,
        safe_child,
        axis=1,
    )
    aligned_child_prior = jnp.where(
        child_parity[..., None] == 1,
        flip_outcome(child_prior),
        child_prior,
    )
    newly_explored = explored & ~explored_action_mask
    return jnp.where(newly_explored[..., None], aligned_child_prior, action_value_prior)


def dirichlet_root_action_selection(
    rng_key: chex.PRNGKey,
    tree: Any,
    node_index: chex.Numeric,
) -> jax.Array:
    del node_index
    q_evidence = _q_evidence_sum_from_unbatched_tree(tree)
    action_value_prior = _root_action_value_priors_from_unbatched_tree(tree)
    alpha_post = action_value_prior + q_evidence
    phi = jax.random.dirichlet(rng_key, alpha_post)
    score = outcome_utility(phi)
    return cast(jax.Array, action_selection.masked_argmax(score, tree.root_invalid_actions))


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


def _dirichlet_q_search_block(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    expand_fn: base.RecurrentFn,
    *,
    action_value_prior: jax.Array,
    explored_action_mask: jax.Array,
    num_simulations: int,
    invalid_actions: chex.Array,
    max_depth: int | None = None,
    loop_fn=jax.lax.fori_loop,
) -> DirichletQSearchOutput:
    search_tree = mctx_search.search(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=expand_fn,
        root_action_selection_fn=dirichlet_root_action_selection,
        interior_action_selection_fn=policy_prior_interior_action_selection,
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        extra_data=DirichletRootExtraData(
            action_value_prior=action_value_prior,
            explored_action_mask=explored_action_mask,
        ),
        loop_fn=loop_fn,
    )
    q_evidence = q_evidence_sum_from_tree(search_tree)
    block_action_value_prior = root_action_value_priors_from_tree(
        search_tree,
        action_value_prior,
        explored_action_mask,
    )
    alpha_post = block_action_value_prior + q_evidence
    explored_actions = explored_action_mask | root_explored_actions_from_tree(search_tree)
    score = outcome_utility(outcome_mean(alpha_post))
    action = action_selection.masked_argmax(score, invalid_actions)
    return DirichletQSearchOutput(
        action=action,
        action_weights=search_tree.summary().visit_probs,
        search_tree=search_tree,
        q_evidence_sum=q_evidence,
        alpha_search=alpha_post,
        explored_action_mask=explored_actions,
    )


def dirichlet_q_policy(
    params: base.Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    expand_fn: base.RecurrentFn,
    *,
    action_value_prior: jax.Array | None = None,
    action_alpha_prior: jax.Array | None = None,
    num_simulations: int,
    invalid_actions: chex.Array,
    num_search_blocks: int = 1,
    max_depth: int | None = None,
    loop_fn=jax.lax.fori_loop,
) -> DirichletQSearchOutput:
    if num_search_blocks < 1:
        raise ValueError(f"num_search_blocks must be >= 1, got {num_search_blocks}")
    if action_value_prior is None:
        if action_alpha_prior is None:
            raise ValueError("action_value_prior is required")
        action_value_prior = action_alpha_prior
    elif action_alpha_prior is not None:
        raise ValueError("pass only one of action_value_prior or action_alpha_prior")
    if num_simulations < 0:
        raise ValueError(f"num_simulations must be >= 0, got {num_simulations}")
    if num_simulations == 0:
        del params, root, expand_fn, max_depth, loop_fn
        q_evidence_total = jnp.zeros_like(action_value_prior)
        alpha_search = action_value_prior
        sampled_outcome = jax.random.dirichlet(rng_key, alpha_search)
        score = outcome_utility(sampled_outcome)
        action = action_selection.masked_argmax(score, invalid_actions)
        action_weights = jax.nn.one_hot(
            action,
            action_value_prior.shape[-2],
            dtype=action_value_prior.dtype,
        )
        return DirichletQSearchOutput(
            action=action,
            action_weights=action_weights,
            search_tree=None,
            q_evidence_sum=q_evidence_total,
            alpha_search=alpha_search,
            explored_action_mask=jnp.zeros(action_value_prior.shape[:-1], dtype=bool),
        )

    block_keys = jax.random.split(rng_key, num_search_blocks)
    initial_explored_action_mask = jnp.zeros(action_value_prior.shape[:-1], dtype=bool)

    def block_body(carry, block_key):
        alpha_base, explored_action_mask, q_evidence_total, _ = carry
        block_output = _dirichlet_q_search_block(
            params=params,
            rng_key=block_key,
            root=root,
            expand_fn=expand_fn,
            action_value_prior=alpha_base,
            explored_action_mask=explored_action_mask,
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            max_depth=max_depth,
            loop_fn=loop_fn,
        )
        next_carry = (
            block_output.alpha_search,
            block_output.explored_action_mask,
            q_evidence_total + block_output.q_evidence_sum,
            block_output.action_weights,
        )
        return next_carry, ()

    zero_q_evidence = jnp.zeros_like(action_value_prior)
    zero_action_weights = jnp.zeros_like(root.prior_logits)
    ( alpha_search, explored_action_mask, q_evidence_total, action_weights), _ = jax.lax.scan(
        block_body,(action_value_prior, initial_explored_action_mask, zero_q_evidence, zero_action_weights),block_keys,
    )

    score = outcome_utility(outcome_mean(alpha_search))
    action = action_selection.masked_argmax(score, invalid_actions)
    return DirichletQSearchOutput(
        action=action,
        action_weights=action_weights,
        search_tree=None,
        q_evidence_sum=q_evidence_total,
        alpha_search=alpha_search,
        explored_action_mask=explored_action_mask,
    )


def posterior_best_policy_target(
    rng_key: chex.PRNGKey,
    alpha_Q_post: jax.Array,
    legal_action_mask: jax.Array,
    num_samples: int,
    *,
    chunk_size: int | None = None,
) -> jax.Array:
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if chunk_size is None:
        chunk_size = num_samples
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    chunk_size = min(chunk_size, num_samples)

    num_actions = alpha_Q_post.shape[-2]

    def sample_chunk(
        total_hits: jax.Array,
        chunk: tuple[chex.PRNGKey, jax.Array],
    ) -> tuple[jax.Array, None]:
        keys, valid_samples = chunk
        phi = jax.vmap(lambda key: jax.random.dirichlet(key, alpha_Q_post))(keys)
        scores = outcome_utility(phi)
        scores = mask_invalid_scores(scores, legal_action_mask[None, ...])
        best_action = jnp.argmax(scores, axis=-1)
        action_hits = jax.nn.one_hot(
            best_action,
            num_actions,
            dtype=alpha_Q_post.dtype,
        )
        valid_weight = valid_samples.astype(alpha_Q_post.dtype).reshape(
            (chunk_size,) + (1,) * (action_hits.ndim - 1)
        )
        return total_hits + jnp.sum(action_hits * valid_weight, axis=0), None

    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    padded_sample_count = num_chunks * chunk_size
    sample_keys = jax.random.split(rng_key, num_samples)
    pad_count = padded_sample_count - num_samples
    if pad_count:
        sample_keys = jnp.concatenate([sample_keys, sample_keys[:pad_count]], axis=0)
    sample_keys = jnp.reshape(
        sample_keys,
        (num_chunks, chunk_size) + sample_keys.shape[1:],
    )
    valid_samples = jnp.arange(padded_sample_count) < num_samples
    valid_samples = jnp.reshape(valid_samples, (num_chunks, chunk_size))

    initial_hits = jnp.zeros(alpha_Q_post.shape[:-1], dtype=alpha_Q_post.dtype)
    total_hits, _ = jax.lax.scan(
        sample_chunk,
        initial_hits,
        (sample_keys, valid_samples),
    )
    target = total_hits / jnp.asarray(num_samples, dtype=alpha_Q_post.dtype)
    target = jnp.where(legal_action_mask, target, 0.0)

    target_sum = jnp.sum(target, axis=-1, keepdims=True)
    legal_count = jnp.sum(legal_action_mask, axis=-1, keepdims=True)
    legal_fallback = legal_action_mask.astype(alpha_Q_post.dtype) / jnp.maximum(legal_count, 1)
    normalized = target / jnp.maximum(target_sum, 1.0)
    return jnp.where(target_sum > 0, normalized, legal_fallback)


def posterior_best_action(
    policy_target: jax.Array,
    legal_action_mask: jax.Array,
) -> jax.Array:
    return jnp.argmax(
        jnp.where(legal_action_mask, policy_target, -jnp.inf),
        axis=-1,
    ).astype(jnp.int32)


def posterior_sample_action(
    rng_key: chex.PRNGKey,
    policy_target: jax.Array,
    legal_action_mask: jax.Array,
) -> jax.Array:
    logits = jnp.log(jnp.clip(policy_target, 1e-8, 1.0))
    logits = jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    return jax.random.categorical(rng_key, logits).astype(jnp.int32)


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
