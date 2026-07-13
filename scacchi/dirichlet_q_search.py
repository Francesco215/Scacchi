from __future__ import annotations

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import mctx

from . import dirichlet_mctx
from .types import SearchConstantsConfig


NO_PARENT = -1


class NodeEmbedding(NamedTuple):
    """Extra state required only by the MCTX Dirichlet-Gumbel adapter."""

    state: Any
    outcome_dist: jax.Array
    alpha_V_prior: jax.Array
    evidence_weight: jax.Array
    root_action: jax.Array
    root_player: jax.Array


def _required_output(value: jax.Array | None, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"evaluator output is missing {name}")
    return value


flip_outcome = dirichlet_mctx.flip_outcome
outcome_utility = dirichlet_mctx.outcome_utility
outcome_mean = dirichlet_mctx.outcome_mean
posterior_best_policy_target = dirichlet_mctx.posterior_best_policy_target


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


def make_dirichlet_expand_fn_from_constants(
    env: Any,
    evaluator: Any,
    constants: SearchConstantsConfig,
):
    """Build the shared environment-step plus leaf-evaluation function."""

    kappa_terminal = float(constants.kappa_terminal)
    kappa_leaf = float(constants.kappa_leaf)

    def expand_fn(_, rng_key: jax.Array, action: jax.Array, env_state: Any):
        del rng_key
        parent_player = env_state.current_player
        child_state = jax.vmap(env.step)(env_state, action)
        prediction = evaluator(child_state.observation)
        alpha_v = _required_output(prediction.alpha_v, "alpha_v")
        logits = mask_invalid_scores(
            prediction.logits,
            child_state.legal_action_mask,
        )

        reward = child_state.rewards[
            jnp.arange(child_state.rewards.shape[0]),
            parent_player,
        ]
        parent_terminal_outcome = terminal_outcome_from_reward(
            reward,
            alpha_v.shape[-1],
        )
        child_terminal_outcome = dirichlet_mctx.align_outcome(
            parent_terminal_outcome,
            parent_player,
            child_state.current_player,
        )
        outcome = jnp.where(
            child_state.terminated[..., None],
            child_terminal_outcome,
            outcome_mean(alpha_v),
        )
        evidence_weight = jnp.where(
            child_state.terminated,
            jnp.asarray(kappa_terminal, dtype=outcome.dtype),
            jnp.asarray(kappa_leaf, dtype=outcome.dtype),
        )
        step = dirichlet_mctx.RecurrentFnOutput(
            prior_logits=logits,
            value=alpha_v,
            outcome=outcome,
            evidence_weight=evidence_weight,
            terminal=child_state.terminated,
            to_play=child_state.current_player,
        )
        return step, child_state

    return expand_fn


def adapt_dirichlet_expand_fn_to_mctx(
    expand_fn: Any,
):
    """Adapt the shared Dirichlet expansion to MCTX's scalar tree contract."""

    def mctx_expand_fn(params, rng_key: jax.Array, action: jax.Array, embedding: NodeEmbedding):
        current_player = embedding.state.current_player
        dirichlet_step, env_state = expand_fn(
            params,
            rng_key,
            action,
            embedding.state,
        )
        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        root_action = jnp.where(
            embedding.root_action == NO_PARENT,
            action,
            embedding.root_action,
        )
        value = outcome_utility(dirichlet_step.outcome)
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = jnp.where(env_state.terminated, 0.0, -jnp.ones_like(value))
        next_embedding = NodeEmbedding(
            state=env_state,
            outcome_dist=dirichlet_step.outcome,
            alpha_V_prior=dirichlet_step.value,
            evidence_weight=dirichlet_step.evidence_weight,
            root_action=root_action,
            root_player=embedding.root_player,
        )
        step = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=dirichlet_step.prior_logits,
            value=value,
        )
        return step, next_embedding

    return mctx_expand_fn


def make_dirichlet_root(
    env_state: Any,
    logits: jax.Array,
    alpha_v: jax.Array,
) -> mctx.RootFnOutput:
    """Construct the MCTX root retained for Dirichlet-Gumbel search."""

    root_outcome = outcome_mean(alpha_v)
    value = outcome_utility(root_outcome)
    root_embedding = NodeEmbedding(
        state=env_state,
        outcome_dist=root_outcome,
        alpha_V_prior=alpha_v,
        evidence_weight=jnp.zeros_like(value),
        root_action=jnp.full_like(env_state.current_player, NO_PARENT),
        root_player=env_state.current_player,
    )
    return mctx.RootFnOutput(
        prior_logits=logits,
        value=value,
        embedding=root_embedding,
    )


def q_evidence_sum_from_tree(tree: Any) -> jax.Array:
    """Extract root-action WDL evidence from an MCTX search tree."""

    embeddings = tree.embeddings
    aligned_outcome = dirichlet_mctx.align_outcome(
        embeddings.outcome_dist,
        embeddings.state.current_player,
        embeddings.root_player,
    )
    valid = (embeddings.root_action != NO_PARENT) & (tree.node_visits > 0)
    weight = jnp.where(valid, embeddings.evidence_weight, 0.0)
    batch_size, num_nodes = tree.node_visits.shape
    num_actions = tree.children_index.shape[-1]
    num_outcomes = embeddings.outcome_dist.shape[-1]
    batch = jnp.broadcast_to(
        jnp.arange(batch_size)[:, None],
        (batch_size, num_nodes),
    )
    safe_action = jnp.where(valid, embeddings.root_action, 0)
    evidence = jnp.zeros(
        (batch_size, num_actions, num_outcomes),
        dtype=embeddings.outcome_dist.dtype,
    )
    return evidence.at[batch, safe_action].add(
        weight[..., None] * aligned_outcome
    )


def root_action_value_priors_from_tree(
    tree: Any,
    action_value_prior: jax.Array,
    explored_action_mask: jax.Array | None = None,
) -> jax.Array:
    """Mix Q fallbacks with explored root children's aligned V priors."""

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
    child_player = jnp.take_along_axis(
        tree.embeddings.state.current_player,
        safe_child,
        axis=1,
    )
    root_player = tree.embeddings.root_player[:, 0, None]
    aligned_child_prior = dirichlet_mctx.align_outcome(
        child_prior,
        child_player,
        root_player,
    )
    newly_explored = explored & ~explored_action_mask
    return jnp.where(
        newly_explored[..., None],
        aligned_child_prior,
        action_value_prior,
    )


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
    logits = jnp.where(
        legal_action_mask,
        logits,
        jnp.finfo(logits.dtype).min,
    )
    return jax.random.categorical(rng_key, logits).astype(jnp.int32)


def posterior_targets(
    alpha_V_prior: jax.Array,
    action_value_prior: jax.Array,
    q_evidence_sum: jax.Array,
    policy_target: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    beta_Q_target = action_value_prior + q_evidence_sum
    v_evidence_sum = jnp.sum(
        policy_target[..., None] * q_evidence_sum,
        axis=-2,
    )
    beta_V_target = alpha_V_prior + v_evidence_sum
    return beta_Q_target, beta_V_target


def q_loss_weight_from_mode(
    mode: str,
    q_evidence_sum: jax.Array,
    posterior_policy_target: jax.Array,
) -> jax.Array:
    if mode == "evidence_mass":
        return jnp.sum(q_evidence_sum, axis=-1) + jnp.zeros_like(
            posterior_policy_target
        )
    if mode == "policy":
        return posterior_policy_target
    raise ValueError(f"unknown q_loss_weight_mode: {mode!r}")
