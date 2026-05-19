from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from scacchi.dirichlet_q_search import (
    NO_PARENT,
    NodeEmbedding,
    flip_outcome,
    posterior_best_policy_target,
    posterior_targets,
    q_evidence_sum_from_tree,
    terminal_outcome_from_reward,
)
from scacchi.play import _root_action_value_priors


class _TinyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array


class _TinyEnv:
    def step(self, state: _TinyState, action: jax.Array) -> _TinyState:
        observation = jnp.stack([action.astype(jnp.float32) + 1.0, state.observation[0]])
        return _TinyState(observation, state.legal_action_mask)


def _fake_tree(embedding: NodeEmbedding, node_visits: jax.Array, num_actions: int):
    return SimpleNamespace(
        embeddings=embedding,
        node_visits=node_visits,
        children_index=jnp.zeros((*node_visits.shape, num_actions), dtype=jnp.int32),
    )


def test_q_evidence_routes_by_root_action_and_aligns_parity():
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.array([[[0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]]),
        evidence_weight=jnp.array([[0.0, 2.0, 3.0, 5.0]]),
        root_action=jnp.array([[NO_PARENT, 0, 2, 0]]),
        depth_parity=jnp.array([[0, 0, 1, 1]]),
        alpha_Q_prior=jnp.zeros((1, 4, 3, 2)),
    )
    tree = _fake_tree(embedding, jnp.array([[1, 1, 1, 1]]), num_actions=3)

    evidence_sum = q_evidence_sum_from_tree(tree)

    expected = jnp.array(
        [
            [
                2.0 * jnp.array([0.2, 0.8]) + 5.0 * jnp.array([0.9, 0.1]),
                jnp.array([0.0, 0.0]),
                3.0 * jnp.array([0.3, 0.7]),
            ]
        ]
    )
    assert evidence_sum.shape == (1, 3, 2)
    assert jnp.allclose(evidence_sum, expected)


def test_terminal_child_outcome_scatters_back_to_root_perspective():
    terminal_parent = terminal_outcome_from_reward(jnp.array([1.0]), 2)
    terminal_child = flip_outcome(terminal_parent)
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.array([[[0.5, 0.5], terminal_child[0]]]),
        evidence_weight=jnp.array([[0.0, 8.0]]),
        root_action=jnp.array([[NO_PARENT, 1]]),
        depth_parity=jnp.array([[0, 1]]),
        alpha_Q_prior=jnp.zeros((1, 2, 2, 2)),
    )
    tree = _fake_tree(embedding, jnp.array([[1, 1]]), num_actions=2)

    evidence_sum = q_evidence_sum_from_tree(tree)

    assert jnp.allclose(terminal_parent, jnp.array([[0.0, 1.0]]))
    assert jnp.allclose(terminal_child, jnp.array([[1.0, 0.0]]))
    assert jnp.allclose(evidence_sum[0, 1], jnp.array([0.0, 8.0]))


def test_posterior_best_policy_target_masks_invalid_actions():
    alpha_Q_post = jnp.array([[[1.0, 2.0], [1.0, 1000.0], [2.0, 1.0]]])
    legal_action_mask = jnp.array([[True, False, True]])

    policy_target = posterior_best_policy_target(
        jax.random.PRNGKey(0),
        alpha_Q_post,
        legal_action_mask,
        num_samples=128,
    )

    assert policy_target.shape == (1, 3)
    assert jnp.allclose(policy_target[0, 1], 0.0)
    assert jnp.allclose(policy_target.sum(axis=-1), 1.0)


def test_root_action_value_priors_use_next_state_alpha_v_in_root_perspective():
    env_state = _TinyState(jnp.array([[7.0, 0.0]]), jnp.array([[True, False, True]]))

    def predict_fn(observation):
        alpha_v = jnp.stack([observation[:, 0] + 1.0, observation[:, 0] + 10.0], axis=-1)
        return jnp.zeros((observation.shape[0], 3)), alpha_v, jnp.ones((observation.shape[0], 3, 2))

    action_value_prior = _root_action_value_priors(_TinyEnv(), predict_fn, env_state)

    assert jnp.allclose(
        action_value_prior,
        jnp.array([[[11.0, 2.0], [11.0, 2.0], [13.0, 4.0]]]),
    )


def test_posterior_targets_add_q_evidence_to_action_value_prior_and_weight_value_evidence():
    alpha_V_prior = jnp.array([[1.0, 1.0]])
    action_value_prior = jnp.array([[[1.0, 2.0], [2.0, 1.0], [1.0, 2.0]]])
    q_evidence_sum = jnp.array([[[2.0, 0.0], [0.0, 0.0], [0.5, 1.5]]])
    policy_target = jnp.array([[0.25, 0.0, 0.75]])

    beta_Q_target, beta_V_target = posterior_targets(
        alpha_V_prior,
        action_value_prior,
        q_evidence_sum,
        policy_target,
    )

    assert jnp.allclose(beta_Q_target, action_value_prior + q_evidence_sum)
    expected_v_evidence = 0.25 * jnp.array([2.0, 0.0]) + 0.75 * jnp.array([0.5, 1.5])
    assert jnp.allclose(beta_V_target, alpha_V_prior + expected_v_evidence)
