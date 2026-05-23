from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from scacchi.posterior_tree import (
    NO_CHILD,
    PosteriorTree,
    flip_outcome_np,
    outcome_mean_np,
    outcome_utility_np,
    posterior_best_policy_target_np,
    run_posterior_tree_search,
)


class ToyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    terminated: jax.Array
    rewards: jax.Array


class ToyEnv:
    def step(self, state: ToyState, action: jax.Array) -> ToyState:
        del action
        next_player = 1 - state.current_player
        return ToyState(
            observation=state.observation + 1,
            legal_action_mask=jnp.array([True, False]),
            current_player=next_player,
            terminated=jnp.array(False),
            rewards=jnp.array([0.0, 0.0]),
        )


class TerminalToyEnv:
    def step(self, state: ToyState, action: jax.Array) -> ToyState:
        del action
        next_player = 1 - state.current_player
        return ToyState(
            observation=state.observation + 1,
            legal_action_mask=jnp.array([True, True]),
            current_player=next_player,
            terminated=jnp.array(True),
            rewards=jnp.array([1.0, -1.0]),
        )


class CountingToyEnv(ToyEnv):
    def __init__(self):
        self.step_calls = 0

    def step(self, state: ToyState, action: jax.Array) -> ToyState:
        self.step_calls += 1
        return super().step(state, action)


def _state(
    *,
    player: int = 0,
    legal=(True, False),
    terminal: bool = False,
    rewards=(0.0, 0.0),
    obs: float = 0.0,
) -> ToyState:
    return ToyState(
        observation=jnp.array([obs], dtype=jnp.float32),
        legal_action_mask=jnp.array(legal),
        current_player=jnp.array(player, dtype=jnp.int32),
        terminated=jnp.array(terminal),
        rewards=jnp.array(rewards, dtype=jnp.float32),
    )


def _config(**overrides):
    values = dict(
        c_leaf=1.0,
        c_terminal=8.0,
        c_state=0.5,
        c_value_search=1.0,
        policy_mc_samples=8,
        backup_mc_samples=1,
        selfplay_action_source="posterior_argmax",
        num_simulations=1,
        inflight_limit=1,
        search_eval_batch_size=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _tree(root_state=None, **overrides) -> PosteriorTree:
    if root_state is None:
        root_state = _state(legal=(True, False))
    params = dict(
        env=ToyEnv(),
        root_state=root_state,
        root_logits=np.zeros((2,), dtype=np.float32),
        root_alpha_v=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        root_alpha_q=np.ones((2, 3), dtype=np.float32),
        tree_index=0,
        rng=np.random.default_rng(0),
        c_leaf=1.0,
        c_terminal=8.0,
        c_state=0.5,
        c_value_search=1.0,
        policy_mc_samples=8,
        backup_mc_samples=1,
        commit="posterior_argmax",
    )
    params.update(overrides)
    return PosteriorTree(**params)


def test_edge_base_switches_to_expanded_child_value_prior_with_alignment():
    tree = _tree(
        root_alpha_q=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    )
    child_id = tree._add_node(
        state=_state(player=1),
        parent=0,
        action_from_parent=0,
        expanded=True,
        in_flight=False,
        prior_logits=np.zeros((2,), dtype=np.float32),
        alpha_v=np.array([2.0, 3.0, 7.0], dtype=np.float32),
        alpha_q=np.ones((2, 3), dtype=np.float32),
    )
    tree.nodes[0].children[0] = child_id

    assert np.allclose(tree.edge_base(0, 0), np.array([7.0, 3.0, 2.0]))
    assert np.allclose(tree.edge_base(0, 1), np.array([4.0, 5.0, 6.0]))


def test_backup_path_adds_direct_leaf_and_ancestor_state_posterior_evidence():
    tree = _tree(root_state=_state(player=0, legal=(True, False)))
    child_id = tree._add_node(
        state=_state(player=1, legal=(False, True)),
        parent=0,
        action_from_parent=0,
        expanded=True,
        in_flight=False,
        prior_logits=np.zeros((2,), dtype=np.float32),
        alpha_v=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        alpha_q=np.ones((2, 3), dtype=np.float32),
    )
    leaf_id = tree._add_node(
        state=_state(player=0, legal=(True, False)),
        parent=child_id,
        action_from_parent=1,
        expanded=True,
        in_flight=False,
        prior_logits=np.zeros((2,), dtype=np.float32),
        alpha_v=np.array([1.0, 1.0, 3.0], dtype=np.float32),
        alpha_q=np.ones((2, 3), dtype=np.float32),
    )
    tree.nodes[0].children[0] = child_id
    tree.nodes[child_id].children[1] = leaf_id

    tree.backup_path(
        ((0, 0), (child_id, 1)),
        leaf_id,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        1.0,
    )

    assert np.allclose(tree.nodes[child_id].evidence[1], np.array([1.0, 0.0, 0.0]))
    expected_child_posterior = np.array([4.0, 1.0, 1.0])
    assert np.allclose(tree.nodes[0].evidence[0], 0.5 * flip_outcome_np(expected_child_posterior))
    assert tree.nodes[child_id].visits[1] == 1
    assert tree.nodes[0].visits[0] == 1


def test_edge_posterior_is_base_plus_completed_evidence_only():
    tree = _tree(
        root_alpha_q=np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    )
    tree.nodes[0].evidence[0] = np.array([0.5, 1.5, 2.5], dtype=np.float32)

    assert np.allclose(tree.edge_posterior(0, 0), np.array([1.5, 3.5, 5.5]))
    assert np.allclose(tree.edge_posterior(0, 1), np.array([10.0, 20.0, 30.0]))


def test_state_search_posterior_uses_only_legal_action_posteriors():
    tree = _tree(root_state=_state(player=0, legal=(False, True)))
    tree.nodes[0].evidence[0] = np.array([100.0, 0.0, 0.0], dtype=np.float32)
    tree.nodes[0].evidence[1] = np.array([0.0, 0.0, 5.0], dtype=np.float32)

    beta = tree.state_search_posterior(0)

    assert np.allclose(beta, tree.edge_posterior(0, 1))


def test_consume_result_marks_expanded_and_backs_up_leaf_mean_evidence():
    tree = _tree(root_state=_state(player=0, legal=(True, False)), c_leaf=2.0)
    step_request = tree.next_step_request()
    assert step_request is not None
    request = tree.consume_step_result(
        step_request,
        tree.env.step(step_request.state, jnp.asarray(step_request.action, dtype=jnp.int32)),
    )
    assert request is not None

    tree.consume_result(
        request,
        logits=np.zeros((2,), dtype=np.float32),
        alpha_v=np.array([1.0, 1.0, 2.0], dtype=np.float32),
        alpha_q=np.ones((2, 3), dtype=np.float32),
    )

    leaf = tree.nodes[request.leaf_id]
    assert leaf.expanded
    assert not leaf.in_flight
    assert tree.inflight == 0
    assert tree.done == 1
    assert np.allclose(tree.nodes[0].evidence[0], 2.0 * np.array([0.5, 0.25, 0.25]))
    assert tree.nodes[0].visits[0] == 1


def test_next_step_request_returns_action_without_stepping_or_adding_posterior_mass():
    tree = _tree(root_state=_state(player=0, legal=(True, False)))

    request = tree.next_step_request()

    assert request is not None
    assert request.parent_id == 0
    assert request.action == 0
    assert tree.inflight == 0
    assert tree.done == 0
    assert np.allclose(tree.nodes[0].evidence, 0.0)
    assert tree.nodes[0].children[0] == NO_CHILD


def test_terminal_leaf_is_backed_up_without_calling_leaf_evaluator():
    tree = _tree(
        env=TerminalToyEnv(),
        root_state=_state(player=0, legal=(True, False)),
        c_terminal=8.0,
    )

    step_request = tree.next_step_request()
    assert step_request is not None
    request = tree.consume_step_result(
        step_request,
        tree.env.step(step_request.state, jnp.asarray(step_request.action, dtype=jnp.int32)),
    )

    assert request is None
    assert tree.done == 1
    assert tree.inflight == 0
    assert np.allclose(tree.nodes[0].evidence[0], np.array([0.0, 0.0, 8.0]))
    assert tree.nodes[0].visits[0] == 1


def test_finish_search_returns_theory_targets_and_scalar_q_action():
    tree = _tree(
        root_state=_state(player=0, legal=(True, True)),
        root_alpha_v=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        root_alpha_q=np.array([[2.0, 2.0, 1.0], [1.0, 2.0, 2.0]], dtype=np.float32),
        c_value_search=0.25,
        commit="scalar_q_argmax",
    )
    tree.nodes[0].evidence[0] = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    tree.nodes[0].evidence[1] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    action, policy, beta_q, beta_v, q_mass, alpha_root = tree.finish()

    expected_alpha = np.array([[2.0, 2.0, 3.0], [1.0, 3.0, 2.0]], dtype=np.float32)
    expected_value_proxy = np.sum(policy[:, None] * expected_alpha, axis=0)
    assert action == int(np.argmax(outcome_utility_np(outcome_mean_np(expected_alpha))))
    assert np.allclose(alpha_root, expected_alpha)
    assert np.allclose(beta_q, expected_alpha)
    assert np.allclose(beta_v, np.array([1.0, 1.0, 1.0]) + 0.25 * expected_value_proxy)
    assert np.allclose(q_mass, np.array([2.0, 1.0]))
    assert np.allclose(policy.sum(), 1.0)


def test_posterior_best_policy_target_matches_monte_carlo_definition_for_one_seed():
    alpha = np.array([[1.0, 1.0, 5.0], [5.0, 1.0, 1.0], [1.0, 10.0, 1.0]])
    legal = np.array([True, False, True])
    samples = 16
    seed = 123

    target = posterior_best_policy_target_np(np.random.default_rng(seed), alpha, legal, samples)

    rng = np.random.default_rng(seed)
    counts = np.zeros((3,), dtype=np.float32)
    legal_actions = np.flatnonzero(legal)
    for _ in range(samples):
        sampled = np.stack([rng.dirichlet(alpha[action]) for action in legal_actions])
        counts[legal_actions[int(np.argmax(outcome_utility_np(sampled)))]] += 1
    expected = counts / samples
    assert np.allclose(target, expected)


def test_run_search_uses_leaf_evaluator_for_root_and_leaf_batches():
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_posterior_tree_search(
        env=ToyEnv(),
        root_states=[_state(player=0, legal=(True, False))],
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(0),
        config=_config(),
    )

    assert calls == [1, 1]
    assert output.action.shape == (1,)
    assert output.action_weights.shape == (1, 2)
    assert output.beta_Q_target.shape == (1, 2, 3)
    assert output.beta_V_target.shape == (1, 3)
    assert np.allclose(np.asarray(output.q_evidence_mass[0, 0]), 1.0)


def test_run_search_batches_multiple_roots_and_keeps_targets_normalized():
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_posterior_tree_search(
        env=ToyEnv(),
        root_states=[
            _state(player=0, legal=(True, False), obs=0.0),
            _state(player=0, legal=(True, False), obs=10.0),
            _state(player=0, legal=(True, False), obs=20.0),
        ],
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(5),
        config=_config(num_simulations=2, search_eval_batch_size=2, policy_mc_samples=4),
    )

    assert calls[0] == 3
    assert max(calls[1:]) <= 2
    assert output.action_weights.shape == (3, 2)
    assert np.allclose(np.asarray(output.action_weights.sum(axis=-1)), 1.0)
    assert np.all(np.asarray(output.q_evidence_mass) >= 0.0)


def test_run_search_steps_all_roots_with_one_batched_env_call_per_wave():
    env = CountingToyEnv()

    def leaf_evaluator(obs):
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_posterior_tree_search(
        env=env,
        root_states=[
            _state(player=0, legal=(True, False), obs=0.0),
            _state(player=0, legal=(True, False), obs=10.0),
            _state(player=0, legal=(True, False), obs=20.0),
        ],
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(6),
        config=_config(num_simulations=1, search_eval_batch_size=8),
    )

    assert env.step_calls == 1
    assert output.action.shape == (3,)


def test_run_search_only_evaluates_active_leaf_states_after_batched_step():
    env = CountingToyEnv()
    calls = []

    def leaf_evaluator(obs):
        calls.append(np.asarray(obs).reshape((obs.shape[0], -1))[:, 0].tolist())
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_posterior_tree_search(
        env=env,
        root_states=[
            _state(
                player=0,
                legal=(False, False),
                terminal=True,
                rewards=(1.0, -1.0),
                obs=0.0,
            ),
            _state(player=0, legal=(True, False), obs=10.0),
            _state(player=0, legal=(True, False), obs=20.0),
        ],
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(7),
        config=_config(num_simulations=1, search_eval_batch_size=8),
    )

    assert env.step_calls == 1
    assert calls == [[0.0, 10.0, 20.0], [11.0, 21.0]]
    assert output.action.shape == (3,)
