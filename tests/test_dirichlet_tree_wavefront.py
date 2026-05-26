from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from scacchi.dirichlet_tree.search import (
    run_wavefront_posterior_tree_search,
    run_wavefront_posterior_tree_search_state_batch,
)
from scacchi.dirichlet_tree.store import InMemoryNodeStore
from scacchi.train import SearchConfig, SearchConstantsConfig, SearchMonteCarloConfig


class ToyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    terminated: jax.Array
    rewards: jax.Array


class CountingToyEnv:
    def __init__(self, *, terminal=False, transposition=False):
        self.step_calls = 0
        self.terminal = terminal
        self.transposition = transposition

    def step(self, state: ToyState, action: jax.Array) -> ToyState:
        del action
        self.step_calls += 1
        obs = jnp.where(
            self.transposition,
            jnp.array([1.0], dtype=jnp.float32),
            state.observation + 1.0,
        )
        terminated = jnp.asarray(self.terminal)
        rewards = jnp.where(
            terminated,
            jnp.array([1.0, -1.0], dtype=jnp.float32),
            jnp.array([0.0, 0.0], dtype=jnp.float32),
        )
        return ToyState(
            observation=obs,
            legal_action_mask=jnp.array([True, False]),
            current_player=1 - state.current_player,
            terminated=terminated,
            rewards=rewards,
        )


def _state(obs=0.0):
    return ToyState(
        observation=jnp.array([obs], dtype=jnp.float32),
        legal_action_mask=jnp.array([True, False]),
        current_player=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False),
        rewards=jnp.array([0.0, 0.0], dtype=jnp.float32),
    )


def _terminal_state(obs=0.0):
    return ToyState(
        observation=jnp.array([obs], dtype=jnp.float32),
        legal_action_mask=jnp.array([False, False]),
        current_player=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(True),
        rewards=jnp.array([1.0, -1.0], dtype=jnp.float32),
    )


def _stack_states(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _config(**overrides):
    return SearchConfig(
        num_simulations=overrides.pop("num_simulations", 1),
        inflight_limit=overrides.pop("inflight_limit", 1),
        max_depth=overrides.pop("max_depth", 8),
        eval_batch_size=overrides.pop("eval_batch_size", 16),
        monte_carlo=SearchMonteCarloConfig(
            policy_samples=overrides.pop("policy_samples", 8),
        ),
        constants=SearchConstantsConfig(
            kappa_leaf=overrides.pop("kappa_leaf", 1.0),
            state_posterior_kappa_n=overrides.pop("state_posterior_kappa_n", 1.0),
        ),
        **overrides,
    )


def test_wavefront_steps_multiple_roots_in_one_batched_call_and_finishes_targets():
    env = CountingToyEnv()
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search(
        env=env,
        root_states=[_state(0.0), _state(10.0), _state(20.0)],
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(0),
        config=_config(),
    )

    assert env.step_calls == 1
    assert calls == [3, 16]
    assert output.action.shape == (3,)
    assert output.action_weights.shape == (3, 2)
    assert np.allclose(np.asarray(output.action_weights.sum(axis=-1)), 1.0)
    assert np.allclose(np.asarray(output.action_weights[:, 1]), 0.0)
    assert np.allclose(np.asarray(output.q_loss_weight.sum(axis=-1)), 1.0)


def test_wavefront_search_result_includes_stability_diagnostics():
    env = CountingToyEnv()

    def leaf_evaluator(obs):
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search_state_batch(
        env=env,
        root_state_batch=_stack_states([_state(0.0), _state(10.0)]),
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(13),
        config=_config(num_simulations=2),
    )

    diagnostics = output.diagnostics
    assert diagnostics is not None
    assert diagnostics.path_depth_mean.shape == (2,)
    assert np.all(np.asarray(diagnostics.path_depth_mean) >= 1.0)
    assert np.all(np.asarray(diagnostics.path_depth_max) >= 1.0)
    assert np.all(np.asarray(diagnostics.expanded_nodes) >= 1.0)
    assert np.all(np.asarray(diagnostics.root_downstream_eval_count) >= 1.0)
    assert np.all(np.asarray(diagnostics.root_gamma) > 0.0)
    assert np.all(np.asarray(diagnostics.root_policy_entropy) >= 0.0)


def test_wavefront_terminal_lanes_skip_leaf_evaluator_after_root_eval():
    env = CountingToyEnv(terminal=True)
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search(
        env=env,
        root_states=[_state(0.0), _state(10.0)],
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(1),
        config=_config(),
    )

    assert calls == [2]
    assert np.allclose(np.asarray(output.q_loss_weight.sum(axis=-1)), 1.0)


def test_wavefront_terminal_root_lanes_finish_without_stalling():
    env = CountingToyEnv()
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search_state_batch(
        env=env,
        root_state_batch=_stack_states([_terminal_state(99.0), _state(0.0)]),
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(11),
        config=_config(),
    )

    assert calls == [2, 16]
    assert output.action.shape == (2,)
    assert np.isclose(float(output.q_loss_weight[0, 0]), 0.0)
    assert float(output.q_loss_weight[1, 0]) > 0.0


def test_store_wavefront_terminal_root_lanes_finish_without_stalling():
    env = CountingToyEnv()
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search(
        env=env,
        root_states=[_terminal_state(99.0), _state(0.0)],
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(12),
        config=_config(),
        store=InMemoryNodeStore(),
    )

    assert calls == [2, 1]
    assert output.action.shape == (2,)
    assert np.isclose(float(output.q_loss_weight[0, 0]), 0.0)
    assert float(output.q_loss_weight[1, 0]) > 0.0


def test_wavefront_duplicate_lanes_evaluate_unique_leaf_once():
    env = CountingToyEnv(transposition=True)
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search(
        env=env,
        root_states=[_state(0.0)],
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(2),
        config=_config(inflight_limit=2),
    )

    assert calls == [1, 16]
    assert output.action.shape == (1,)
    assert np.isclose(float(output.q_loss_weight[0, 0]), 1.0)


def test_wavefront_state_batch_entrypoint_avoids_root_state_list():
    env = CountingToyEnv()
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search_state_batch(
        env=env,
        root_state_batch=_stack_states([_state(0.0), _state(10.0), _state(20.0)]),
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(3),
        config=_config(),
    )

    assert env.step_calls == 1
    assert calls == [3, 16]
    assert output.action.shape == (3,)
    assert np.allclose(np.asarray(output.action_weights.sum(axis=-1)), 1.0)
    assert np.allclose(np.asarray(output.action_weights[:, 1]), 0.0)
    assert np.all(np.asarray(output.q_loss_weight[:, 0]) > 0.0)


def test_wavefront_state_batch_deduplicates_identical_roots():
    env = CountingToyEnv()
    calls = []

    def leaf_evaluator(obs):
        calls.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search_state_batch(
        env=env,
        root_state_batch=_stack_states([_state(0.0), _state(0.0), _state(0.0)]),
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(4),
        config=_config(),
    )

    assert calls == [1, 16]
    assert output.action.shape == (3,)
    assert np.allclose(np.asarray(output.action_weights[0]), np.asarray(output.action_weights[1]))
    assert np.allclose(np.asarray(output.q_loss_weight[:, 0]), 1.0)


def test_wavefront_tree_training_exports_algorithm_eligible_nodes():
    env = CountingToyEnv()

    def leaf_evaluator(obs):
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search_state_batch(
        env=env,
        root_state_batch=_stack_states([_state(0.0)]),
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(5),
        config=_config(num_simulations=2),
    )

    assert output.tree_data is not None
    assert output.tree_data.obs.shape[0] == 3
    assert int(np.sum(np.asarray(output.tree_data.policy_loss_mask))) == 2
    assert int(np.sum(np.asarray(output.tree_data.value_loss_mask))) == 2
    assert int(np.sum(np.asarray(output.tree_data.outcome_mask))) == 0
    active_rows = np.flatnonzero(np.asarray(output.tree_data.policy_loss_mask))
    assert set(np.asarray(output.tree_data.obs[active_rows]).reshape((-1,)).tolist()) == {0.0, 1.0}
    assert np.all(np.asarray(output.tree_data.q_loss_weight[active_rows, 0]) > 0.0)


def test_wavefront_tree_training_includes_eligible_root():
    env = CountingToyEnv()

    def leaf_evaluator(obs):
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search_state_batch(
        env=env,
        root_state_batch=_stack_states([_state(0.0)]),
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(7),
        config=_config(num_simulations=1),
    )

    assert output.tree_data is not None
    assert int(np.sum(np.asarray(output.tree_data.policy_loss_mask))) == 1
    row = int(np.argmax(np.asarray(output.tree_data.policy_loss_mask)))
    assert np.allclose(np.asarray(output.tree_data.obs[row]), np.array([0.0], dtype=np.float32))
    assert np.asarray(output.tree_data.q_loss_weight[row, 0]) > 0.0


def test_wavefront_tree_training_excludes_terminal_leaves():
    env = CountingToyEnv(terminal=True)

    def leaf_evaluator(obs):
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    output = run_wavefront_posterior_tree_search_state_batch(
        env=env,
        root_state_batch=_stack_states([_state(0.0), _state(10.0)]),
        leaf_evaluator=leaf_evaluator,
        rng_key=jax.random.PRNGKey(6),
        config=_config(),
    )

    assert output.tree_data is not None
    assert output.tree_data.obs.shape[0] == 4
    assert int(np.sum(np.asarray(output.tree_data.policy_loss_mask))) == 2
    assert int(np.sum(np.asarray(output.tree_data.value_loss_mask))) == 2
    assert int(np.sum(np.asarray(output.tree_data.outcome_mask))) == 0
    assert int(np.sum(np.asarray(output.tree_data.v_target_kind == 2))) == 2
