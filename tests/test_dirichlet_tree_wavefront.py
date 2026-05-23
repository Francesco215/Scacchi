from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from scacchi.dirichlet_tree.search import (
    run_wavefront_posterior_tree_search,
    run_wavefront_posterior_tree_search_state_batch,
)


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


def _stack_states(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _config(**overrides):
    values = dict(
        search_policy="posterior_tree_wavefront",
        num_simulations=1,
        wavefront_num_lanes_per_root=1,
        wavefront_max_depth=8,
        search_eval_batch_size=16,
        c_leaf=1.0,
        c_terminal=8.0,
        c_state=0.5,
        c_value_search=0.25,
        policy_mc_samples=8,
        wavefront_final_action_mode="argmax_q_mean",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


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
    assert np.all(np.asarray(output.q_evidence_mass[:, 0]) > 0.0)


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
    assert np.allclose(np.asarray(output.q_evidence_mass[:, 0]), 8.0)


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
        config=_config(wavefront_num_lanes_per_root=2),
    )

    assert calls == [1, 16]
    assert output.action.shape == (1,)
    assert np.isclose(float(output.q_evidence_mass[0, 0]), 1.0)


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
    assert np.all(np.asarray(output.q_evidence_mass[:, 0]) > 0.0)


def test_wavefront_eval_padding_can_be_disabled():
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
        rng_key=jax.random.PRNGKey(33),
        config=_config(wavefront_pad_eval_batches=False),
    )

    assert calls == [3, 3]
    assert output.action.shape == (3,)
    assert np.all(np.asarray(output.q_evidence_mass[:, 0]) > 0.0)


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
    assert np.allclose(np.asarray(output.q_evidence_mass[:, 0]), 1.0)
