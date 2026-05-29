from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.play_search import _run_posterior_tree_search_step


class ToyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    terminated: jax.Array
    rewards: jax.Array


class ToyEnv:
    pass


def _batched_state(obs_values):
    obs = jnp.asarray(obs_values, dtype=jnp.float32)[:, None]
    batch = obs.shape[0]
    return ToyState(
        observation=obs,
        legal_action_mask=jnp.tile(jnp.array([[True, False]]), (batch, 1)),
        current_player=jnp.zeros((batch,), dtype=jnp.int32),
        terminated=jnp.zeros((batch,), dtype=jnp.bool_),
        rewards=jnp.zeros((batch, 2), dtype=jnp.float32),
    )


def _config(**overrides):
    values = dict(
        kappa_leaf=1.0,
        kappa_terminal=8.0,
        epsilon_terminal=1e-6,
        state_posterior_kappa_n=1.0,
        policy_mc_samples=8,
        backup_mc_samples=1,
        selfplay_action_source="posterior_argmax",
        num_simulations=1,
        inflight_limit=1,
        search_eval_batch_size=4,
        categorical_draw_rule="policy_prior",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_posterior_tree_search_step_uses_padded_fused_transition_eval():
    leaf_batch_sizes = []
    transition_batch_sizes = []

    def leaf_evaluator(obs):
        leaf_batch_sizes.append(int(obs.shape[0]))
        batch = obs.shape[0]
        logits = jnp.zeros((batch, 2), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, 2, 3), dtype=jnp.float32)
        return logits, alpha_v, alpha_q

    def transition_evaluator(states, actions):
        transition_batch_sizes.append(int(actions.shape[0]))
        child_states = ToyState(
            observation=states.observation + 1.0,
            legal_action_mask=states.legal_action_mask,
            current_player=1 - states.current_player,
            terminated=states.terminated,
            rewards=states.rewards,
        )
        return child_states, leaf_evaluator(child_states.observation)

    output = _run_posterior_tree_search_step(
        env=ToyEnv(),
        config=_config(),
        env_state=_batched_state([0.0, 10.0, 20.0]),
        leaf_evaluator=leaf_evaluator,
        transition_evaluator=transition_evaluator,
        search_key=jax.random.PRNGKey(0),
        device_put_cpu=lambda value: value,
    )

    assert leaf_batch_sizes == [3, 4]
    assert transition_batch_sizes == [4]
    assert output.action_weights.shape == (3, 2)
    assert output.played_action.shape == (3,)
    assert output.beta_Q_target.shape == (3, 2, 3)
    assert output.beta_V_target.shape == (3, 3)
    assert output.q_loss_weight.shape == (3, 2)
    assert output.search_loss_mask.shape == (3,)
    np.testing.assert_allclose(np.asarray(output.action_weights.sum(axis=-1)), 1.0)


def test_posterior_tree_search_step_rejects_non_dirichlet_transition_output():
    def leaf_evaluator(obs):
        batch = obs.shape[0]
        return (
            jnp.zeros((batch, 2), dtype=jnp.float32),
            jnp.ones((batch, 3), dtype=jnp.float32),
            jnp.ones((batch, 2, 3), dtype=jnp.float32),
        )

    def bad_transition_evaluator(states, actions):
        del actions
        return states, (jnp.zeros((states.observation.shape[0], 2)),)

    with pytest.raises(ValueError, match="transition_evaluator"):
        _run_posterior_tree_search_step(
            env=ToyEnv(),
            config=_config(),
            env_state=_batched_state([0.0]),
            leaf_evaluator=leaf_evaluator,
            transition_evaluator=bad_transition_evaluator,
            search_key=jax.random.PRNGKey(0),
            device_put_cpu=lambda value: value,
        )
