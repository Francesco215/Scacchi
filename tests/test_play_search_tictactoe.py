from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pgx

from scacchi.dirichlet_tree.native import OUTCOME_DRAW, TARGET_CATEGORICAL
from scacchi.play_search import _run_posterior_tree_search_step


def _dumb_tic_tac_toe_model(obs):
    batch = obs.shape[0]
    return (
        jnp.zeros((batch, 9), dtype=jnp.float32),
        jnp.ones((batch, 3), dtype=jnp.float32),
        jnp.ones((batch, 9, 3), dtype=jnp.float32),
    )


def test_pgx_tic_tac_toe_empty_root_solves_to_draw_with_dumb_model():
    env = pgx.make("tic_tac_toe")
    root = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 1))
    config = SimpleNamespace(
        search_backend="dqaz",
        solve_categorical=True,
        selfplay_action_source="posterior_argmax",
        num_simulations=100_000,
        policy_mc_samples=8,
        state_posterior_kappa_n=4.0,
        search_eval_batch_size=1024,
        search_pad_to_eval_batch=True,
        kappa_terminal=80.0,
        epsilon_terminal=0.01,
    )

    env_step = jax.jit(jax.vmap(env.step))

    def transition_evaluator(states, actions):
        child_states = env_step(states, actions)
        return child_states, _dumb_tic_tac_toe_model(child_states.observation)

    output = _run_posterior_tree_search_step(
        env=env,
        config=config,
        env_state=root,
        leaf_evaluator=_dumb_tic_tac_toe_model,
        transition_evaluator=transition_evaluator,
        search_key=jax.random.PRNGKey(1),
        device_put_cpu=lambda value: value,
    )

    legal = np.asarray(root.legal_action_mask)[0]
    assert bool(legal[int(np.asarray(output.played_action)[0])])
    np.testing.assert_allclose(np.asarray(output.action_weights).sum(axis=-1), 1.0)
    assert np.asarray(output.search_loss_mask).tolist() == [True]

    assert np.asarray(output.v_target_kind).tolist() == [int(TARGET_CATEGORICAL)]
    assert np.asarray(output.v_target_outcome).tolist() == [int(OUTCOME_DRAW)]
    assert int(np.asarray(output.v_target_distance)[0]) > 0

    q_kind = np.asarray(output.q_target_kind)[0]
    q_outcome = np.asarray(output.q_target_outcome)[0]
    assert np.all(q_kind[legal] == int(TARGET_CATEGORICAL))
    assert np.all(q_outcome[legal] == int(OUTCOME_DRAW))
