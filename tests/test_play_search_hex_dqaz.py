from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from scacchi.envs import make_env
from scacchi.play_search import _run_posterior_tree_search_step


def test_pgx_hex_dqaz_wdl3_smoke():
    env = make_env("hex", 3)
    batch_size = 2
    root = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), batch_size))

    def leaf_evaluator(obs):
        batch = obs.shape[0]
        return (
            jnp.zeros((batch, env.num_actions), dtype=jnp.float32),
            jnp.ones((batch, 3), dtype=jnp.float32),
            jnp.ones((batch, env.num_actions, 3), dtype=jnp.float32),
        )

    env_step = jax.jit(jax.vmap(env.step))

    def transition_evaluator(states, actions):
        child_states = env_step(states, actions)
        return child_states, leaf_evaluator(child_states.observation)

    config = SimpleNamespace(
        search_backend="dqaz",
        solve_categorical=True,
        selfplay_action_source="posterior_argmax",
        num_simulations=2,
        policy_mc_samples=4,
        state_posterior_kappa_n=16.0,
        search_eval_batch_size=2,
        search_pad_to_eval_batch=True,
        kappa_terminal=8.0,
        epsilon_terminal=0.05,
    )

    output = _run_posterior_tree_search_step(
        env=env,
        config=config,
        env_state=root,
        leaf_evaluator=leaf_evaluator,
        transition_evaluator=transition_evaluator,
        search_key=jax.random.PRNGKey(1),
        device_put_cpu=lambda value: value,
    )

    legal = np.asarray(root.legal_action_mask)
    played = np.asarray(output.played_action)
    assert np.all(legal[np.arange(batch_size), played])
    assert output.action_weights.shape == (batch_size, env.num_actions)
    assert output.beta_Q_target.shape == (batch_size, env.num_actions, 3)
    assert output.beta_V_target.shape == (batch_size, 3)
    np.testing.assert_allclose(np.asarray(output.action_weights).sum(axis=-1), 1.0)
    assert np.asarray(output.search_loss_mask).tolist() == [True, True]
