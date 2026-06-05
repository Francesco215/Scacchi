import jax
import jax.numpy as jnp
import numpy as np

from scacchi.envs import make_env
from scacchi.play_search import (
    _flatten_padded_valid_actions_np,
    _padded_valid_actions_from_mask,
    _run_posterior_tree_search_step,
)
from scacchi.types import (
    Config,
    DQAZSearchConfig,
    EnvConfig,
    ModelConfig,
    SearchConfig,
    SearchConstantsConfig,
    SelfplayConfig,
)


def test_padded_valid_actions_from_mask_packs_active_legal_rows():
    legal_action_mask = jnp.asarray(
        [
            [False, True, False, True],
            [True, False, True, False],
            [True, True, False, False],
        ],
        dtype=jnp.bool_,
    )
    policy_logits = jnp.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ],
        dtype=jnp.float32,
    )
    alpha_q = jnp.arange(3 * 4 * 3, dtype=jnp.float32).reshape((3, 4, 3)) + 1.0
    terminated = jnp.asarray([False, False, True])
    active_mask = jnp.asarray([True, False, True])

    padded_actions, padded_logits, padded_q, valid_counts = (
        _padded_valid_actions_from_mask(
            legal_action_mask,
            policy_logits,
            alpha_q,
            terminated,
            active_mask,
        )
    )
    offsets, legal_actions, compact_logits, compact_q = (
        _flatten_padded_valid_actions_np(
            padded_actions,
            padded_logits,
            padded_q,
            valid_counts,
        )
    )

    np.testing.assert_array_equal(offsets, np.array([0, 2, 2, 2], dtype=np.int64))
    np.testing.assert_array_equal(legal_actions, np.array([1, 3], dtype=np.int32))
    np.testing.assert_allclose(compact_logits, np.array([1.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(compact_q, np.asarray(alpha_q[0, [1, 3]]))


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

    config = Config(
        env=EnvConfig(id="hex", board_size=3, num_outcomes=3),
        model=ModelConfig(network="boardlaw_dirichlet"),
        selfplay=SelfplayConfig(action_source="posterior_argmax"),
        search=SearchConfig(
            kind="dqaz",
            dqaz=DQAZSearchConfig(
                num_simulations=2,
                policy_samples=4,
                state_posterior_kappa_n=16.0,
                inflight_limit=2,
                eval_batch_size=2,
                pad_to_eval_batch=True,
                jax_backup=True,
                epsilon_terminal=0.05,
                constants=SearchConstantsConfig(kappa_terminal=8.0),
            ),
        ),
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
