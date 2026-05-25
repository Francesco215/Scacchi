from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from scacchi.exact_hex import exact_hex_actions, relabel_selfplay_with_exact_hex
from scacchi.play import SelfplayOutput


def test_exact_hex_relabels_empty_4x4_root_with_wdl_targets():
    obs = jnp.zeros((1, 1, 4, 4, 4), dtype=jnp.bool_)
    legal = jnp.ones((1, 1, 17), dtype=jnp.bool_).at[..., -1].set(False)
    data = SelfplayOutput(
        obs=obs,
        reward=jnp.zeros((1, 1)),
        terminated=jnp.zeros((1, 1), dtype=jnp.bool_),
        action_weights=jnp.zeros((1, 1, 17), dtype=jnp.float32),
        played_action=jnp.zeros((1, 1), dtype=jnp.int32),
        legal_action_mask=legal,
        beta_Q_target=jnp.zeros((1, 1, 17, 3), dtype=jnp.float32),
        beta_V_target=jnp.zeros((1, 1, 3), dtype=jnp.float32),
        q_loss_weight=jnp.zeros((1, 1, 17), dtype=jnp.float32),
        discount=-jnp.ones((1, 1), dtype=jnp.float32),
    )
    config = SimpleNamespace(
        board_size=4,
        num_outcomes=3,
        kappa_terminal=8.0,
        epsilon_terminal=0.05,
        exact_hex_solver_extra_batch_size=2,
    )

    relabeled = relabel_selfplay_with_exact_hex(data, config, jax.random.PRNGKey(0))

    assert relabeled.obs.shape == (1, 3, 4, 4, 4)
    assert relabeled.action_weights.shape == (1, 3, 17)
    expected_best = np.zeros((17,), dtype=np.float32)
    expected_best[[3, 6, 9, 12]] = 0.25
    assert np.allclose(np.asarray(relabeled.action_weights[0, 0]), expected_best)
    assert np.allclose(np.asarray(relabeled.q_loss_weight[0, 0]), expected_best)
    assert np.allclose(np.asarray(relabeled.beta_V_target[0, 0]), [0.05, 0.05, 8.05])
    assert np.allclose(np.asarray(relabeled.beta_Q_target[0, 0, 3]), [0.05, 0.05, 8.05])
    assert np.allclose(np.asarray(relabeled.beta_Q_target[0, 0, 0]), [8.05, 0.05, 0.05])
    assert bool(relabeled.search_loss_mask[0, 0])
    assert relabeled.tree_data is None

    action = exact_hex_actions(obs[0], legal[0], config)
    assert int(action[0]) == 3
