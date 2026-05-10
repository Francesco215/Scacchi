from types import SimpleNamespace

import jax.numpy as jnp

from scacchi.loss import make_compute_loss_input
from scacchi.play import SelfplayOutput


def test_compute_loss_input_carries_stored_posterior_targets_directly():
    beta_V_target = jnp.array(
        [
            [[2.0, 1.0, 3.0]],
            [[4.0, 2.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    beta_Q_target = jnp.array(
        [
            [[[2.0, 1.0, 3.0], [1.0, 1.0, 1.0]]],
            [[[4.0, 2.0, 1.0], [1.0, 5.0, 1.0]]],
        ],
        dtype=jnp.float32,
    )
    q_target_mask = jnp.array(
        [
            [[True, False]],
            [[True, True]],
        ]
    )
    data = SelfplayOutput(
        obs=jnp.zeros((2, 1, 2, 2, 1), dtype=jnp.float32),
        reward=jnp.zeros((2, 1), dtype=jnp.float32),
        terminated=jnp.array([[False], [True]]),
        policy_target=jnp.array([[[0.75, 0.25]], [[0.25, 0.75]]], dtype=jnp.float32),
        played_action=jnp.zeros((2, 1), dtype=jnp.int32),
        discount=jnp.zeros((2, 1), dtype=jnp.float32),
        beta_V_target=beta_V_target,
        value_target_mask=jnp.array([[True], [False]]),
        beta_Q_target=beta_Q_target,
        q_target_mask=q_target_mask,
    )

    sample = make_compute_loss_input(SimpleNamespace())(data)

    assert jnp.array_equal(sample.beta_V_tgt, beta_V_target)
    assert jnp.array_equal(sample.beta_Q_tgt, beta_Q_target)
    assert jnp.array_equal(sample.q_tgt_mask, q_target_mask)
