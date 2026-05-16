from types import SimpleNamespace

import jax.numpy as jnp
import optax

from scacchi.loss import Sample, _compute_losses, make_compute_loss_input
from scacchi.play import SelfplayOutput


def test_compute_loss_input_preserves_root_legal_action_mask():
    data = SelfplayOutput(
        obs=jnp.zeros((3, 2, 1)),
        reward=jnp.zeros((3, 2)),
        terminated=jnp.array(
            [
                [False, False],
                [True, False],
                [False, False],
            ]
        ),
        action_weights=jnp.zeros((3, 2, 4)),
        legal_action_mask=jnp.array(
            [
                [[True, True, False, False], [True, False, True, False]],
                [[False, True, True, False], [True, True, False, False]],
                [[True, False, False, True], [False, True, False, True]],
            ]
        ),
        discount=-jnp.ones((3, 2)),
    )
    config = SimpleNamespace(max_num_steps=3, selfplay_batch_size=2)

    sample = make_compute_loss_input(config)(data)

    assert jnp.array_equal(sample.policy_mask, data.legal_action_mask)
    assert jnp.array_equal(
        sample.value_mask,
        jnp.array(
            [
                [True, False],
                [True, False],
                [False, False],
            ]
        ),
    )


def test_policy_loss_ignores_illegal_logits():
    data = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[1.0, 0.0, 0.0]]),
        value_tgt=jnp.array([0.0]),
        policy_mask=jnp.array([[True, False, True]]),
        value_mask=jnp.array([True]),
    )
    value = jnp.array([0.0])

    high_illegal_loss, _ = _compute_losses(
        jnp.array([[0.0, 1000.0, 0.0]]),
        value,
        data,
    )
    low_illegal_loss, _ = _compute_losses(
        jnp.array([[0.0, -1000.0, 0.0]]),
        value,
        data,
    )

    assert jnp.allclose(high_illegal_loss, low_illegal_loss)
    assert jnp.allclose(high_illegal_loss, jnp.log(2.0))


def test_value_mask_excludes_policy_and_value_losses_from_average():
    data = Sample(
        obs=jnp.zeros((2, 1)),
        policy_tgt=jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        value_tgt=jnp.array([0.0, 0.0]),
        policy_mask=jnp.ones((2, 2), dtype=jnp.bool_),
        value_mask=jnp.array([True, False]),
    )
    logits = jnp.array(
        [
            [0.0, 0.0],
            [-1000.0, 1000.0],
        ]
    )
    value = jnp.array([0.0, 1000.0])

    policy_loss, value_loss = _compute_losses(logits, value, data)

    expected_policy_loss = optax.softmax_cross_entropy(
        logits[:1],
        data.policy_tgt[:1],
    )[0]
    assert jnp.allclose(policy_loss, expected_policy_loss)
    assert jnp.allclose(value_loss, 0.0)
