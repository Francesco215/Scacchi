from types import SimpleNamespace

import jax.numpy as jnp
import optax

from scacchi.loss import (
    Sample,
    _compute_dirichlet_losses,
    _compute_losses,
    make_compute_loss_input,
)
from scacchi.play import SelfplayOutput


def _sample_posterior_fields(num_rows: int, num_actions: int = 2, num_outcomes: int = 2):
    return {
        "beta_Q_target": jnp.zeros((num_rows, num_actions, num_outcomes)),
        "q_target_mask": jnp.zeros((num_rows, num_actions), dtype=jnp.bool_),
        "beta_V_target": jnp.zeros((num_rows, num_outcomes)),
        "value_target_mask": jnp.zeros((num_rows,), dtype=jnp.bool_),
    }


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
        played_action=jnp.array(
            [
                [0, 2],
                [1, 0],
                [3, 1],
            ]
        ),
        legal_action_mask=jnp.array(
            [
                [[True, True, False, False], [True, False, True, False]],
                [[False, True, True, False], [True, True, False, False]],
                [[True, False, False, True], [False, True, False, True]],
            ]
        ),
        beta_Q_target=jnp.ones((3, 2, 4, 2)),
        q_target_mask=jnp.array(
            [
                [[True, False, False, False], [False, False, True, False]],
                [[False, True, False, False], [True, False, False, False]],
                [[False, False, False, True], [False, True, False, False]],
            ]
        ),
        beta_V_target=jnp.ones((3, 2, 2)),
        value_target_mask=jnp.array(
            [
                [True, False],
                [True, True],
                [False, False],
            ]
        ),
        discount=-jnp.ones((3, 2)),
    )
    config = SimpleNamespace(max_num_steps=3, selfplay_batch_size=2)

    sample = make_compute_loss_input(config)(data)

    assert jnp.array_equal(sample.policy_mask, data.legal_action_mask)
    assert jnp.array_equal(sample.played_action, data.played_action)
    assert jnp.array_equal(sample.beta_Q_target, data.beta_Q_target)
    assert jnp.array_equal(sample.q_target_mask, data.q_target_mask)
    assert jnp.array_equal(sample.beta_V_target, data.beta_V_target)
    assert jnp.array_equal(sample.value_target_mask, data.value_target_mask)
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
        played_action=jnp.array([0]),
        policy_mask=jnp.array([[True, False, True]]),
        value_mask=jnp.array([True]),
        **_sample_posterior_fields(1, num_actions=3),
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
        played_action=jnp.array([0, 1]),
        policy_mask=jnp.ones((2, 2), dtype=jnp.bool_),
        value_mask=jnp.array([True, False]),
        **_sample_posterior_fields(2),
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


def test_dirichlet_outcome_losses_use_played_action():
    data = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[1.0, 0.0]]),
        value_tgt=jnp.array([1.0]),
        played_action=jnp.array([1]),
        policy_mask=jnp.array([[True, True]]),
        value_mask=jnp.array([True]),
        **_sample_posterior_fields(1),
    )
    logits = jnp.array([[0.0, 0.0]])
    alpha_v = jnp.array([[1.0, 3.0]])
    alpha_q = jnp.array([[[100.0, 1.0], [1.0, 4.0]]])
    config = SimpleNamespace(
        policy_loss_weight=1.0,
        value_outcome_weight=1.0,
        q_outcome_weight=0.25,
    )

    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    assert jnp.allclose(metrics.value_outcome_loss, -jnp.log(0.75))
    assert jnp.allclose(metrics.q_outcome_loss, -jnp.log(0.8))
