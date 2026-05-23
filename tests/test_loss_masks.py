from types import SimpleNamespace

import jax.numpy as jnp
import optax

from scacchi.loss import (
    DIRICHLET_KL_LOSS_CUTOFF,
    Sample,
    _compute_dirichlet_losses,
    _compute_losses,
    _dirichlet_kl,
    make_compute_loss_input,
)
from scacchi.play import SelfplayOutput


def _sample_posterior_fields(num_rows: int, num_actions: int = 2, num_outcomes: int = 2):
    return {
        "beta_Q_target": jnp.ones((num_rows, num_actions, num_outcomes)),
        "beta_V_target": jnp.ones((num_rows, num_outcomes)),
        "q_evidence_mass": jnp.zeros((num_rows, num_actions)),
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
        beta_V_target=jnp.ones((3, 2, 2)),
        q_evidence_mass=jnp.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
                [[0.0, 3.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 5.0], [0.0, 6.0, 0.0, 0.0]],
            ]
        ),
        discount=-jnp.ones((3, 2)),
    )
    config = SimpleNamespace(max_num_steps=3, selfplay_batch_size=2)

    sample = make_compute_loss_input(config)(data)

    assert jnp.array_equal(sample.policy_mask, data.legal_action_mask)
    assert jnp.array_equal(sample.played_action, data.played_action)
    assert jnp.array_equal(sample.beta_Q_target, data.beta_Q_target)
    assert jnp.array_equal(sample.beta_V_target, data.beta_V_target)
    assert jnp.array_equal(sample.q_evidence_mass, data.q_evidence_mass)
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
        value_dir_kl_weight=0.0,
        q_dir_kl_weight=0.0,
        value_outcome_weight=1.0,
        q_outcome_weight=0.25,
    )

    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    assert jnp.allclose(metrics.value_outcome_loss, -jnp.log(0.75))
    assert jnp.allclose(metrics.q_outcome_loss, -jnp.log(0.8))


def test_dirichlet_kl_is_zero_for_identical_parameters_and_positive_otherwise():
    beta = jnp.array([[2.0, 3.0]])

    same = _dirichlet_kl(beta, beta)
    different = _dirichlet_kl(beta, jnp.array([[3.0, 2.0]]))

    assert jnp.allclose(same, 0.0, atol=1e-6)
    assert different[0] > 0.0


def test_dirichlet_kl_losses_use_value_policy_and_q_evidence_masks():
    data = Sample(
        obs=jnp.zeros((2, 1)),
        policy_tgt=jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        value_tgt=jnp.array([1.0, 1.0]),
        played_action=jnp.array([0, 0]),
        policy_mask=jnp.array(
            [
                [True, False, True],
                [True, True, True],
            ]
        ),
        value_mask=jnp.array([True, False]),
        beta_Q_target=jnp.array(
            [
                [[1.0, 1.0], [1000.0, 1.0], [1.0, 1.0]],
                [[1000.0, 1.0], [1000.0, 1.0], [1000.0, 1.0]],
            ]
        ),
        beta_V_target=jnp.array([[1.0, 2.0], [1000.0, 1.0]]),
        q_evidence_mass=jnp.array([[0.0, 100.0, 2.0], [100.0, 100.0, 100.0]]),
    )
    logits = jnp.zeros((2, 3))
    alpha_v = jnp.array([[1.0, 2.0], [1.0, 1000.0]])
    alpha_q = jnp.array(
        [
            [[1.0, 1.0], [1.0, 1000.0], [2.0, 1.0]],
            [[1.0, 1000.0], [1.0, 1000.0], [1.0, 1000.0]],
        ]
    )
    config = SimpleNamespace(
        policy_loss_weight=0.0,
        value_dir_kl_weight=1.0,
        q_dir_kl_weight=1.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
        dirichlet_kl_loss_cutoff=DIRICHLET_KL_LOSS_CUTOFF,
    )

    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    expected_q = _dirichlet_kl(data.beta_Q_target[0, 2], alpha_q[0, 2])
    assert jnp.allclose(metrics.value_dir_kl_loss, 0.0, atol=1e-6)
    assert jnp.allclose(metrics.q_dir_kl_loss, expected_q, atol=1e-6)
    assert jnp.allclose(metrics.q_evidence_mass_mean, 2.0)


def test_dirichlet_kl_losses_ignore_huge_and_nonfinite_terms():
    data = Sample(
        obs=jnp.zeros((3, 1)),
        policy_tgt=jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        value_tgt=jnp.array([1.0, 1.0, 1.0]),
        played_action=jnp.array([0, 0, 0]),
        policy_mask=jnp.ones((3, 3), dtype=jnp.bool_),
        value_mask=jnp.ones((3,), dtype=jnp.bool_),
        beta_Q_target=jnp.array(
            [
                [[2.0, 3.0], [1e6, 1.0], [jnp.nan, 1.0]],
                [[1e6, 1.0], [1e6, 1.0], [1e6, 1.0]],
                [[jnp.nan, 1.0], [jnp.nan, 1.0], [jnp.nan, 1.0]],
            ]
        ),
        beta_V_target=jnp.array(
            [
                [2.0, 3.0],
                [1e6, 1.0],
                [jnp.nan, 1.0],
            ]
        ),
        q_evidence_mass=jnp.ones((3, 3)),
    )
    logits = jnp.zeros((3, 3))
    alpha_v = jnp.array([[2.0, 3.0], [1.0, 1e6], [1.0, 1.0]])
    alpha_q = jnp.array(
        [
            [[2.0, 3.0], [1.0, 1e6], [1.0, 1.0]],
            [[1.0, 1e6], [1.0, 1e6], [1.0, 1e6]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ]
    )
    config = SimpleNamespace(
        policy_loss_weight=0.0,
        value_dir_kl_weight=1.0,
        q_dir_kl_weight=1.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
    )

    raw_value_kl = _dirichlet_kl(data.beta_V_target, alpha_v)
    raw_q_kl = _dirichlet_kl(data.beta_Q_target, alpha_q)
    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    assert raw_value_kl[1] > DIRICHLET_KL_LOSS_CUTOFF
    assert not bool(jnp.isfinite(raw_value_kl[2]))
    assert raw_q_kl[0, 1] > DIRICHLET_KL_LOSS_CUTOFF
    assert not bool(jnp.isfinite(raw_q_kl[0, 2]))
    assert jnp.allclose(metrics.value_dir_kl_loss, 0.0, atol=1e-6)
    assert jnp.allclose(metrics.q_dir_kl_loss, 0.0, atol=1e-6)


def test_dirichlet_kl_loss_cutoff_comes_from_config():
    data = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[1.0]]),
        value_tgt=jnp.array([1.0]),
        played_action=jnp.array([0]),
        policy_mask=jnp.array([[True]]),
        value_mask=jnp.array([True]),
        beta_Q_target=jnp.array([[[2.0, 3.0]]]),
        beta_V_target=jnp.array([[2.0, 3.0]]),
        q_evidence_mass=jnp.array([[1.0]]),
    )
    logits = jnp.zeros((1, 1))
    alpha_v = jnp.array([[3.0, 2.0]])
    alpha_q = jnp.array([[[3.0, 2.0]]])
    raw_kl = _dirichlet_kl(data.beta_V_target, alpha_v)[0]

    keep_config = SimpleNamespace(
        policy_loss_weight=0.0,
        value_dir_kl_weight=1.0,
        q_dir_kl_weight=1.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
        dirichlet_kl_loss_cutoff=raw_kl + 1e-3,
    )
    drop_config = SimpleNamespace(
        policy_loss_weight=0.0,
        value_dir_kl_weight=1.0,
        q_dir_kl_weight=1.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
        dirichlet_kl_loss_cutoff=raw_kl - 1e-3,
    )

    _, keep_metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, keep_config)
    _, drop_metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, drop_config)

    assert jnp.allclose(keep_metrics.value_dir_kl_loss, raw_kl)
    assert jnp.allclose(drop_metrics.value_dir_kl_loss, 0.0)


def test_policy_kl_hat_is_nll_minus_sampled_target_entropy():
    data = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[0.25, 0.75]]),
        value_tgt=jnp.array([1.0]),
        played_action=jnp.array([1]),
        policy_mask=jnp.array([[True, True]]),
        value_mask=jnp.array([True]),
        beta_Q_target=jnp.ones((1, 2, 2)),
        beta_V_target=jnp.ones((1, 2)),
        q_evidence_mass=jnp.zeros((1, 2)),
    )
    logits = jnp.array([[0.0, 0.0]])
    alpha_v = jnp.ones((1, 2))
    alpha_q = jnp.ones((1, 2, 2))
    config = SimpleNamespace(
        policy_loss_weight=1.0,
        value_dir_kl_weight=0.0,
        q_dir_kl_weight=0.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
    )

    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    expected_entropy = -jnp.sum(data.policy_tgt[0] * jnp.log(data.policy_tgt[0]))
    assert jnp.allclose(metrics.policy_target_entropy, expected_entropy)
    assert jnp.allclose(
        metrics.policy_kl_hat,
        metrics.policy_nll_loss - metrics.policy_target_entropy,
    )
