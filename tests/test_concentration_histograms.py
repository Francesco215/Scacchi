import numpy as np
import jax.numpy as jnp

from scacchi.dirichlet_mctx.native_targets import (
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
)
from scacchi.loss import (
    CONCENTRATION_HISTOGRAM_BIN_EDGES,
    CONCENTRATION_HISTOGRAM_NUM_BINS,
    CONCENTRATION_HISTOGRAM_SERIES,
    Sample,
    _compute_dirichlet_losses,
    _masked_concentration_histogram_counts,
    _zero_train_metrics_like,
)
from scacchi.types import (
    Config,
    ModelConfig,
    Network,
    SelfplayConfig,
    TrainingConfig,
    TrainingLossConfig,
)


def _alpha_with_mass(mass: jnp.ndarray) -> jnp.ndarray:
    return mass[..., None] * jnp.asarray([0.25, 0.75])


def test_fixed_concentration_histogram_boundaries_mask_and_overflow():
    edges = jnp.asarray(CONCENTRATION_HISTOGRAM_BIN_EDGES)
    concentration = jnp.asarray(
        [
            0.0,
            edges[1] / 2.0,
            edges[1],
            edges[2],
            edges[-2],
            edges[-1],
            edges[-1] * 2.0,
            jnp.nan,
            jnp.inf,
        ]
    )
    mask = jnp.asarray(
        [True, True, True, False, True, True, True, True, True]
    )

    counts = _masked_concentration_histogram_counts(concentration, mask)

    assert counts.shape == (CONCENTRATION_HISTOGRAM_NUM_BINS,)
    assert jnp.sum(counts) == 6
    assert counts[0] == 2
    assert counts[1] == 1
    assert counts[-1] == 3
    assert jnp.count_nonzero(counts) == 3


def test_scalar_model_metrics_have_zero_histogram_with_stable_shape():
    metrics = _zero_train_metrics_like(jnp.asarray(0.0))

    assert metrics.dirichlet_concentration_histogram_counts.shape == (
        len(CONCENTRATION_HISTOGRAM_SERIES),
        CONCENTRATION_HISTOGRAM_NUM_BINS,
    )
    assert jnp.all(metrics.dirichlet_concentration_histogram_counts == 0)


def test_dirichlet_histograms_compare_prior_and_posterior_on_same_masks():
    alpha_v_mass = jnp.asarray([1.0, 2.0, 4.0])
    beta_v_mass = jnp.asarray([2.0, 3.0, 8.0])
    alpha_q_mass = jnp.asarray(
        [[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]
    )
    beta_q_mass = jnp.asarray(
        [[2.0, 3.0], [5.0, 9.0], [17.0, 33.0]]
    )
    alpha_v = _alpha_with_mass(alpha_v_mass)
    beta_v = _alpha_with_mass(beta_v_mass)
    alpha_q = _alpha_with_mass(alpha_q_mass)
    beta_q = _alpha_with_mass(beta_q_mass)

    q_target_kind = jnp.asarray(
        [
            [TARGET_DIRICHLET, TARGET_CATEGORICAL],
            [TARGET_DIRICHLET, TARGET_DIRICHLET],
            [TARGET_CATEGORICAL, TARGET_DIRICHLET],
        ],
        dtype=jnp.int8,
    )
    v_target_kind = jnp.asarray(
        [TARGET_DIRICHLET, TARGET_CATEGORICAL, TARGET_DIRICHLET],
        dtype=jnp.int8,
    )
    q_loss_weight = jnp.asarray([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    data = Sample(
        obs=jnp.zeros((3, 1)),
        policy_tgt=jnp.full((3, 2), 0.5),
        value_tgt=jnp.ones((3,)),
        played_action=jnp.zeros((3,), dtype=jnp.int32),
        policy_mask=jnp.ones((3, 2), dtype=jnp.bool_),
        value_mask=jnp.ones((3,), dtype=jnp.bool_),
        beta_Q_target=beta_q,
        beta_V_target=beta_v,
        q_loss_weight=q_loss_weight,
        q_target_kind=q_target_kind,
        q_target_weight=jnp.ones((3, 2)),
        q_target_outcome=jnp.zeros((3, 2), dtype=jnp.int8),
        q_target_distance=jnp.zeros((3, 2), dtype=jnp.int32),
        v_target_kind=v_target_kind,
        v_target_weight=jnp.ones((3,)),
        v_target_outcome=jnp.zeros((3,), dtype=jnp.int8),
        v_target_distance=jnp.zeros((3,), dtype=jnp.int32),
    )
    config = Config(
        model=ModelConfig(network=Network.boardlaw_dirichlet),
        selfplay=SelfplayConfig(max_num_steps=1),
        training=TrainingConfig(
            losses=TrainingLossConfig(
                value_dir_kl_weight=1.0,
                q_dir_kl_weight=1.0,
            )
        ),
    )

    _, metrics = _compute_dirichlet_losses(
        jnp.zeros((3, 2)),
        alpha_v,
        alpha_q,
        data,
        config,
    )
    counts = np.asarray(metrics.dirichlet_concentration_histogram_counts)
    edges = np.asarray(CONCENTRATION_HISTOGRAM_BIN_EDGES)
    selected = (
        np.asarray([1.0, 4.0]),
        np.asarray([2.0, 8.0]),
        np.asarray([1.0, 8.0, 32.0]),
        np.asarray([2.0, 9.0, 33.0]),
    )

    assert counts.shape == (
        len(CONCENTRATION_HISTOGRAM_SERIES),
        CONCENTRATION_HISTOGRAM_NUM_BINS,
    )
    assert tuple(counts.sum(axis=-1)) == (2.0, 2.0, 3.0, 3.0)
    for actual, values in zip(counts, selected, strict=True):
        expected, _ = np.histogram(values, bins=edges)
        np.testing.assert_array_equal(actual, expected)
