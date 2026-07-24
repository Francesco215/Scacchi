from __future__ import annotations

from typing import cast

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import pgx

from scacchi.dirichlet_mctx.native_targets import (
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    TARGET_PAD,
)
from scacchi.dirichlet_mctx.base import RootFnOutput
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.tree import instantiate_tree_from_root
from scacchi.search_diagnostics import (
    DistillationDiscrepancy,
    SearchDiagnostics,
    dirichlet_kl,
    distillation_discrepancy,
    native_displacement,
    policy_displacement,
    root_search_diagnostics,
)
from scacchi.logger import training_metrics
from scacchi.loss import (
    Sample,
    _zero_train_metrics_like,
    evaluate_distillation_discrepancy,
    train,
)
from scacchi.network import build_model
from scacchi.pipeline import (
    _with_capture_diagnostics,
    _with_search_diagnostics,
    make_training_iteration,
    optimizer_updates_per_iteration,
)
from scacchi.types import (
    Config,
    DirichletThompsonSearchConfig,
    EnvConfig,
    ModelConfig,
    Network,
    QLossWeightMode,
    SearchConfig,
    SearchKind,
    SelfplayConfig,
    TrainingConfig,
    TrainingLossConfig,
)


class _TinyScalarProbeModel(nnx.Module):
    def __init__(self) -> None:
        self.policy_logits = nnx.Param(jnp.array([2.0, -2.0]))

    def __call__(
        self,
        obs: jax.Array,
        *,
        train: bool,
    ) -> tuple[jax.Array, jax.Array]:
        del train
        logits = jnp.broadcast_to(
            self.policy_logits[None, :],
            (obs.shape[0], 2),
        )
        return logits, jnp.zeros((obs.shape[0],), dtype=logits.dtype)


def test_dirichlet_kl_matches_closed_form_beta_example() -> None:
    target = jnp.array([[2.0, 1.0]])
    prior = jnp.array([[1.0, 1.0]])

    actual = dirichlet_kl(target, prior)

    assert jnp.allclose(actual, jnp.log(2.0) - 0.5, atol=1e-6)


def test_semantic_kl_ignores_mass_while_full_kl_detects_it() -> None:
    prior = jnp.array([[1.0, 3.0]])
    target = jnp.array([[2.0, 6.0]])
    kind = jnp.array([TARGET_DIRICHLET], dtype=jnp.int8)
    outcome = jnp.array([-1], dtype=jnp.int8)

    diagnostics = native_displacement(prior, target, kind, outcome)

    assert jnp.allclose(diagnostics.semantic_kl, 0.0, atol=1e-7)
    assert diagnostics.dirichlet_kl[0] > 0.0


def test_native_displacement_keeps_distinct_native_populations() -> None:
    prior = jnp.array(
        [
            [1.0, 1.0],
            [4.0, 1.0],
            [2.0, 2.0],
        ]
    )
    target = jnp.array(
        [
            [2.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    kind = jnp.array(
        [TARGET_DIRICHLET, TARGET_CATEGORICAL, TARGET_PAD],
        dtype=jnp.int8,
    )
    outcome = jnp.array([-1, 0, -1], dtype=jnp.int8)

    diagnostics = native_displacement(prior, target, kind, outcome)

    expected_mean_kl = (
        (2.0 / 3.0) * jnp.log(4.0 / 3.0)
        + (1.0 / 3.0) * jnp.log(2.0 / 3.0)
    )
    assert jnp.allclose(diagnostics.semantic_kl[0], expected_mean_kl)
    assert jnp.allclose(
        diagnostics.dirichlet_kl[0],
        jnp.log(2.0) - 0.5,
        atol=1e-6,
    )
    assert jnp.allclose(
        diagnostics.semantic_kl[1],
        -jnp.log(0.8),
    )
    assert jnp.allclose(
        diagnostics.categorical_surprisal[1],
        -jnp.log(0.8),
    )
    assert jnp.array_equal(
        diagnostics.semantic_mask,
        jnp.array([True, True, False]),
    )
    assert jnp.array_equal(
        diagnostics.dirichlet_mask,
        jnp.array([True, False, False]),
    )
    assert jnp.array_equal(
        diagnostics.categorical_mask,
        jnp.array([False, True, False]),
    )
    assert jnp.isfinite(diagnostics.semantic_kl).all()
    assert jnp.isfinite(diagnostics.dirichlet_kl).all()


def test_fixed_probe_q_weighting_matches_loss_weights_and_native_populations() -> None:
    prior_q = jnp.array([[[1.0, 1.0], [1.0, 3.0]]])
    target_q = jnp.array([[[2.0, 1.0], [0.0, 0.0]]])
    q_kind = jnp.array(
        [[TARGET_DIRICHLET, TARGET_CATEGORICAL]],
        dtype=jnp.int8,
    )
    q_outcome = jnp.array([[-1, 0]], dtype=jnp.int8)
    q_weight = jnp.array([[2.0, 8.0]])

    discrepancy = distillation_discrepancy(
        prior_logits=jnp.zeros((1, 2)),
        target_policy=jnp.array([[0.5, 0.5]]),
        legal_action_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        policy_row_mask=jnp.ones((1,), dtype=jnp.bool_),
        prior_alpha_v=jnp.ones((1, 2)),
        prior_alpha_q=prior_q,
        target_alpha_v=jnp.zeros((1, 2)),
        target_alpha_q=target_q,
        v_target_kind=jnp.array([TARGET_PAD], dtype=jnp.int8),
        v_target_outcome=jnp.array([-1], dtype=jnp.int8),
        q_target_kind=q_kind,
        q_target_outcome=q_outcome,
        v_mask=jnp.zeros((1,), dtype=jnp.bool_),
        q_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        q_sample_weight=q_weight,
    )
    native_q = native_displacement(
        prior_q,
        target_q,
        q_kind,
        q_outcome,
    )

    assert discrepancy.q_semantic_kl_count == 2
    assert jnp.allclose(
        discrepancy.q_semantic_kl_sum,
        jnp.sum(native_q.semantic_kl),
    )
    assert discrepancy.q_weighted_semantic_kl_weight == 10
    assert jnp.allclose(
        discrepancy.q_weighted_semantic_kl_sum,
        2.0 * native_q.semantic_kl[0, 0]
        + 8.0 * native_q.semantic_kl[0, 1],
    )
    # A categorical certificate belongs to the semantic population but never
    # to the full-Dirichlet population.
    assert discrepancy.q_dirichlet_kl_count == 1
    assert discrepancy.q_weighted_dirichlet_kl_weight == 2
    assert jnp.allclose(
        discrepancy.q_weighted_dirichlet_kl_sum,
        2.0 * native_q.dirichlet_kl[0, 0],
    )


def test_policy_displacement_masks_and_renormalizes_legal_target() -> None:
    target = jnp.array([[2.0, 0.0, 100.0]])
    logits = jnp.zeros((1, 3))
    legal = jnp.array([[True, True, False]])

    kl, valid = policy_displacement(target, logits, legal)

    assert bool(valid[0])
    assert jnp.allclose(kl[0], jnp.log(2.0))


def test_generation_diagnostics_exclude_nonfinite_native_displacements() -> None:
    prior_v = jnp.array([[1.0, 1.0]])
    prior_q = jnp.ones((1, 2, 2))
    root = RootFnOutput(
        prior_logits=jnp.zeros((1, 2)),
        value=prior_v,
        action_values=prior_q,
        embedding=jnp.zeros((1,), dtype=jnp.int32),
        terminal_outcome=jnp.asarray(
            [int(NO_OUTCOME)],
            dtype=jnp.int8,
        ),
        to_play=jnp.zeros((1,), dtype=jnp.int32),
    )
    legal = jnp.ones((1, 2), dtype=jnp.bool_)
    tree = instantiate_tree_from_root(root, 0, ~legal)

    diagnostics = root_search_diagnostics(
        prior_logits=root.prior_logits,
        prior_alpha_v=prior_v,
        prior_alpha_q=prior_q,
        target_policy=jnp.array([[1.0, 0.0]]),
        target_alpha_v=jnp.zeros_like(prior_v),
        target_alpha_q=jnp.zeros_like(prior_q),
        q_target_kind=jnp.full(
            (1, 2),
            TARGET_DIRICHLET,
            dtype=jnp.int8,
        ),
        q_target_outcome=jnp.full((1, 2), -1, dtype=jnp.int8),
        v_target_kind=jnp.asarray(
            [TARGET_DIRICHLET],
            dtype=jnp.int8,
        ),
        v_target_outcome=jnp.asarray([-1], dtype=jnp.int8),
        legal_action_mask=legal,
        tree=tree,
        summary=tree.summary(),
    )

    assert jnp.isfinite(diagnostics.search_v_semantic_kl_sum).all()
    assert jnp.isfinite(diagnostics.search_v_dirichlet_kl_sum).all()
    assert jnp.isfinite(diagnostics.search_q_semantic_kl_sum).all()
    assert jnp.isfinite(diagnostics.search_q_dirichlet_kl_sum).all()
    assert diagnostics.search_v_semantic_kl_count[0] == 0
    assert diagnostics.search_v_dirichlet_kl_count[0] == 0
    assert diagnostics.search_q_semantic_kl_count[0] == 0
    assert diagnostics.search_q_dirichlet_kl_count[0] == 0


def test_additive_diagnostics_pool_before_logging_ratios() -> None:
    shape = (2, 2)
    fields = {
        field: jnp.zeros(shape, dtype=jnp.float32)
        for field in SearchDiagnostics._fields
    }
    fields.update(
        search_policy_kl_sum=jnp.array([[1.0, 2.0], [2.0, 3.0]]),
        search_policy_kl_count=jnp.ones(shape),
        search_q_semantic_kl_sum=jnp.full(shape, 2.0),
        search_q_semantic_kl_count=jnp.full(shape, 5.0),
        search_q_policy_semantic_kl_sum=jnp.ones(shape),
        search_q_policy_semantic_kl_count=jnp.ones(shape),
        search_root_count=jnp.ones(shape),
        search_legal_action_count=jnp.array([[2.0, 3.0], [2.0, 3.0]]),
        search_visited_action_count=jnp.array([[1.0, 2.0], [1.0, 1.0]]),
        search_expanded_node_count=jnp.full(shape, 3.0),
        search_simulation_active_count=jnp.full(shape, 3.0),
        search_executed_simulation_row_count=jnp.full(shape, 4.0),
        search_requested_simulation_count=jnp.full(shape, 4.0),
        search_max_depth_sum=jnp.full(shape, 2.0),
        search_policy_support_sum=jnp.full(shape, 3.0),
        search_policy_ess_sum=jnp.full(shape, 2.0),
        search_policy_top1_agreement_count=jnp.array(
            [[1.0, 0.0], [1.0, 1.0]]
        ),
        search_root_policy_target_enabled_count=jnp.ones(shape),
        search_root_policy_target_categorical_population_count=jnp.array(
            [[1.0, 0.0], [0.0, 0.0]]
        ),
        search_root_policy_target_prefix_eligible_count=jnp.ones(shape),
        search_root_policy_target_prefix_accepted_count=jnp.array(
            [[1.0, 1.0], [1.0, 0.0]]
        ),
        search_root_policy_target_prefix_fallback_count=jnp.array(
            [[0.0, 0.0], [0.0, 1.0]]
        ),
        search_root_policy_target_prefix_tail_clipped_count=jnp.array(
            [[0.0, 0.0], [0.0, 1.0]]
        ),
        search_root_policy_target_prefix_density_abs_sum=jnp.full(
            shape,
            0.002,
        ),
        search_root_policy_target_native_l1_sum=jnp.full(shape, 0.1),
        search_root_policy_target_native_l2_sq_sum=jnp.full(
            shape,
            0.01,
        ),
        search_root_policy_target_native_top1_agreement_count=jnp.array(
            [[1.0, 1.0], [1.0, 0.0]]
        ),
        search_root_action_estimator_enabled_count=jnp.ones(shape),
        search_root_action_prefix_eligible_count=jnp.ones(shape),
        search_root_action_prefix_accepted_count=jnp.array(
            [[1.0, 1.0], [1.0, 0.0]]
        ),
        search_root_action_prefix_fallback_count=jnp.array(
            [[0.0, 0.0], [0.0, 1.0]]
        ),
        search_root_action_prefix_tail_clipped_count=jnp.array(
            [[0.0, 0.0], [0.0, 1.0]]
        ),
        search_root_action_prefix_density_abs_sum=jnp.full(
            shape,
            0.004,
        ),
        search_root_action_native_l1_sum=jnp.full(shape, 0.2),
        search_root_action_native_l2_sq_sum=jnp.full(shape, 0.02),
        search_root_action_native_top1_agreement_count=jnp.array(
            [[1.0, 0.0], [1.0, 0.0]]
        ),
        search_solved_root_count=jnp.array(
            [[1.0, 0.0], [0.0, 0.0]]
        ),
        search_kappa_numeric_repair_count=jnp.full(shape, 2.0),
        search_kappa_raw_innovation_l2_sum=jnp.full(shape, 4.0),
        search_kappa_semantic_innovation_l2_sum=jnp.full(shape, 2.0),
        search_kappa_concentration_innovation_abs_sum=jnp.full(
            shape,
            6.0,
        ),
        search_kappa_raw_dcache_dlogkappa_l2_sum=jnp.ones(shape),
        search_kappa_mean_dcache_dlogkappa_l2_sum=jnp.full(
            shape,
            0.5,
        ),
        search_kappa_log_concentration_dcache_dlogkappa_abs_sum=(
            jnp.full(shape, 0.25)
        ),
        search_kappa_numeric_path_count=jnp.ones(shape),
        search_kappa_path_gamma_product_sum=jnp.full(shape, 0.25),
        search_kappa_path_gamma_log_attenuation_sum=jnp.full(
            shape,
            0.5,
        ),
        search_kappa_categorical_publication_path_count=jnp.array(
            [[1.0, 0.0], [0.0, 1.0]]
        ),
        search_root_policy_top2_margin_sum=jnp.array(
            [[0.1, 0.2], [0.3, 0.4]]
        ),
        search_root_policy_top2_margin_count=jnp.ones(shape),
        search_root_policy_top2_margin_tie_count=jnp.array(
            [[1.0, 0.0], [0.0, 0.0]]
        ),
        search_root_policy_top2_margin_below_reference_count=jnp.array(
            [[1.0, 1.0], [0.0, 0.0]]
        ),
        search_root_policy_top2_margin_reference_scale_sum=jnp.full(
            shape,
            0.125,
        ),
        search_root_plurality_commitment_count=jnp.array(
            [[1.0, 1.0], [1.0, 0.0]]
        ),
        search_root_plurality_max_count_tie_count=jnp.array(
            [[1.0, 0.0], [1.0, 0.0]]
        ),
        search_root_plurality_tie_multiplicity_sum=jnp.array(
            [[2.0, 0.0], [3.0, 0.0]]
        ),
        search_root_plurality_lowest_uniform_disagreement_count=jnp.array(
            [[1.0, 0.0], [1.0, 0.0]]
        ),
        search_root_plurality_expected_disagreement_sum=jnp.array(
            [[0.5, 0.0], [2.0 / 3.0, 0.0]]
        ),
    )
    diagnostics = SearchDiagnostics(**fields)
    base_metrics = _zero_train_metrics_like(jnp.asarray(0.0))._replace(
        data_frame_count=jnp.asarray(100.0),
        data_termination_count=jnp.asarray(4.0),
    )

    pooled = _with_search_diagnostics(base_metrics, diagnostics)
    logged = training_metrics(
        pooled,
        seconds=1.0,
        hours=0.0,
        frames=4,
        frames_this_iteration=4,
        optimizer_updates=12,
        optimizer_updates_this_iteration=3,
        completed_iterations=4,
    )

    assert jnp.allclose(pooled.search_policy_kl_sum, 8.0)
    assert jnp.allclose(pooled.search_policy_kl_count, 4.0)
    assert logged["search/policy_displacement_kl_nats"] == 2.0
    assert logged["search/root_count"] == 4.0
    assert logged["search/root_action_coverage"] == 0.5
    assert logged["search/expanded_node_fraction_of_requested"] == 0.75
    assert logged["search/useful_recurrent_rows_total"] == 12.0
    assert logged["search/executed_recurrent_rows_total"] == 16.0
    assert logged["search/recurrent_row_utilization"] == 0.75
    assert logged["search/max_depth_mean"] == 2.0
    assert logged["search/root_plurality_commitment_count"] == 3.0
    assert logged["search/root_plurality_max_count_tie_count"] == 2.0
    assert (
        logged["search/root_plurality_max_count_tie_fraction"]
        == pytest.approx(2.0 / 3.0)
    )
    assert (
        logged["search/root_plurality_tied_max_multiplicity_mean"] == 2.5
    )
    assert (
        logged["search/root_plurality_lowest_uniform_disagreement_fraction"]
        == pytest.approx(2.0 / 3.0)
    )
    assert (
        logged[
            "search/root_plurality_lowest_uniform_disagreement_given_tie"
        ]
        == 1.0
    )
    assert (
        logged["search/root_plurality_expected_disagreement_fraction"]
        == pytest.approx(7.0 / 18.0)
    )
    assert logged["search/q_semantic_displacement_kl_nats"] == 0.4
    assert (
        logged[
            "search/q_semantic_displacement_total_per_root_nats"
        ]
        == 2.0
    )
    assert (
        logged[
            "search/q_policy_weighted_semantic_displacement_kl_nats"
        ]
        == 1.0
    )
    assert logged["search/policy_support_mean"] == 3.0
    assert logged["search/policy_ess_mean"] == 2.0
    assert logged["search/policy_prior_target_top1_agreement"] == 0.75
    assert (
        logged["search/root_policy_target_categorical_population_fraction"]
        == 0.25
    )
    assert (
        logged["search/root_policy_target_prefix_acceptance_fraction"]
        == 0.75
    )
    assert (
        logged["search/root_policy_target_prefix_fallback_fraction"]
        == 0.25
    )
    assert (
        logged["search/root_policy_target_prefix_tail_clipped_fraction"]
        == 0.25
    )
    assert jnp.allclose(
        cast(
            float,
            logged["search/root_policy_target_prefix_density_abs_mean"],
        ),
        0.002,
    )
    assert jnp.allclose(
        cast(float, logged["search/root_policy_target_vs_native_m32_l1"]),
        0.1,
    )
    assert (
        logged[
            "search/root_policy_target_vs_native_m32_top1_agreement"
        ]
        == 0.75
    )
    assert logged["search/root_action_prefix_acceptance_fraction"] == 0.75
    assert logged["search/root_action_prefix_fallback_fraction"] == 0.25
    assert (
        logged["search/root_action_prefix_tail_clipped_fraction"] == 0.25
    )
    assert jnp.allclose(
        cast(float, logged["search/root_action_prefix_density_abs_mean"]),
        0.004,
    )
    assert jnp.allclose(
        cast(float, logged["search/root_action_vs_native_m32_l1"]),
        0.2,
    )
    assert (
        logged["search/root_action_vs_native_m32_top1_agreement"] == 0.5
    )
    assert logged["search/kappa_solved_bypass_fraction"] == 0.25
    assert logged["search/kappa_numeric_repair_count"] == 8.0
    assert logged["search/kappa_raw_innovation_l2_mean"] == 2.0
    assert logged["search/kappa_semantic_innovation_l2_mean"] == 1.0
    assert logged["search/kappa_concentration_innovation_abs_mean"] == 3.0
    assert logged["search/kappa_dcache_dlogkappa_raw_l2_mean"] == 0.5
    assert logged["search/kappa_dmean_dlogkappa_l2_mean"] == 0.25
    assert (
        logged[
            "search/kappa_dlog_concentration_dlogkappa_abs_mean"
        ]
        == 0.125
    )
    assert logged["search/kappa_numeric_path_count"] == 4.0
    assert (
        logged["search/kappa_numeric_path_gamma_product_mean"] == 0.25
    )
    assert (
        logged[
            "search/kappa_numeric_path_gamma_log_attenuation_mean"
        ]
        == 0.5
    )
    assert logged["search/kappa_categorical_publication_path_count"] == 2.0
    assert jnp.allclose(
        cast(
            float,
            logged["search/kappa_categorical_publication_path_fraction"],
        ),
        1.0 / 6.0,
    )
    assert jnp.allclose(
        cast(float, logged["search/root_policy_top2_margin_mean"]),
        0.25,
    )
    assert logged["search/root_policy_top2_margin_tie_fraction"] == 0.25
    assert (
        logged["search/root_policy_top2_margin_below_reference_fraction"]
        == 0.5
    )
    assert (
        logged["search/root_policy_top2_margin_reference_scale"] == 0.125
    )
    assert logged["search/root_policy_top2_margin_count"] == 4.0
    assert logged["data/terminal_events_per_1k_frames"] == 40.0
    assert logged["train/optimizer_updates"] == 12
    assert logged["train/optimizer_updates_this_iteration"] == 3
    assert logged["train/completed_iterations"] == 4


def test_optimizer_update_budget_matches_minibatch_construction() -> None:
    config = Config(
        selfplay=SelfplayConfig(batch_size=8, max_num_steps=6),
        training=TrainingConfig(
            batch_size=4,
            max_updates_per_iter=5,
        ),
    )

    assert optimizer_updates_per_iteration(config) == 5


def test_fixed_probe_capture_logs_raw_gaps_delta_and_fraction() -> None:
    zero = jnp.asarray(0.0)
    before = DistillationDiscrepancy(
        policy_kl_sum=jnp.asarray(10.0),
        policy_kl_count=jnp.asarray(5.0),
        v_semantic_kl_sum=zero,
        v_semantic_kl_count=zero,
        v_dirichlet_kl_sum=zero,
        v_dirichlet_kl_count=zero,
        q_semantic_kl_sum=zero,
        q_semantic_kl_count=zero,
        q_dirichlet_kl_sum=zero,
        q_dirichlet_kl_count=zero,
        q_weighted_semantic_kl_sum=jnp.asarray(18.0),
        q_weighted_semantic_kl_weight=jnp.asarray(6.0),
        q_weighted_dirichlet_kl_sum=jnp.asarray(10.0),
        q_weighted_dirichlet_kl_weight=jnp.asarray(5.0),
    )
    after = before._replace(
        policy_kl_sum=jnp.asarray(4.0),
        q_weighted_semantic_kl_sum=jnp.asarray(6.0),
        q_weighted_dirichlet_kl_sum=jnp.asarray(5.0),
    )
    metrics = _with_capture_diagnostics(
        _zero_train_metrics_like(zero),
        before,
        after,
    )

    logged = training_metrics(
        metrics,
        seconds=1.0,
        hours=0.0,
        frames=1,
        frames_this_iteration=1,
    )

    assert logged["capture/train_probe/policy_gap_before_nats"] == 2.0
    assert logged["capture/train_probe/policy_gap_after_nats"] == 0.8
    assert logged["capture/train_probe/policy_gap_delta_nats"] == 1.2
    policy_fraction = logged["capture/train_probe/policy_fraction"]
    assert isinstance(policy_fraction, (int, float))
    assert jnp.allclose(policy_fraction, 0.6)
    assert logged["capture/train_probe/policy_fraction_defined"] == 1
    assert (
        logged[
            "capture/train_probe/q_loss_weighted_semantic_gap_before_nats"
        ]
        == 3.0
    )
    assert (
        logged[
            "capture/train_probe/q_loss_weighted_semantic_gap_after_nats"
        ]
        == 1.0
    )
    q_weighted_fraction = logged[
        "capture/train_probe/q_loss_weighted_semantic_fraction"
    ]
    assert isinstance(q_weighted_fraction, (int, float))
    assert jnp.allclose(q_weighted_fraction, 2.0 / 3.0)
    assert (
        logged[
            "capture/train_probe/q_loss_weighted_semantic_total_weight_before"
        ]
        == 6.0
    )
    assert (
        logged[
            "capture/train_probe/q_loss_weighted_full_dirichlet_gap_before_nats"
        ]
        == 2.0
    )
    assert (
        logged[
            "capture/train_probe/q_loss_weighted_full_dirichlet_fraction"
        ]
        == 0.5
    )


def test_fixed_probe_discrepancy_decreases_after_targeted_gradient_step() -> None:
    model = _TinyScalarProbeModel()
    config = Config(
        training=TrainingConfig(
            batch_size=1,
            learning_rate=0.25,
        )
    )
    sample = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[0.0, 1.0]]),
        value_tgt=jnp.zeros((1,)),
        played_action=jnp.ones((1,), dtype=jnp.int32),
        policy_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        value_mask=jnp.ones((1,), dtype=jnp.bool_),
        beta_Q_target=jnp.ones((1, 2, 2)),
        beta_V_target=jnp.ones((1, 2)),
        q_loss_weight=jnp.zeros((1, 2)),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.sgd(0.25),
        wrt=nnx.Param,
    )

    before = evaluate_distillation_discrepancy(model, sample, config)
    train(model, optimizer, sample, config)
    after = evaluate_distillation_discrepancy(model, sample, config)

    assert before.policy_kl_count == after.policy_kl_count == 1
    assert before.policy_kl_sum > after.policy_kl_sum
    assert (before.policy_kl_sum - after.policy_kl_sum) / (
        before.policy_kl_sum
    ) > 0


def test_training_iteration_measures_same_probe_before_and_after_updates() -> None:
    env = pgx.make("tic_tac_toe")
    config = Config(
        env=EnvConfig(id="tic_tac_toe", num_outcomes=3),
        model=ModelConfig(
            network=Network.boardlaw_dirichlet,
            num_channels=4,
            num_layers=1,
        ),
        selfplay=SelfplayConfig(
            batch_size=2,
            max_num_steps=1,
            search=SearchConfig(
                kind=SearchKind.dirichlet_thompson,
                dirichlet_thompson=DirichletThompsonSearchConfig(
                    num_simulations=0,
                    policy_samples=1,
                ),
            ),
        ),
        training=TrainingConfig(
            batch_size=2,
            max_updates_per_iter=1,
            learning_rate=1e-2,
            losses=TrainingLossConfig(
                value_dir_kl_weight=0.1,
                q_dir_kl_weight=0.1,
                # With zero simulations, evidence-mass weighting correctly
                # has no Q population.  This test needs a nonempty Q probe to
                # exercise before/after capture, so opt into the public-policy
                # population explicitly.
                q_loss_weight_mode=QLossWeightMode.policy,
            ),
        ),
    )
    model = build_model(
        config,
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        rngs=nnx.Rngs(0),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adam(config.training.learning_rate),
        wrt=nnx.Param,
    )

    metrics = make_training_iteration(env, config)(
        model,
        optimizer,
        jax.random.PRNGKey(0),
    )

    assert metrics.policy_loss.shape == (1,)
    assert metrics.search_root_count == 2
    assert metrics.capture_policy_before_count == 2
    assert metrics.capture_policy_after_count == 2
    assert metrics.capture_v_dirichlet_before_count == 2
    assert metrics.capture_v_dirichlet_after_count == 2
    assert metrics.capture_q_dirichlet_before_count == 2
    assert metrics.capture_q_dirichlet_after_count == 2
    assert metrics.capture_q_weighted_dirichlet_before_weight > 0
    assert (
        metrics.capture_q_weighted_dirichlet_before_weight
        == metrics.capture_q_weighted_dirichlet_after_weight
    )
    assert jnp.isfinite(metrics.capture_policy_before_sum)
    assert jnp.isfinite(metrics.capture_policy_after_sum)
