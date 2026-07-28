from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import pytest

from scacchi import checkpoint
from scacchi.dirichlet_mctx.native_targets import (
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    TARGET_PAD,
)
from scacchi.dirichlet_q_search import build_q_supervision
from scacchi.logger import training_metrics
from scacchi.loss import (
    Sample,
    _compute_dirichlet_losses,
    _native_dirichlet_loss,
    _native_masked_mean,
)
from scacchi.types import Config, config_to_dict, load_config


def _mixed_targets():
    beta = jnp.array(
        [
            [[2.0, 3.0], [1.0, 1.0], [4.0, 2.0]],
            [[3.0, 2.0], [2.0, 5.0], [1.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    target_kind = jnp.array(
        [
            [TARGET_DIRICHLET, TARGET_CATEGORICAL, TARGET_PAD],
            [TARGET_DIRICHLET, TARGET_DIRICHLET, TARGET_CATEGORICAL],
        ],
        dtype=jnp.int8,
    )
    target_outcome = jnp.array(
        [[-1, 1, -1], [-1, -1, 0]],
        dtype=jnp.int8,
    )
    legal = target_kind != int(TARGET_PAD)
    solved = target_kind == int(TARGET_CATEGORICAL)
    target_weight = legal.astype(jnp.float32)
    evidence = jnp.array(
        [[[2.0], [0.0], [0.0]], [[3.0], [0.0], [0.0]]],
        dtype=jnp.float32,
    )
    policy = jnp.array(
        [[0.2, 0.8, 0.0], [0.7, 0.0, 0.3]],
        dtype=jnp.float32,
    )
    alpha = jnp.array(
        [
            [[3.0, 2.0], [2.0, 4.0], [1.0, 1.0]],
            [[4.0, 2.0], [3.0, 4.0], [5.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    return (
        beta,
        target_kind,
        target_outcome,
        target_weight,
        evidence,
        policy,
        solved,
        legal,
        alpha,
    )


def _pair_losses(alpha):
    beta, kind, outcome, weight, *_ = _mixed_targets()
    return _native_dirichlet_loss(
        beta,
        alpha,
        kind,
        outcome,
        weight,
        categorical_epsilon=1e-4,
        loss_mode="full",
    )


def _old_evidence_masked_mean(alpha, evidence):
    _, kind, _, _, _, policy, solved, legal, _ = _mixed_targets()
    old_weight = jnp.sum(evidence, axis=-1) + jnp.zeros_like(policy)
    old_weight = jnp.where(
        legal & solved,
        jnp.maximum(old_weight, 1.0),
        old_weight,
    )
    return _native_masked_mean(
        _pair_losses(alpha),
        old_weight > 0,
        kind,
    )


def _new_uniform_evidence_mean(alpha, evidence):
    _, kind, _, _, _, policy, solved, legal, _ = _mixed_targets()
    supervision = build_q_supervision(
        "positive_search_evidence_or_solved",
        "mean_over_selected_state_action_pairs",
        evidence,
        policy,
        solved,
        legal,
    )
    return _native_masked_mean(
        _pair_losses(alpha),
        supervision.selected,
        kind,
    )


def test_default_loss_and_gradient_match_old_evidence_masked_mean():
    *_, evidence, _, _, _, alpha = _mixed_targets()

    old_value, old_grad = jax.value_and_grad(
        lambda candidate: _old_evidence_masked_mean(candidate, evidence)
    )(alpha)
    new_value, new_grad = jax.value_and_grad(
        lambda candidate: _new_uniform_evidence_mean(candidate, evidence)
    )(alpha)

    assert jnp.array_equal(new_value, old_value)
    assert jnp.array_equal(new_grad, old_grad)


def test_default_loss_and_gradient_are_invariant_to_evidence_scale():
    *_, evidence, _, _, _, alpha = _mixed_targets()

    value, grad = jax.value_and_grad(
        lambda candidate: _new_uniform_evidence_mean(candidate, evidence)
    )(alpha)
    scaled_value, scaled_grad = jax.value_and_grad(
        lambda candidate: _new_uniform_evidence_mean(
            candidate,
            100.0 * evidence,
        )
    )(alpha)

    assert jnp.array_equal(scaled_value, value)
    assert jnp.array_equal(scaled_grad, grad)


def test_zero_to_positive_evidence_adds_exactly_one_selected_pair():
    _, _, _, _, evidence, policy, solved, legal, _ = _mixed_targets()
    before = build_q_supervision(
        "positive_search_evidence_or_solved",
        "mean_over_selected_state_action_pairs",
        evidence,
        policy,
        solved,
        legal,
    )
    after_evidence = evidence.at[1, 1, 0].set(1.0)
    after = build_q_supervision(
        "positive_search_evidence_or_solved",
        "mean_over_selected_state_action_pairs",
        after_evidence,
        policy,
        solved,
        legal,
    )

    assert int(jnp.sum(after.selected)) == int(jnp.sum(before.selected)) + 1
    assert bool(after.selected[1, 1])


def test_solved_action_is_selected_without_policy_or_unresolved_evidence():
    _, _, _, _, evidence, policy, solved, legal, _ = _mixed_targets()
    supervision = build_q_supervision(
        "positive_posterior_policy_or_solved",
        "mean_over_selected_state_action_pairs",
        evidence,
        policy.at[0, 1].set(0.0),
        solved,
        legal,
    )

    assert bool(supervision.selected[0, 1])
    assert supervision.pair_weight[0, 1] == 1.0


def test_uniform_denominator_counts_selected_state_action_pairs():
    losses = jnp.array([[1.0, 3.0, 100.0], [5.0, 100.0, 7.0]])
    _, _, _, _, evidence, policy, solved, legal, _ = _mixed_targets()
    supervision = build_q_supervision(
        "positive_search_evidence_or_solved",
        "mean_over_selected_state_action_pairs",
        evidence,
        policy,
        solved,
        legal,
    )

    reduced = jnp.sum(
        jnp.where(supervision.selected, losses, 0.0)
    ) / jnp.sum(supervision.selected)

    assert int(jnp.sum(supervision.selected)) == 4
    assert reduced == pytest.approx((1.0 + 3.0 + 5.0 + 7.0) / 4.0)


def test_policy_action_set_excludes_zero_policy_unless_solved():
    _, _, _, _, evidence, policy, solved, legal, _ = _mixed_targets()
    supervision = build_q_supervision(
        "positive_posterior_policy_or_solved",
        "mean_over_selected_state_action_pairs",
        evidence,
        policy,
        solved,
        legal,
    )

    assert not bool(supervision.selected[1, 1])
    assert bool(supervision.selected[1, 2])


def test_q_population_metrics_use_direct_semantic_counts():
    (
        beta,
        kind,
        outcome,
        target_weight,
        evidence,
        policy,
        solved,
        legal,
        alpha,
    ) = _mixed_targets()
    supervision = build_q_supervision(
        "positive_search_evidence_or_solved",
        "mean_over_selected_state_action_pairs",
        evidence,
        policy,
        solved,
        legal,
    )
    data = Sample(
        obs=jnp.zeros((2, 1)),
        policy_tgt=policy,
        value_tgt=jnp.zeros((2,)),
        played_action=jnp.zeros((2,), dtype=jnp.int32),
        policy_mask=legal,
        value_mask=jnp.ones((2,), dtype=jnp.bool_),
        beta_Q_target=beta,
        beta_V_target=jnp.ones((2, 2)),
        q_pair_weight=supervision.pair_weight,
        q_supervised_pair_mask=supervision.selected,
        q_positive_evidence_action_mask=legal
        & (jnp.sum(evidence, axis=-1) > 0),
        q_positive_policy_action_mask=legal & (policy > 0),
        q_target_kind=kind,
        q_target_weight=target_weight,
        q_target_outcome=outcome,
        q_target_distance=jnp.zeros_like(kind, dtype=jnp.int32),
    )

    _, metrics = _compute_dirichlet_losses(
        jnp.zeros_like(policy),
        jnp.ones((2, 2)),
        alpha,
        data,
        Config(),
    )

    assert metrics.q_positive_evidence_action_count == 2
    assert metrics.q_positive_policy_action_count == 4
    assert metrics.q_solved_action_count == 2
    assert metrics.q_supervised_action_count == 4
    assert metrics.q_supervised_actions_per_row == 2
    assert metrics.q_supervised_action_fraction == pytest.approx(4.0 / 5.0)

    logged = training_metrics(
        metrics,
        seconds=1.0,
        hours=0.0,
        frames=2,
        frames_this_iteration=2,
    )
    assert logged["train/q_supervised_actions_per_row"] == 2
    assert logged["data/q_positive_evidence_action_count"] == 2
    assert logged["data/q_positive_policy_action_count"] == 4
    assert logged["data/q_solved_action_count"] == 2
    assert logged["data/q_supervised_action_count"] == 4
    assert logged["data/q_supervised_action_fraction"] == pytest.approx(
        4.0 / 5.0
    )


@pytest.mark.parametrize(
    "action_set",
    [
        "positive_search_evidence_or_solved",
        "positive_posterior_policy_or_solved",
    ],
)
def test_legacy_weighted_loss_and_gradient_match_old_formula(action_set):
    _, _, _, _, evidence, policy, solved, legal, alpha = _mixed_targets()
    source = (
        jnp.sum(evidence, axis=-1)
        if action_set == "positive_search_evidence_or_solved"
        else policy
    )
    old_weight = jnp.where(
        legal & solved,
        jnp.maximum(source, 1.0),
        source,
    )
    old_weight = jnp.where(legal & (old_weight > 0), old_weight, 0.0)

    def old_loss(candidate):
        pair_loss = _pair_losses(candidate)
        return jnp.sum(old_weight * pair_loss) / jnp.sum(old_weight)

    supervision = build_q_supervision(
        action_set,
        "legacy_normalized_source_weighted_mean",
        evidence,
        policy,
        solved,
        legal,
    )

    def new_loss(candidate):
        pair_loss = _pair_losses(candidate)
        return (
            jnp.sum(supervision.pair_weight * pair_loss)
            / jnp.sum(supervision.pair_weight)
        )

    old_value, old_grad = jax.value_and_grad(old_loss)(alpha)
    new_value, new_grad = jax.value_and_grad(new_loss)(alpha)
    assert jnp.array_equal(new_value, old_value)
    assert jnp.array_equal(new_grad, old_grad)


@pytest.mark.parametrize(
    ("old_action_source", "old_reduction", "action_set", "reduction"),
    [
        (
            "evidence_mass",
            "masked_mean",
            "positive_search_evidence_or_solved",
            "mean_over_selected_state_action_pairs",
        ),
        (
            "policy",
            "masked_mean",
            "positive_posterior_policy_or_solved",
            "mean_over_selected_state_action_pairs",
        ),
        (
            "evidence_mass",
            "weighted",
            "positive_search_evidence_or_solved",
            "legacy_normalized_source_weighted_mean",
        ),
        (
            "policy",
            "weighted",
            "positive_posterior_policy_or_solved",
            "legacy_normalized_source_weighted_mean",
        ),
    ],
)
def test_legacy_config_migration(
    old_action_source,
    old_reduction,
    action_set,
    reduction,
):
    with pytest.warns(FutureWarning, match="deprecated"):
        config = load_config(
            OmegaConf.create(
                {
                    "training": {
                        "losses": {
                            "q_loss_weight_mode": old_action_source,
                            "q_dir_kl_reduction": old_reduction,
                        }
                    }
                }
            )
        )

    assert config.training.losses.q_supervision.action_set == action_set
    assert config.training.losses.q_supervision.reduction == reduction


def test_mixing_old_and_new_q_supervision_fields_fails_clearly():
    with pytest.raises(ValueError, match="cannot mix q_supervision"):
        load_config(
            OmegaConf.create(
                {
                    "training": {
                        "losses": {
                            "q_loss_weight_mode": "evidence_mass",
                            "q_supervision": {
                                "action_set": (
                                    "positive_search_evidence_or_solved"
                                ),
                            },
                        }
                    }
                }
            )
        )


def test_new_config_serialization_contains_only_q_supervision_schema():
    serialized = config_to_dict(Config())
    losses = serialized["training"]["losses"]

    assert losses["q_supervision"] == {
        "action_set": "positive_search_evidence_or_solved",
        "reduction": "mean_over_selected_state_action_pairs",
    }
    assert "q_loss_weight_mode" not in losses
    assert "q_dir_kl_reduction" not in losses


def test_nested_and_flat_checkpoint_metadata_migrate_q_supervision():
    with pytest.warns(FutureWarning, match="deprecated"):
        nested = checkpoint._load_checkpoint_config(
            {
                "env": {"id": "hex"},
                "training": {
                    "losses": {
                        "q_loss_weight_mode": "evidence_mass",
                        "q_dir_kl_reduction": "masked_mean",
                    }
                },
            }
        )
    with pytest.warns(FutureWarning, match="deprecated"):
        flat = checkpoint._load_checkpoint_config(
            {
                "env_id": "hex",
                "q_loss_weight_mode": "policy",
                "q_dir_kl_reduction": "weighted",
            }
        )

    assert (
        nested.training.losses.q_supervision.reduction
        == "mean_over_selected_state_action_pairs"
    )
    assert (
        flat.training.losses.q_supervision.action_set
        == "positive_posterior_policy_or_solved"
    )
    assert (
        flat.training.losses.q_supervision.reduction
        == "legacy_normalized_source_weighted_mean"
    )


def test_active_configs_use_only_new_q_supervision_schema():
    config_dir = Path(__file__).parents[1] / "scacchi" / "configs"
    for path in config_dir.glob("*.yaml"):
        text = path.read_text()
        assert "q_loss_weight_mode:" not in text
        assert "q_dir_kl_reduction:" not in text
