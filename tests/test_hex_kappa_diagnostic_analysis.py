from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np
import pytest

from scripts.hex_kappa_diagnostic_analysis import (
    CHANNEL_ACTIVE_SCOPE,
    DEFAULT_LOCAL_LOG_STEP,
    DEPLOYMENT_SCOPE,
    PROBE_FORMAT,
    _average_ranks,
    _cluster_bootstrap,
    _extract_probe_arrays,
    _kappa_key,
    _margin_flip_summary,
    _spearman,
    analyze,
)


REFERENCE_KAPPA = 3.0
LOCAL_MINUS_KAPPA = REFERENCE_KAPPA * math.exp(
    -DEFAULT_LOCAL_LOG_STEP
)
LOCAL_PLUS_KAPPA = REFERENCE_KAPPA * math.exp(
    DEFAULT_LOCAL_LOG_STEP
)
KAPPAS = (
    LOCAL_MINUS_KAPPA,
    REFERENCE_KAPPA,
    LOCAL_PLUS_KAPPA,
    8.0,
    64.0,
)


def _channel(
    *,
    normalized_derivative: float,
    raw_innovation: float,
    raw_derivative: float,
    margin: float | None,
) -> dict[str, Any]:
    return {
        "numeric_repairs": {
            "count": 2,
            "raw_innovation_l2_mean": raw_innovation,
            "raw_dcache_dlogkappa_l2_mean": raw_derivative,
            "mean_dcache_dlogkappa_l2_mean": normalized_derivative,
        },
        "commitment_policy_top2": {
            "margin": margin,
            "reference_scale": (
                1.0 / 32.0 if margin is not None else None
            ),
        },
    }


def _synthetic_probe() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    positions: list[dict[str, Any]] = []
    root_ids = np.arange(30, dtype=np.int32)
    stage_ids = np.arange(30, dtype=np.int8) % 3
    rank_within_stage = np.arange(30) // 3
    cluster_ids = rank_within_stage.astype(np.int32)
    root_weights = (1.0 + rank_within_stage).astype(np.float64)
    digests = np.asarray(
        [f"{root_id:064x}".encode("ascii") for root_id in root_ids],
        dtype="S64",
    )
    margins_by_rank = np.asarray(
        [0.0, 0.001, 0.006, 0.02, 0.04, 0.055, 0.0625, 0.08, 0.12, 0.2]
    )

    for root_id, stage_id, rank, root_weight, digest in zip(
        root_ids,
        stage_ids,
        rank_within_stage,
        root_weights,
        digests,
        strict=True,
    ):
        solved_root = int(rank) == 9
        margin = (
            None if solved_root else float(margins_by_rank[rank])
        )
        policy_margin = float(margins_by_rank[rank])
        reference = np.asarray(
            [
                0.5 + 0.5 * policy_margin,
                0.5 - 0.5 * policy_margin,
            ],
            dtype=np.float64,
        )
        response = 0.02 + 0.03 * float(rank)
        direction = 1.0 if int(root_id) % 2 == 0 else -1.0
        half_shift = 0.5 * DEFAULT_LOCAL_LOG_STEP * response
        shift = direction * np.asarray([half_shift, -half_shift])
        local_minus = reference - shift
        local_plus = reference + shift

        candidate_8 = reference.copy()
        if rank <= 4:
            candidate_8 = reference[::-1].copy()
            if margin == 0.0:
                candidate_8 = np.asarray([0.49, 0.51])
        candidate_64 = reference.copy()
        if rank <= 7:
            candidate_64 = reference[::-1].copy()
            if margin == 0.0:
                candidate_64 = np.asarray([0.48, 0.52])

        policies = {
            LOCAL_MINUS_KAPPA: local_minus,
            REFERENCE_KAPPA: reference,
            LOCAL_PLUS_KAPPA: local_plus,
            8.0: candidate_8,
            64.0: candidate_64,
        }
        if solved_root:
            policies = {
                kappa: reference.copy()
                for kappa in policies
            }
        by_kappa: dict[str, Any] = {}
        for kappa, policy in policies.items():
            raw_innovation = 0.4 + 0.01 * float(rank)
            raw_derivative = 0.2 * raw_innovation
            if kappa == 64.0 and int(root_id) == 5:
                raw_derivative = 0.5
            by_kappa[_kappa_key(kappa)] = {
                "root_policy": policy.tolist(),
                "solved_root": solved_root,
                "kappa_channel": _channel(
                    normalized_derivative=response,
                    raw_innovation=raw_innovation,
                    raw_derivative=raw_derivative,
                    margin=margin,
                ),
            }
        positions.append(
            {
                "root_id": int(root_id),
                "stage_id": int(stage_id),
                "corpus_root_weight": float(root_weight),
                "state_sha256": digest.decode("ascii"),
                "by_kappa": by_kappa,
            }
        )

    payload = {
        "format": PROBE_FORMAT,
        "protocol": {"reference_kappa": REFERENCE_KAPPA},
        "execution": {"kappas": list(KAPPAS)},
        "positions": positions,
    }
    roots = {
        "root_id": root_ids,
        "stage_id": stage_ids,
        "game_cluster_id": cluster_ids,
        "root_weight": root_weights,
        "state_sha256": digests,
    }
    return payload, roots


def test_average_ranks_and_spearman_use_average_tie_ranks():
    values = np.asarray([10.0, 20.0, 20.0, 40.0])

    np.testing.assert_array_equal(
        _average_ranks(values),
        np.asarray([1.0, 2.5, 2.5, 4.0]),
    )
    assert _spearman(values, values) == pytest.approx(1.0)
    assert _spearman(values, -values) == pytest.approx(-1.0)
    assert _spearman(values, np.ones(4)) is None


def test_game_cluster_bootstrap_is_reproducible_and_handles_no_roots():
    derivative = np.asarray([1.0, 2.0, 3.0, 4.0])
    response = derivative.copy()
    clusters = np.asarray([0, 0, 1, 1])

    first = _cluster_bootstrap(
        derivative=derivative,
        response=response,
        cluster_ids=clusters,
        replicates=200,
        rng=np.random.default_rng(17),
    )
    second = _cluster_bootstrap(
        derivative=derivative,
        response=response,
        cluster_ids=clusters,
        replicates=200,
        rng=np.random.default_rng(17),
    )

    assert first == second
    assert first["game_clusters"] == 2
    assert first["valid_spearman_replicates"] > 0
    assert first["spearman_95"] == pytest.approx([1.0, 1.0])

    empty = _cluster_bootstrap(
        derivative=np.asarray([]),
        response=np.asarray([]),
        cluster_ids=np.asarray([]),
        replicates=200,
        rng=np.random.default_rng(17),
    )
    assert empty["game_clusters"] == 0
    assert empty["spearman_95"] is None
    assert empty["mean_response_95"] is None


def test_analysis_covers_response_bounds_deciles_and_margin_flips():
    payload, roots = _synthetic_probe()

    result = analyze(
        payload=payload,
        roots=roots,
        bootstrap_replicates=200,
        bootstrap_seed=29,
    )

    deployment = result["central_local_response"][DEPLOYMENT_SCOPE]
    channel_active = result["central_local_response"][
        CHANNEL_ACTIVE_SCOPE
    ]
    overall = deployment["overall"]
    active_overall = channel_active["overall"]
    assert overall["numeric_derivative_eligible_roots"] == 30
    assert active_overall["numeric_derivative_eligible_roots"] == 27
    assert active_overall[
        "spearman_derivative_vs_policy_l1_response"
    ] == pytest.approx(1.0)
    assert overall[
        "spearman_derivative_vs_policy_l1_response"
    ] < active_overall[
        "spearman_derivative_vs_policy_l1_response"
    ]
    assert overall["game_cluster_bootstrap"]["game_clusters"] == 10
    assert active_overall["game_cluster_bootstrap"]["game_clusters"] == 9
    assert active_overall["game_cluster_bootstrap"]["spearman_95"] == (
        pytest.approx([1.0, 1.0])
    )
    for stage in ("early", "mid", "late"):
        assert channel_active[stage][
            "spearman_derivative_vs_policy_l1_response"
        ] == pytest.approx(1.0)
    assert result["reference_root_status"] == {
        "solved_bypass_roots": 3,
        "unresolved_numeric_channel_roots": 27,
        "by_stage": {
            stage: {
                "roots": 10,
                "solved_bypass_roots": 1,
                "unresolved_numeric_channel_roots": 9,
            }
            for stage in ("early", "mid", "late")
        },
    }

    bound_audit = result["raw_derivative_bound_audit"]
    assert bound_audit[_kappa_key(64.0)]["overall"][
        "violation_root_ids"
    ] == [5]
    for kappa in KAPPAS[:-1]:
        assert bound_audit[_kappa_key(kappa)]["overall"][
            "violation_count"
        ] == 0

    deciles = result["reference_normalized_derivative_deciles"]
    assert len(deciles[DEPLOYMENT_SCOPE]["overall"]) == 10
    assert all(
        bucket["roots"] == 3
        for bucket in deciles[DEPLOYMENT_SCOPE]["overall"]
    )
    assert len(deciles[CHANNEL_ACTIVE_SCOPE]["overall"]) == 10
    assert all(
        len(deciles[DEPLOYMENT_SCOPE][stage]) == 10
        for stage in ("early", "mid", "late")
    )
    assert all(
        len(deciles[CHANNEL_ACTIVE_SCOPE][stage]) == 9
        for stage in ("early", "mid", "late")
    )

    flips = result["margin_stratified_policy_flips"]
    expected_candidates = {
        _kappa_key(LOCAL_MINUS_KAPPA),
        _kappa_key(LOCAL_PLUS_KAPPA),
        _kappa_key(8.0),
        _kappa_key(64.0),
    }
    assert set(flips) == expected_candidates
    assert flips[_kappa_key(64.0)]["top_action_flip_count"] > (
        flips[_kappa_key(8.0)]["top_action_flip_count"]
    )
    for candidate in flips.values():
        bounds = candidate["necessary_movement_bounds"]
        assert bounds["l1_violation_count"] == 0
        assert bounds["linf_violation_count"] == 0
        assert set(candidate["by_reference_margin"]) == {
            "tie",
            "positive_at_most_one_reference_scale",
            "one_to_two_reference_scales",
            "above_two_reference_scales",
        }


def test_margin_bound_audit_detects_inconsistent_claimed_margin():
    payload, roots = _synthetic_probe()
    arrays = _extract_probe_arrays(
        payload=payload,
        roots=roots,
        local_log_step=DEFAULT_LOCAL_LOG_STEP,
        comparison_kappas=(8.0, 64.0),
    )
    margins = arrays.reference_margins.copy()
    margins[3] = 0.9
    tampered = replace(arrays, reference_margins=margins)

    summary = _margin_flip_summary(
        arrays=tampered,
        candidate_kappa=64.0,
    )

    assert 3 in summary["necessary_movement_bounds"][
        "l1_violation_root_ids"
    ]
    assert 3 in summary["necessary_movement_bounds"][
        "linf_violation_root_ids"
    ]


def test_probe_alignment_rejects_state_digest_mismatch():
    payload, roots = _synthetic_probe()
    payload["positions"][0]["state_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="state digest"):
        analyze(
            payload=payload,
            roots=roots,
            bootstrap_replicates=200,
        )


def test_probe_requires_boolean_reference_solved_root():
    payload, roots = _synthetic_probe()
    payload["positions"][0]["by_kappa"][
        _kappa_key(REFERENCE_KAPPA)
    ].pop("solved_root")

    with pytest.raises(ValueError, match="solved_root must be Boolean"):
        analyze(
            payload=payload,
            roots=roots,
            bootstrap_replicates=200,
        )
