#!/usr/bin/env python3
"""Explain test-time kappa on one frozen, stage-stratified Hex6 corpus.

The probe changes no training or search implementation.  It restores one
exact checkpoint step, replays the immutable roots collected by
``e4_repair_context_corpus.py``, and reruns the production
Dirichlet-Thompson search with identical root states and PRNG keys for every
requested kappa.  Only these evaluation-time fields are overridden:

* ``kappa``;
* ``root_action_estimator=prefix_cdf``; and
* ``prefix_cdf_half_width=10`` (Q21).

The checkpoint's internal ``posterior_policy_estimator`` is deliberately
preserved.  Thus the report measures the effect of kappa in the deployed
search path while removing finite-M32 noise from unresolved root commitment.
The normal production safety guard still falls back to the native readout and
is reported explicitly.

Example
-------

.. code-block:: bash

   JAX_PLATFORMS=cuda,cpu uv run python scripts/hex_kappa_fixed_probe.py \
     --checkpoint checkpoints/hex6_root_q21_target_s0 \
     --checkpoint-step 75 \
     --corpus experiments/e4/hex6_repair_contexts_v1 \
     --kappas 0.25,3,4,8,16,64 \
     --reference-kappa 3 \
     --exact-max-empties 15 \
     --output experiments/kappa_fixed_probe/e8_step75.json

This is a local diagnostic artifact.  It performs no network logging.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scacchi.types import PosteriorPolicyEstimator, SearchKind


FORMAT = "scacchi.hex_kappa_fixed_probe.v2"
STAGE_NAMES = ("early", "mid", "late")
Q21_HALF_WIDTH = 10
MAX_EXACT_EMPTY_CELLS = 15

# Public ``SearchOutput`` keeps diagnostics additive and batch-shaped.  This
# table is the script's explicit compatibility boundary: the first name is
# used in the artifact and the second is the existing SearchDiagnostics
# field returned by production search.
_SEARCH_DIAGNOSTIC_EXPORTS = (
    ("solved", "search_solved_root_count"),
    ("structural_support", "search_structural_support_sum"),
    ("repaired_actions", "search_repaired_action_count"),
    ("categorical_actions", "search_categorical_action_count"),
    ("legal_actions", "search_legal_action_count"),
    (
        "prefix_eligible",
        "search_root_action_prefix_eligible_count",
    ),
    (
        "prefix_accepted",
        "search_root_action_prefix_accepted_count",
    ),
    (
        "prefix_fallback",
        "search_root_action_prefix_fallback_count",
    ),
    (
        "prefix_tail_clipped",
        "search_root_action_prefix_tail_clipped_count",
    ),
    (
        "prefix_nonfinite",
        "search_root_action_prefix_nonfinite_count",
    ),
    (
        "kappa_numeric_repair_count",
        "search_kappa_numeric_repair_count",
    ),
    (
        "kappa_raw_innovation_l2_sum",
        "search_kappa_raw_innovation_l2_sum",
    ),
    (
        "kappa_semantic_innovation_l2_sum",
        "search_kappa_semantic_innovation_l2_sum",
    ),
    (
        "kappa_concentration_innovation_abs_sum",
        "search_kappa_concentration_innovation_abs_sum",
    ),
    (
        "kappa_raw_dcache_dlogkappa_l2_sum",
        "search_kappa_raw_dcache_dlogkappa_l2_sum",
    ),
    (
        "kappa_mean_dcache_dlogkappa_l2_sum",
        "search_kappa_mean_dcache_dlogkappa_l2_sum",
    ),
    (
        "kappa_log_concentration_dcache_dlogkappa_abs_sum",
        "search_kappa_log_concentration_dcache_dlogkappa_abs_sum",
    ),
    (
        "kappa_numeric_path_count",
        "search_kappa_numeric_path_count",
    ),
    (
        "kappa_path_gamma_product_sum",
        "search_kappa_path_gamma_product_sum",
    ),
    (
        "kappa_path_gamma_log_attenuation_sum",
        "search_kappa_path_gamma_log_attenuation_sum",
    ),
    (
        "kappa_categorical_publication_path_count",
        "search_kappa_categorical_publication_path_count",
    ),
    (
        "active_simulation_rows",
        "search_simulation_active_count",
    ),
    (
        "root_policy_top2_margin_sum",
        "search_root_policy_top2_margin_sum",
    ),
    (
        "root_policy_top2_margin_count",
        "search_root_policy_top2_margin_count",
    ),
    (
        "root_policy_top2_margin_tie_count",
        "search_root_policy_top2_margin_tie_count",
    ),
    (
        "root_policy_top2_margin_below_reference_count",
        "search_root_policy_top2_margin_below_reference_count",
    ),
    (
        "root_policy_top2_margin_reference_scale_sum",
        "search_root_policy_top2_margin_reference_scale_sum",
    ),
)


def _parse_kappas(value: str) -> tuple[float, ...]:
    """Parse a unique, positive, finite comma-separated kappa grid."""

    try:
        kappas = tuple(
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "kappas must be comma-separated floating-point values"
        ) from error
    if not kappas:
        raise argparse.ArgumentTypeError("at least one kappa is required")
    if any(not math.isfinite(kappa) or kappa <= 0.0 for kappa in kappas):
        raise argparse.ArgumentTypeError(
            "every kappa must be finite and positive"
        )
    if len(set(kappas)) != len(kappas):
        raise argparse.ArgumentTypeError("kappas must be unique")
    return kappas


def _kappa_key(kappa: float) -> str:
    return format(float(kappa), ".17g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entropy(policy: np.ndarray) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64)
    terms = np.zeros_like(policy)
    positive = policy > 0.0
    terms[positive] = policy[positive] * np.log(policy[positive])
    return -np.sum(terms, axis=-1)


def _effective_support(policy: np.ndarray) -> np.ndarray:
    """Inverse-Simpson effective action count, ``1 / sum(p**2)``."""

    policy = np.asarray(policy, dtype=np.float64)
    squared_mass = np.sum(policy * policy, axis=-1)
    return np.divide(
        1.0,
        squared_mass,
        out=np.zeros_like(squared_mass),
        where=squared_mass > 0.0,
    )


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence in nats."""

    left, right = np.broadcast_arrays(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    midpoint = 0.5 * (left + right)

    def kl(distribution: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(distribution)
        positive = distribution > 0.0
        terms[positive] = distribution[positive] * (
            np.log(distribution[positive])
            - np.log(midpoint[positive])
        )
        return np.sum(terms, axis=-1)

    return 0.5 * (kl(left) + kl(right))


def _mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)) if values.size else None


def _quantile(values: np.ndarray, probability: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    return (
        float(np.quantile(values, probability))
        if values.size
        else None
    )


def _distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(finite)),
        "mean": _mean(finite),
        "p05": _quantile(finite, 0.05),
        "p25": _quantile(finite, 0.25),
        "median": _quantile(finite, 0.50),
        "p75": _quantile(finite, 0.75),
        "p95": _quantile(finite, 0.95),
        "max": float(np.max(finite)) if finite.size else None,
    }


def _validated_probability_rows(
    policy: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64)
    if policy.ndim != 2:
        raise ValueError(f"{label} must have shape [root, action]")
    if not np.all(np.isfinite(policy)) or np.any(policy < -1e-7):
        raise ValueError(f"{label} contains invalid probabilities")
    policy = np.maximum(policy, 0.0)
    mass = np.sum(policy, axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError(f"{label} contains a zero-mass row")
    return policy / mass


def _summarize_mask(
    *,
    mask: np.ndarray,
    sample_weights: np.ndarray,
    prior: np.ndarray,
    policies: dict[float, np.ndarray],
    diagnostics: dict[float, dict[str, np.ndarray]],
    reference_kappa: float,
    oracle: dict[str, Any] | None,
) -> dict[str, Any]:
    count = int(np.sum(mask))
    output: dict[str, Any] = {"roots": count, "kappas": {}}
    if count == 0:
        return output

    local_prior = prior[mask]
    reference = policies[reference_kappa][mask]
    reference_top = np.argmax(reference, axis=-1)
    for kappa, complete_policy in policies.items():
        policy = complete_policy[mask]
        diagnostic = diagnostics[kappa]
        l1_to_reference = np.sum(np.abs(policy - reference), axis=-1)
        js_to_reference = _js(policy, reference)
        prior_l1 = np.sum(np.abs(policy - local_prior), axis=-1)
        prior_js = _js(policy, local_prior)
        structural_support = diagnostic["structural_support"][mask]
        unresolved = diagnostic["solved"][mask] <= 0.5
        positive_support = unresolved & (structural_support > 0.0)
        local_n = structural_support[positive_support]
        implied_length = (
            1.0 / np.log1p(float(kappa) / local_n)
            if local_n.size
            else np.asarray([], dtype=np.float64)
        )
        descendant_weight = (
            local_n / (float(kappa) + local_n)
            if local_n.size
            else np.asarray([], dtype=np.float64)
        )
        eligible = diagnostic["prefix_eligible"][mask] > 0.5
        accepted = diagnostic["prefix_accepted"][mask] > 0.5
        fallback = diagnostic["prefix_fallback"][mask] > 0.5
        summary: dict[str, Any] = {
            "root_policy_vs_reference_kappa": {
                "mean_l1": _mean(l1_to_reference),
                "p95_l1": _quantile(l1_to_reference, 0.95),
                "max_l1": (
                    float(np.max(l1_to_reference))
                    if l1_to_reference.size
                    else None
                ),
                "mean_js_nats": _mean(js_to_reference),
                "p95_js_nats": _quantile(js_to_reference, 0.95),
                "top_action_flip_fraction": _mean(
                    np.argmax(policy, axis=-1) != reference_top
                ),
            },
            "prior_to_search_displacement": {
                "mean_l1": _mean(prior_l1),
                "p95_l1": _quantile(prior_l1, 0.95),
                "mean_js_nats": _mean(prior_js),
                "p95_js_nats": _quantile(prior_js, 0.95),
            },
            "policy_shape": {
                "mean_entropy_nats": _mean(_entropy(policy)),
                "p95_entropy_nats": _quantile(_entropy(policy), 0.95),
                "mean_inverse_simpson_ess": _mean(
                    _effective_support(policy)
                ),
                "p95_inverse_simpson_ess": _quantile(
                    _effective_support(policy),
                    0.95,
                ),
            },
            "search_structure": {
                "solved_root_fraction": _mean(
                    diagnostic["solved"][mask]
                ),
                "solved_root_bypass_fraction": _mean(
                    diagnostic["solved"][mask]
                ),
                "mean_categorical_edge_fraction": _mean(
                    np.divide(
                        diagnostic["categorical_actions"][mask],
                        diagnostic["legal_actions"][mask],
                        out=np.zeros(count, dtype=np.float64),
                        where=diagnostic["legal_actions"][mask] > 0.0,
                    )
                ),
                "mean_structural_support": _mean(
                    structural_support
                ),
                "unresolved_roots": int(np.sum(unresolved)),
                "zero_structural_support_fraction_of_unresolved": (
                    _mean(structural_support[unresolved] <= 0.0)
                    if np.any(unresolved)
                    else None
                ),
                "positive_unresolved_root_n_down": (
                    _distribution_summary(local_n)
                ),
                "implied_local_e_fold_length": {
                    "definition": "1/log(1+kappa/n_down)",
                    **_distribution_summary(implied_length),
                },
                "descendant_mix_weight_gamma": {
                    "definition": "n_down/(kappa+n_down)",
                    **_distribution_summary(descendant_weight),
                },
                "mean_repaired_action_fraction": _mean(
                    np.divide(
                        diagnostic["repaired_actions"][mask],
                        diagnostic["legal_actions"][mask],
                        out=np.zeros(count, dtype=np.float64),
                        where=diagnostic["legal_actions"][mask] > 0.0,
                    )
                ),
            },
            "q21_action_readout": {
                "eligible_roots": int(np.sum(eligible)),
                "accepted_fraction_of_eligible": (
                    _mean(accepted[eligible])
                    if np.any(eligible)
                    else None
                ),
                "fallback_fraction_of_eligible": (
                    _mean(fallback[eligible])
                    if np.any(eligible)
                    else None
                ),
            },
        }
        if oracle is not None:
            oracle_mask = mask & oracle["available"]
            oracle_policy = complete_policy[oracle_mask]
            oracle_reference = policies[reference_kappa][oracle_mask]
            oracle_weights = sample_weights[oracle_mask]
            oracle_flips = (
                np.argmax(oracle_policy, axis=-1)
                != np.argmax(oracle_reference, axis=-1)
            )
            candidate_top_regret = oracle["top_action_regret"][kappa][
                oracle_mask
            ]
            reference_top_regret = oracle["top_action_regret"][
                reference_kappa
            ][oracle_mask]
            top_regret_delta = (
                candidate_top_regret - reference_top_regret
            )
            strictly_worse_flips = oracle_flips & (top_regret_delta > 0.0)
            positive_flip_regret = np.where(
                oracle_flips,
                np.maximum(top_regret_delta, 0.0),
                0.0,
            )
            oracle_root_count = int(np.sum(oracle_mask))
            flip_count = int(np.sum(oracle_flips))
            strictly_worse_flip_count = int(
                np.sum(strictly_worse_flips)
            )
            oracle_sample_weight = float(np.sum(oracle_weights))
            flip_sample_weight = float(
                np.sum(oracle_weights[oracle_flips])
            )
            strictly_worse_flip_sample_weight = float(
                np.sum(oracle_weights[strictly_worse_flips])
            )
            weighted_positive_flip_regret = float(
                np.sum(oracle_weights * positive_flip_regret)
            )
            both_optimal = (
                oracle["top_action_optimal"][kappa][oracle_mask] > 0.5
            ) & (
                oracle["top_action_optimal"][reference_kappa][oracle_mask]
                > 0.5
            )
            flipped_but_equivalent = oracle_flips & both_optimal
            summary["exact_oracle"] = {
                "roots": oracle_root_count,
                "mean_normalized_expected_regret": _mean(
                    oracle["expected_regret"][kappa][oracle_mask]
                ),
                "mean_normalized_top_action_regret": _mean(
                    candidate_top_regret
                ),
                "top_action_optimal_fraction": _mean(
                    oracle["top_action_optimal"][kappa][oracle_mask]
                ),
                "mean_optimal_action_mass": _mean(
                    oracle["optimal_action_mass"][kappa][oracle_mask]
                ),
                "mean_normalized_expected_regret_delta_vs_reference": _mean(
                    oracle["expected_regret"][kappa][oracle_mask]
                    - oracle["expected_regret"][reference_kappa][oracle_mask]
                ),
                "top_action_flip_fraction_vs_reference": _mean(oracle_flips),
                "flip_but_both_actions_optimal_fraction_of_all": _mean(
                    flipped_but_equivalent
                ),
                "flip_but_both_actions_optimal_fraction_of_flips": (
                    _mean(both_optimal[oracle_flips])
                    if np.any(oracle_flips)
                    else None
                ),
                "decisive_flips_vs_reference": {
                    "definitions": {
                        "oracle_root_denominator": (
                            "all exact-oracle roots in this summary slice"
                        ),
                        "oracle_sample_weight_denominator": (
                            "sum of corpus root_weight over all exact-oracle "
                            "roots in this summary slice"
                        ),
                        "strictly_worse_outcome": (
                            "the candidate top action has larger normalized "
                            "oracle outcome regret than the reference-kappa "
                            "top action"
                        ),
                        "positive_regret_attributable_to_flips": (
                            "1[top actions differ] * "
                            "max(candidate normalized top-action regret - "
                            "reference normalized top-action regret, 0)"
                        ),
                    },
                    "oracle_root_denominator": oracle_root_count,
                    "oracle_sample_weight_denominator": (
                        oracle_sample_weight
                    ),
                    "top_action_flip_count": flip_count,
                    "top_action_flip_fraction_of_oracle_roots": (
                        float(flip_count / oracle_root_count)
                        if oracle_root_count
                        else None
                    ),
                    "top_action_flip_sample_weight": flip_sample_weight,
                    "top_action_flip_fraction_of_oracle_sample_weight": (
                        flip_sample_weight / oracle_sample_weight
                        if oracle_sample_weight > 0.0
                        else None
                    ),
                    "strictly_worse_outcome_flip_count": (
                        strictly_worse_flip_count
                    ),
                    "strictly_worse_outcome_flip_fraction_of_oracle_roots": (
                        float(
                            strictly_worse_flip_count / oracle_root_count
                        )
                        if oracle_root_count
                        else None
                    ),
                    "strictly_worse_outcome_flip_fraction_of_flips": (
                        float(strictly_worse_flip_count / flip_count)
                        if flip_count
                        else None
                    ),
                    "strictly_worse_outcome_flip_sample_weight": (
                        strictly_worse_flip_sample_weight
                    ),
                    "strictly_worse_outcome_flip_fraction_of_oracle_sample_weight": (
                        strictly_worse_flip_sample_weight
                        / oracle_sample_weight
                        if oracle_sample_weight > 0.0
                        else None
                    ),
                    "strictly_worse_outcome_flip_fraction_of_flipped_sample_weight": (
                        strictly_worse_flip_sample_weight / flip_sample_weight
                        if flip_sample_weight > 0.0
                        else None
                    ),
                    "positive_normalized_top_action_regret_delta_sum_on_flips": (
                        float(np.sum(positive_flip_regret))
                    ),
                    "positive_normalized_top_action_regret_delta_mean_per_oracle_root": (
                        _mean(positive_flip_regret)
                    ),
                    "positive_normalized_top_action_regret_delta_mean_per_flip": (
                        _mean(positive_flip_regret[oracle_flips])
                        if np.any(oracle_flips)
                        else None
                    ),
                    "sample_weighted_positive_normalized_top_action_regret_delta_numerator_on_flips": (
                        weighted_positive_flip_regret
                    ),
                    "sample_weighted_positive_normalized_top_action_regret_delta_mean_per_oracle_sample": (
                        weighted_positive_flip_regret
                        / oracle_sample_weight
                        if oracle_sample_weight > 0.0
                        else None
                    ),
                    "sample_weighted_positive_normalized_top_action_regret_delta_mean_per_flipped_sample": (
                        weighted_positive_flip_regret
                        / flip_sample_weight
                        if flip_sample_weight > 0.0
                        else None
                    ),
                },
            }
        output["kappas"][_kappa_key(kappa)] = summary
    return output


def _summarize(
    *,
    stage_ids: np.ndarray,
    prior: np.ndarray,
    policies: dict[float, np.ndarray],
    diagnostics: dict[float, dict[str, np.ndarray]],
    reference_kappa: float,
    oracle: dict[str, Any] | None,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    if sample_weights is None:
        sample_weights = np.ones(len(stage_ids), dtype=np.float64)
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        if sample_weights.shape != (len(stage_ids),):
            raise ValueError(
                "sample_weights must have shape [root]; got "
                f"{sample_weights.shape} for {len(stage_ids)} roots"
            )
        if (
            not np.all(np.isfinite(sample_weights))
            or np.any(sample_weights < 0.0)
        ):
            raise ValueError(
                "sample_weights must contain finite non-negative values"
            )
    all_roots = np.ones(len(stage_ids), dtype=bool)
    return {
        "overall": _summarize_mask(
            mask=all_roots,
            sample_weights=sample_weights,
            prior=prior,
            policies=policies,
            diagnostics=diagnostics,
            reference_kappa=reference_kappa,
            oracle=oracle,
        ),
        "by_stage": {
            name: _summarize_mask(
                mask=stage_ids == stage_id,
                sample_weights=sample_weights,
                prior=prior,
                policies=policies,
                diagnostics=diagnostics,
                reference_kappa=reference_kappa,
                oracle=oracle,
            )
            for stage_id, name in enumerate(STAGE_NAMES)
        },
    }


def _effective_search_config(
    search_config: Any,
    *,
    kappa: float,
) -> Any:
    """Override only kappa and the deployed root Q21 readout."""

    return replace(
        search_config,
        kappa=float(kappa),
        root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
        prefix_cdf_half_width=Q21_HALF_WIDTH,
    )


def _make_probe_inference(
    env: Any,
    search_config: Any,
    *,
    q_loss_weight_mode: str,
):
    """Build the production search path with a compact diagnostic output."""

    from flax import nnx
    import jax
    import jax.numpy as jnp

    from scacchi.dirichlet_q_search import make_dirichlet_expand_fn
    from scacchi.play_search import (
        _run_dirichlet_thompson_search,
        _with_root_policy_top2_margin_diagnostics,
        make_evaluator,
    )

    @nnx.jit
    def infer(model, state, rng_key):
        evaluator = make_evaluator(model)
        prediction = evaluator(state.observation)
        output = _run_dirichlet_thompson_search(
            state,
            prediction,
            make_dirichlet_expand_fn(env, evaluator),
            rng_key,
            search_config,
            q_loss_weight_mode,
        )
        output = _with_root_policy_top2_margin_diagnostics(
            output,
            state.legal_action_mask,
            action_commitment_type="posterior_argmax",
            margin_reference_scale=(
                1.0 / max(1, int(search_config.policy_samples))
            ),
        )
        commitment = output.commitment_policy
        if commitment is None:
            raise ValueError(
                "Q21 root action estimator did not expose commitment policy"
            )
        masked_logits = jnp.where(
            state.legal_action_mask,
            prediction.logits,
            jnp.finfo(prediction.logits.dtype).min,
        )
        prior = jax.nn.softmax(masked_logits, axis=-1)
        diagnostic = output.posterior.diagnostics
        if diagnostic is None:
            raise ValueError("Dirichlet search did not expose diagnostics")
        return (
            prior,
            commitment,
            *(
                getattr(diagnostic, field)
                for _, field in _SEARCH_DIAGNOSTIC_EXPORTS
            ),
        )

    return infer


def _load_corpus(
    corpus: Path,
    *,
    roots_per_stage: int | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    manifest_path = corpus / "manifest.json"
    roots_path = corpus / "roots.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_digest = manifest.get("artifacts", {}).get("roots.npz")
    actual_digest = _sha256(roots_path)
    if expected_digest != actual_digest:
        raise ValueError(
            "frozen roots digest disagrees with manifest: "
            f"{actual_digest} != {expected_digest}"
        )
    roots = dict(np.load(roots_path, allow_pickle=False))
    count = len(roots["root_id"])
    if int(manifest.get("root_count", -1)) != count:
        raise ValueError("manifest root_count disagrees with roots.npz")
    stage_ids = np.asarray(roots["stage_id"], dtype=np.int8)
    if not set(np.unique(stage_ids)).issubset({0, 1, 2}):
        raise ValueError("frozen roots contain an unknown stage id")

    if roots_per_stage is None:
        indices = np.arange(count, dtype=np.int64)
    else:
        if roots_per_stage < 1:
            raise ValueError("--roots-per-stage must be positive")
        selected: list[int] = []
        state_hashes = np.asarray(roots["state_sha256"]).astype("S64")
        for stage_id in range(len(STAGE_NAMES)):
            candidates = np.flatnonzero(stage_ids == stage_id)
            if len(candidates) < roots_per_stage:
                raise ValueError(
                    f"stage {STAGE_NAMES[stage_id]} has only "
                    f"{len(candidates)} roots"
                )
            ordered = sorted(
                candidates.tolist(),
                key=lambda index: (
                    bytes(state_hashes[index]),
                    int(roots["root_id"][index]),
                ),
            )
            selected.extend(ordered[:roots_per_stage])
        indices = np.asarray(
            sorted(selected, key=lambda index: int(roots["root_id"][index])),
            dtype=np.int64,
        )
    selected_roots = {
        name: np.asarray(value)[indices]
        for name, value in roots.items()
    }
    return manifest, selected_roots, indices


def _slice_state(state: Any, start: int, stop: int) -> Any:
    import jax

    return jax.tree.map(lambda value: value[start:stop], state)


def _key_data(key: Any) -> list[int]:
    import jax

    data = key
    if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        data = jax.random.key_data(key)
    return [
        int(value)
        for value in np.asarray(jax.device_get(data)).reshape(-1)
    ]


def _run_searches(
    *,
    model: Any,
    env: Any,
    states: Any,
    stored_search_config: Any,
    q_loss_weight_mode: str,
    kappas: Sequence[float],
    batch_size: int,
    seed: int,
) -> tuple[
    np.ndarray,
    dict[float, np.ndarray],
    dict[float, dict[str, np.ndarray]],
    list[dict[str, Any]],
    dict[str, float],
]:
    import jax

    root_count = int(states.current_player.shape[0])
    batch_ranges = [
        (start, min(start + batch_size, root_count))
        for start in range(0, root_count, batch_size)
    ]
    base_key = jax.random.PRNGKey(seed)
    tree_keys = [
        jax.random.fold_in(base_key, 1_000_000 + index)
        for index in range(len(batch_ranges))
    ]
    key_provenance = [
        {
            "batch_index": index,
            "root_index_start": start,
            "root_index_stop": stop,
            "tree_key_data": _key_data(tree_keys[index]),
        }
        for index, (start, stop) in enumerate(batch_ranges)
    ]

    prior_reference: np.ndarray | None = None
    policies: dict[float, np.ndarray] = {}
    all_diagnostics: dict[float, dict[str, np.ndarray]] = {}
    elapsed_by_kappa: dict[str, float] = {}
    names = tuple(name for name, _ in _SEARCH_DIAGNOSTIC_EXPORTS)
    for kappa in kappas:
        effective = _effective_search_config(
            stored_search_config,
            kappa=kappa,
        )
        infer = _make_probe_inference(
            env,
            effective,
            q_loss_weight_mode=q_loss_weight_mode,
        )
        prior_chunks: list[np.ndarray] = []
        policy_chunks: list[np.ndarray] = []
        diagnostic_chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in names
        }
        started = time.perf_counter()
        for batch_index, (start, stop) in enumerate(batch_ranges):
            output = jax.block_until_ready(
                infer(
                    model,
                    _slice_state(states, start, stop),
                    tree_keys[batch_index],
                )
            )
            prior_chunks.append(np.asarray(jax.device_get(output[0])))
            policy_chunks.append(np.asarray(jax.device_get(output[1])))
            for name, value in zip(names, output[2:], strict=True):
                diagnostic_chunks[name].append(
                    np.asarray(jax.device_get(value), dtype=np.float64)
                )
        elapsed_by_kappa[_kappa_key(kappa)] = (
            time.perf_counter() - started
        )
        prior = _validated_probability_rows(
            np.concatenate(prior_chunks, axis=0),
            label=f"kappa={kappa} prior",
        )
        policy = _validated_probability_rows(
            np.concatenate(policy_chunks, axis=0),
            label=f"kappa={kappa} Q21 policy",
        )
        if prior_reference is None:
            prior_reference = prior
        elif not np.allclose(prior, prior_reference, rtol=0.0, atol=1e-7):
            raise AssertionError(
                "network prior changed across kappa runs on frozen states"
            )
        policies[float(kappa)] = policy
        all_diagnostics[float(kappa)] = {
            name: np.concatenate(values, axis=0)
            for name, values in diagnostic_chunks.items()
        }
    if prior_reference is None:
        raise AssertionError("no kappa search was executed")
    return (
        prior_reference,
        policies,
        all_diagnostics,
        key_provenance,
        elapsed_by_kappa,
    )


def _exact_oracle(
    *,
    states: Any,
    policies: dict[float, np.ndarray],
    exact_max_empties: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any] | None]]:
    if exact_max_empties <= 0:
        return None, [None] * int(states.current_player.shape[0])
    if exact_max_empties > MAX_EXACT_EMPTY_CELLS:
        raise ValueError(
            "--exact-max-empties may not exceed "
            f"{MAX_EXACT_EMPTY_CELLS}"
        )

    import jax

    from scacchi.hex_oracle import assess_policy_against_oracle, solve_hex
    from scacchi.hex_oracle import position_from_pgx_state

    root_count = int(states.current_player.shape[0])
    available = np.zeros(root_count, dtype=bool)
    records: list[dict[str, Any] | None] = [None] * root_count
    expected_regret = {
        kappa: np.full(root_count, np.nan, dtype=np.float64)
        for kappa in policies
    }
    top_action_regret = {
        kappa: np.full(root_count, np.nan, dtype=np.float64)
        for kappa in policies
    }
    top_action_optimal = {
        kappa: np.zeros(root_count, dtype=np.float64)
        for kappa in policies
    }
    optimal_action_mass = {
        kappa: np.full(root_count, np.nan, dtype=np.float64)
        for kappa in policies
    }
    for index in range(root_count):
        scalar_state = jax.tree.map(lambda value: value[index], states)
        position = position_from_pgx_state(scalar_state)
        if position.empty_count > exact_max_empties:
            continue
        result = solve_hex(position)
        available[index] = True
        values = dict(result.action_values)
        assessments: dict[str, Any] = {}
        for kappa, policy in policies.items():
            assessment = assess_policy_against_oracle(
                policy[index],
                result,
            )
            # Hex oracle outcomes are {-1,+1}; divide by two to put decision
            # regret on [0,1].
            expected_regret[kappa][index] = assessment.regret / 2.0
            top_action_regret[kappa][index] = (
                result.outcome - values[assessment.top_action]
            ) / 2.0
            top_action_optimal[kappa][index] = float(
                assessment.top_action_is_optimal
            )
            optimal_action_mass[kappa][index] = (
                assessment.optimal_action_mass
            )
            assessments[_kappa_key(kappa)] = {
                "normalized_expected_regret": (
                    expected_regret[kappa][index]
                ),
                "normalized_top_action_regret": (
                    top_action_regret[kappa][index]
                ),
                "top_action_optimal": bool(
                    top_action_optimal[kappa][index]
                ),
                "optimal_action_mass": optimal_action_mass[kappa][index],
            }
        records[index] = {
            "oracle_position_id": position.position_id,
            "oracle_outcome": result.outcome,
            "oracle_optimal_actions": list(result.optimal_actions),
            "by_kappa": assessments,
        }
    return (
        {
            "available": available,
            "expected_regret": expected_regret,
            "top_action_regret": top_action_regret,
            "top_action_optimal": top_action_optimal,
            "optimal_action_mass": optimal_action_mass,
        },
        records,
    )


def _per_root_mean(
    diagnostic: dict[str, np.ndarray],
    *,
    numerator: str,
    denominator: str,
    index: int,
) -> float | None:
    count = float(diagnostic[denominator][index])
    value = float(diagnostic[numerator][index])
    if count <= 0.0 or not math.isfinite(count) or not math.isfinite(value):
        return None
    return value / count


def _root_kappa_channel(
    diagnostic: dict[str, np.ndarray],
    *,
    index: int,
) -> dict[str, Any]:
    """Decode one root's additive κ diagnostics without changing search."""

    repair_count = int(diagnostic["kappa_numeric_repair_count"][index])
    numeric_path_count = int(diagnostic["kappa_numeric_path_count"][index])
    active_simulation_rows = int(
        diagnostic["active_simulation_rows"][index]
    )
    categorical_path_count = int(
        diagnostic["kappa_categorical_publication_path_count"][index]
    )
    margin_count = float(
        diagnostic["root_policy_top2_margin_count"][index]
    )
    margin = _per_root_mean(
        diagnostic,
        numerator="root_policy_top2_margin_sum",
        denominator="root_policy_top2_margin_count",
        index=index,
    )
    margin_reference_scale = _per_root_mean(
        diagnostic,
        numerator="root_policy_top2_margin_reference_scale_sum",
        denominator="root_policy_top2_margin_count",
        index=index,
    )
    return {
        "numeric_repairs": {
            "count": repair_count,
            "raw_innovation_l2_mean": _per_root_mean(
                diagnostic,
                numerator="kappa_raw_innovation_l2_sum",
                denominator="kappa_numeric_repair_count",
                index=index,
            ),
            "semantic_innovation_l2_mean": _per_root_mean(
                diagnostic,
                numerator="kappa_semantic_innovation_l2_sum",
                denominator="kappa_numeric_repair_count",
                index=index,
            ),
            "concentration_innovation_abs_mean": _per_root_mean(
                diagnostic,
                numerator="kappa_concentration_innovation_abs_sum",
                denominator="kappa_numeric_repair_count",
                index=index,
            ),
            "raw_dcache_dlogkappa_l2_mean": _per_root_mean(
                diagnostic,
                numerator="kappa_raw_dcache_dlogkappa_l2_sum",
                denominator="kappa_numeric_repair_count",
                index=index,
            ),
            "mean_dcache_dlogkappa_l2_mean": _per_root_mean(
                diagnostic,
                numerator="kappa_mean_dcache_dlogkappa_l2_sum",
                denominator="kappa_numeric_repair_count",
                index=index,
            ),
            "log_concentration_dcache_dlogkappa_abs_mean": (
                _per_root_mean(
                    diagnostic,
                    numerator=(
                        "kappa_log_concentration_"
                        "dcache_dlogkappa_abs_sum"
                    ),
                    denominator="kappa_numeric_repair_count",
                    index=index,
                )
            ),
        },
        "numeric_paths": {
            "count": numeric_path_count,
            "gamma_product_mean": _per_root_mean(
                diagnostic,
                numerator="kappa_path_gamma_product_sum",
                denominator="kappa_numeric_path_count",
                index=index,
            ),
            "gamma_log_attenuation_mean": _per_root_mean(
                diagnostic,
                numerator="kappa_path_gamma_log_attenuation_sum",
                denominator="kappa_numeric_path_count",
                index=index,
            ),
        },
        "categorical_publication_paths": {
            "count": categorical_path_count,
            "active_simulation_rows": active_simulation_rows,
            "fraction_of_active_simulation_rows": (
                float(categorical_path_count / active_simulation_rows)
                if active_simulation_rows > 0
                else None
            ),
        },
        "commitment_policy_top2": {
            "eligible": margin_count > 0.0,
            "margin": margin,
            "tie": (
                bool(
                    diagnostic[
                        "root_policy_top2_margin_tie_count"
                    ][index]
                    > 0.5
                )
                if margin_count > 0.0
                else None
            ),
            "below_reference_scale": (
                bool(
                    diagnostic[
                        "root_policy_top2_margin_below_reference_count"
                    ][index]
                    > 0.5
                )
                if margin_count > 0.0
                else None
            ),
            "reference_scale": margin_reference_scale,
        },
    }


def _paired_root_response(
    *,
    kappa: float,
    reference_kappa: float,
    policy: np.ndarray,
    reference_policy: np.ndarray,
    channel: dict[str, Any],
    reference_channel: dict[str, Any],
) -> dict[str, Any]:
    """Pair observed root movement with the reference local κ scale."""

    delta_log_kappa = math.log(float(kappa) / float(reference_kappa))
    absolute_delta = abs(delta_log_kappa)
    policy_l1 = float(
        np.sum(
            np.abs(
                np.asarray(policy, dtype=np.float64)
                - np.asarray(reference_policy, dtype=np.float64)
            )
        )
    )
    reference_margin = reference_channel["commitment_policy_top2"][
        "margin"
    ]
    reference_derivative = reference_channel["numeric_repairs"][
        "mean_dcache_dlogkappa_l2_mean"
    ]
    return {
        "reference_kappa": float(reference_kappa),
        "delta_log_kappa": delta_log_kappa,
        "absolute_delta_log_kappa": absolute_delta,
        "root_policy_l1": policy_l1,
        "root_policy_js_nats": float(
            _js(
                np.asarray(policy, dtype=np.float64)[None, :],
                np.asarray(reference_policy, dtype=np.float64)[None, :],
            )[0]
        ),
        "root_policy_l1_per_abs_delta_log_kappa": (
            policy_l1 / absolute_delta if absolute_delta > 0.0 else None
        ),
        "top_action_flipped": bool(
            int(np.argmax(policy)) != int(np.argmax(reference_policy))
        ),
        "reference_top_action": int(np.argmax(reference_policy)),
        "candidate_top_action": int(np.argmax(policy)),
        "reference_commitment_policy_top2_margin": reference_margin,
        "candidate_commitment_policy_top2_margin": channel[
            "commitment_policy_top2"
        ]["margin"],
        "root_policy_l1_meets_flip_margin_necessary_condition": (
            bool(policy_l1 + 1e-12 >= reference_margin)
            if reference_margin is not None
            else None
        ),
        "reference_numeric_repair_mean_dmean_dlogkappa_l2": (
            reference_derivative
        ),
        "reference_first_order_cache_movement_scale_l2": (
            reference_derivative * absolute_delta
            if reference_derivative is not None
            else None
        ),
    }


def _position_rows(
    *,
    roots: dict[str, np.ndarray],
    prior: np.ndarray,
    policies: dict[float, np.ndarray],
    diagnostics: dict[float, dict[str, np.ndarray]],
    reference_kappa: float,
    oracle_records: Sequence[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    action_count = np.asarray(roots["action_count"], dtype=np.int32)
    for index in range(len(action_count)):
        stage_id = int(roots["stage_id"][index])
        state_digest = roots["state_sha256"][index]
        if isinstance(state_digest, bytes | np.bytes_):
            state_digest = bytes(state_digest).decode("ascii")
        reference_policy = policies[reference_kappa][index]
        reference_channel = _root_kappa_channel(
            diagnostics[reference_kappa],
            index=index,
        )
        by_kappa: dict[str, Any] = {}
        for kappa, policy in policies.items():
            diagnostic = diagnostics[kappa]
            n_down = float(diagnostic["structural_support"][index])
            solved = bool(diagnostic["solved"][index] > 0.5)
            channel = _root_kappa_channel(diagnostic, index=index)
            by_kappa[_kappa_key(kappa)] = {
                "root_policy": policy[index].tolist(),
                "top_action": int(np.argmax(policy[index])),
                "entropy_nats": float(_entropy(policy[index : index + 1])[0]),
                "inverse_simpson_ess": float(
                    _effective_support(policy[index : index + 1])[0]
                ),
                "solved_root": solved,
                "structural_support_n_down": n_down,
                "implied_local_e_fold_length": (
                    1.0 / math.log1p(float(kappa) / n_down)
                    if not solved and n_down > 0.0
                    else None
                ),
                "descendant_mix_weight_gamma": (
                    n_down / (float(kappa) + n_down)
                    if not solved and n_down > 0.0
                    else None
                ),
                "categorical_action_fraction": (
                    float(diagnostic["categorical_actions"][index])
                    / float(diagnostic["legal_actions"][index])
                    if diagnostic["legal_actions"][index] > 0.0
                    else None
                ),
                "q21_prefix_eligible": bool(
                    diagnostic["prefix_eligible"][index] > 0.5
                ),
                "q21_prefix_accepted": bool(
                    diagnostic["prefix_accepted"][index] > 0.5
                ),
                "q21_prefix_fallback": bool(
                    diagnostic["prefix_fallback"][index] > 0.5
                ),
                "q21_tail_range_clipped": bool(
                    diagnostic["prefix_tail_clipped"][index] > 0.5
                ),
                "q21_nonfinite": bool(
                    diagnostic["prefix_nonfinite"][index] > 0.5
                ),
                "kappa_channel": channel,
                "paired_response_vs_reference": _paired_root_response(
                    kappa=kappa,
                    reference_kappa=reference_kappa,
                    policy=policy[index],
                    reference_policy=reference_policy,
                    channel=channel,
                    reference_channel=reference_channel,
                ),
            }
        rows.append(
            {
                "root_id": int(roots["root_id"][index]),
                "state_sha256": str(state_digest),
                "source_checkpoint_step": int(
                    roots["checkpoint_step"][index]
                ),
                "stage": STAGE_NAMES[stage_id],
                "stage_id": stage_id,
                "root_ply": int(action_count[index]),
                "empty_count": 36 - int(action_count[index]),
                "corpus_root_weight": float(
                    roots["root_weight"][index]
                ),
                "prior_policy": prior[index].tolist(),
                "by_kappa": by_kappa,
                "exact_oracle": oracle_records[index],
            }
        )
    return rows


def _write_immutable(path: Path, payload: dict[str, Any]) -> str:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(
            f"refusing to overwrite diagnostic artifact: {resolved}"
        )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".tmp",
    )
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
        Path(temporary_name).replace(resolved)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO / "experiments/e4/hex6_repair_contexts_v1",
    )
    parser.add_argument(
        "--kappas",
        type=_parse_kappas,
        default=_parse_kappas("0.25,3,4,8,16,64"),
    )
    parser.add_argument("--reference-kappa", type=float, default=3.0)
    parser.add_argument(
        "--search-source",
        choices=("eval", "selfplay"),
        default="eval",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--roots-per-stage",
        type=int,
        default=None,
        help="optional deterministic equal-size subset from each stage",
    )
    parser.add_argument(
        "--exact-max-empties",
        type=int,
        default=0,
        help=(
            "solve positions at or below this empty-cell count; 0 disables "
            f"exact scoring and the safe maximum is {MAX_EXACT_EMPTY_CELLS}"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if (
        not math.isfinite(args.reference_kappa)
        or args.reference_kappa <= 0.0
    ):
        raise ValueError("--reference-kappa must be finite and positive")
    if args.reference_kappa not in set(args.kappas):
        raise ValueError("--reference-kappa must appear in --kappas")
    if not 0 <= args.exact_max_empties <= MAX_EXACT_EMPTY_CELLS:
        raise ValueError(
            "--exact-max-empties must be between 0 and "
            f"{MAX_EXACT_EMPTY_CELLS}"
        )

    import jax

    from e4_repair_context_corpus import (
        _load_checkpoint_config_and_model,
        _replay_roots,
    )

    manifest, roots, selected_indices = _load_corpus(
        args.corpus.resolve(),
        roots_per_stage=args.roots_per_stage,
    )
    config, env, model, checkpoint_progress = (
        _load_checkpoint_config_and_model(
            args.checkpoint,
            args.checkpoint_step,
        )
    )
    search = (
        config.eval.player_search
        if args.search_source == "eval"
        else config.selfplay.search
    )
    if search.kind != SearchKind.dirichlet_thompson:
        raise ValueError(
            f"{args.search_source} search must be Dirichlet-Thompson"
        )
    stored_search_config = search.dirichlet_thompson
    states = _replay_roots(env, roots)
    (
        prior,
        policies,
        diagnostics,
        key_provenance,
        elapsed_by_kappa,
    ) = _run_searches(
        model=model,
        env=env,
        states=states,
        stored_search_config=stored_search_config,
        q_loss_weight_mode=str(config.training.losses.q_loss_weight_mode),
        kappas=args.kappas,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    oracle, oracle_records = _exact_oracle(
        states=states,
        policies=policies,
        exact_max_empties=args.exact_max_empties,
    )
    summary = _summarize(
        stage_ids=np.asarray(roots["stage_id"], dtype=np.int8),
        prior=prior,
        policies=policies,
        diagnostics=diagnostics,
        reference_kappa=args.reference_kappa,
        oracle=oracle,
        sample_weights=np.asarray(roots["root_weight"], dtype=np.float64),
    )
    source_paths = {
        "manifest": args.corpus.resolve() / "manifest.json",
        "roots": args.corpus.resolve() / "roots.npz",
    }
    payload = {
        "format": FORMAT,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
            "immutable_output": True,
        },
        "protocol": {
            "causal_factor": (
                "evaluation Dirichlet-Thompson kappa, the prior mass in "
                "gamma=n/(kappa+n)"
            ),
            "frozen": [
                "checkpoint weights and exact retained step",
                "frozen root states and ordering",
                "32-simulation budget and all non-kappa search settings",
                "one tree PRNG key per recorded batch, reused for every kappa",
                "guarded prefix-CDF Q21 deployed root action readout",
            ],
            "root_readout_override": {
                "root_action_estimator": "prefix_cdf",
                "prefix_cdf_half_width": Q21_HALF_WIDTH,
                "prefix_cdf_grid_points": 2 * Q21_HALF_WIDTH + 1,
                "scope": (
                    "deployed root commitment distribution only; the "
                    "checkpoint's internal posterior estimator is preserved"
                ),
                "safety": (
                    "production tail/density/nonfinite guard with native "
                    "readout fallback, reported per root"
                ),
            },
            "reference_kappa": args.reference_kappa,
            "metrics": {
                "js_units": "natural-log nats",
                "ess": "inverse Simpson effective action count 1/sum(p^2)",
                "root_n_down": (
                    "sum of unresolved root edge structural visit counts; "
                    "categorical payloads are distances and excluded"
                ),
                "local_e_fold_length": "1/log(1+kappa/n_down)",
                "per_root_kappa_channel": (
                    "each value is the production additive search diagnostic "
                    "for one frozen root, reduced only within that root: "
                    "numeric-repair event means, completed numeric-path "
                    "means, and categorical-publication path counts"
                ),
                "local_derivative_validation": (
                    "the reference first-order cache movement scale is "
                    "abs(delta log kappa) times the reference root's mean "
                    "normalized-cache derivative norm across numeric repair "
                    "events; it is compared descriptively with observed "
                    "paired root-policy movement and is not a derivative of "
                    "the final root policy"
                ),
                "root_policy_top2_margin": (
                    "top probability minus second probability in the actual "
                    "ephemeral Q21 commitment policy; solved roots bypass "
                    "this diagnostic"
                ),
                "exact_regret": (
                    "(oracle root outcome - policy expected outcome)/2, "
                    "in [0,1]"
                ),
                "decisive_flip_regret": (
                    "1[candidate and reference top actions differ] * "
                    "max(candidate normalized top-action regret - reference "
                    "normalized top-action regret, 0); summaries report both "
                    "root-unweighted and corpus-root_weight-weighted "
                    "aggregates with explicit denominators"
                ),
            },
            "interpretation_limit": (
                "explanatory frozen-distribution sensitivity diagnostic; "
                "not an optimizer and not evidence of minimax Hex strength"
            ),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "step": args.checkpoint_step,
            **checkpoint_progress,
            "search_source": args.search_source,
            "stored_search_parameters": asdict(stored_search_config),
            "preserved_internal_posterior_policy_estimator": str(
                stored_search_config.posterior_policy_estimator
            ),
        },
        "corpus": {
            "path": str(args.corpus.resolve()),
            "manifest_sha256": _sha256(source_paths["manifest"]),
            "roots_sha256": _sha256(source_paths["roots"]),
            "manifest": manifest,
            "selected_root_count": len(selected_indices),
            "selected_indices_sha256": hashlib.sha256(
                selected_indices.astype(np.int64).tobytes()
            ).hexdigest(),
            "roots_per_stage_override": args.roots_per_stage,
        },
        "execution": {
            "command": shlex.join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "kappas": list(args.kappas),
            "elapsed_seconds_by_kappa_including_compile": elapsed_by_kappa,
            "tree_key_provenance": key_provenance,
            "external_logging": False,
        },
        "exact_oracle": {
            "enabled": args.exact_max_empties > 0,
            "maximum_empty_cells": args.exact_max_empties,
            "hard_safety_limit": MAX_EXACT_EMPTY_CELLS,
            "solved_positions": (
                int(np.sum(oracle["available"]))
                if oracle is not None
                else 0
            ),
        },
        "summary": summary,
        "positions": _position_rows(
            roots=roots,
            prior=prior,
            policies=policies,
            diagnostics=diagnostics,
            reference_kappa=args.reference_kappa,
            oracle_records=oracle_records,
        ),
    }
    digest = _write_immutable(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "roots": len(selected_indices),
                "kappas": list(args.kappas),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
