#!/usr/bin/env python3
"""Fit a power-temperature approximation to finite plurality commitment.

For a completed unresolved root, let ``q`` be the posterior-best action
population and let

    Phi_M(q)_a = P[a = lowest-index argmax(C), C ~ Multinomial(M, q)].

E11 samples from ``Phi_32(q_Q21)`` by drawing and counting 32 categorical
votes.  This benchmark asks whether the cheaper law

    Power_T(q)_a = q_a ** (1 / T) / sum_b q_b ** (1 / T)

is an adequate frozen-root substitute.  The temperature is selected on a
game-cluster-disjoint fitting split and evaluated on untouched validation and
test roots.  It also evaluates a posterior-action-mean score proxy to make
explicit why the input to the transform must be the winner population rather
than a repaired posterior mean.

The benchmark is standalone: it imports the existing E4/E11 replay and exact
plurality helpers but changes no search, action-commitment, or training code.
Categorical and Q21-guard-fallback roots retain E11's native bypass and are
counted rather than silently included in the fitted law.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import jax.numpy as jnp

from hex_plurality_fixed_root_benchmark import (
    STAGE_NAMES,
    _eligible_root_mask,
    _entropy,
    _extract_roots,
    _multinomial_calibration,
    _sanitize_policy,
    _scalar_summary,
    _weighted_mean,
    _weighted_quantile,
    exact_plurality_law,
)
from scacchi.dirichlet_mctx.estimator_diagnostics import (
    binary_posterior_best_policy_prefix_quadrature,
    binary_posterior_best_policy_quadrature,
)
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.posterior_updates import (
    DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE,
)
from scacchi.dirichlet_q_search import (
    posterior_plurality_action,
    posterior_sample_action,
)


def power_policy(policy: np.ndarray, temperature: float) -> np.ndarray:
    """Apply a support-preserving power transform in stable log space."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"temperature must be finite and positive, got {temperature}"
        )
    probability = np.asarray(policy, dtype=np.float64)
    one_dimensional = probability.ndim == 1
    if one_dimensional:
        probability = probability[None, :]
    if probability.ndim != 2:
        raise ValueError(
            "policy must have shape [root, action] or [action], got "
            f"{probability.shape}"
        )
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("policy must be finite and nonnegative")
    mass = np.sum(probability, axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("policy rows must have positive mass")
    probability = probability / mass
    positive = probability > 0.0
    log_probability = np.full_like(probability, -np.inf)
    log_probability[positive] = (
        np.log(probability[positive]) / temperature
    )
    row_max = np.max(log_probability, axis=-1, keepdims=True)
    powered = np.where(
        positive,
        np.exp(log_probability - row_max),
        0.0,
    )
    powered /= np.sum(powered, axis=-1, keepdims=True)
    return powered[0] if one_dimensional else powered


def _kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise KL(left || right) on their common finite support."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"KL shape mismatch: {left.shape} != {right.shape}")
    terms = np.zeros_like(left)
    positive = left > 0.0
    if np.any(positive & (right <= 0.0)):
        result = np.full(left.shape[:-1], np.inf, dtype=np.float64)
        safe_rows = ~np.any(positive & (right <= 0.0), axis=-1)
        local = positive & safe_rows[..., None]
        terms[local] = left[local] * (
            np.log(left[local]) - np.log(right[local])
        )
        result[safe_rows] = np.sum(terms[safe_rows], axis=-1)
        return result
    terms[positive] = left[positive] * (
        np.log(left[positive]) - np.log(right[positive])
    )
    return np.sum(terms, axis=-1)


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    middle = 0.5 * (left + right)
    return 0.5 * (_kl(left, middle) + _kl(right, middle))


def _comparison_arrays(
    target: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, np.ndarray]:
    difference = candidate - target
    l1 = np.sum(np.abs(difference), axis=-1)
    target_entropy = _entropy(target)
    candidate_entropy = _entropy(candidate)
    return {
        "l1": l1,
        "tv": 0.5 * l1,
        "forward_kl_nats": _kl(target, candidate),
        "reverse_kl_nats": _kl(candidate, target),
        "js_nats": _js(target, candidate),
        "top_action_agreement": (
            np.argmax(target, axis=-1)
            == np.argmax(candidate, axis=-1)
        ).astype(np.float64),
        "target_entropy_nats": target_entropy,
        "candidate_entropy_nats": candidate_entropy,
        "entropy_delta_nats": candidate_entropy - target_entropy,
        "target_ess": np.exp(target_entropy),
        "candidate_ess": np.exp(candidate_entropy),
    }


def _comparison_summary(
    target: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    arrays = _comparison_arrays(target, candidate)
    result: dict[str, Any] = {
        "roots": int(target.shape[0]),
        "weighted_root_mass": float(np.sum(weights)),
    }
    for name, values in arrays.items():
        if name == "top_action_agreement":
            result["top_action_agreement_weighted"] = _weighted_mean(
                values,
                weights,
            )
            result["top_action_agreement_unweighted"] = float(
                np.mean(values)
            )
        else:
            result[name] = _scalar_summary(values, weights)
    normalized_weight = weights / np.sum(weights)
    target_frequency = np.sum(
        normalized_weight[:, None] * target,
        axis=0,
    )
    candidate_frequency = np.sum(
        normalized_weight[:, None] * candidate,
        axis=0,
    )
    result["occupancy_weighted_action_frequency"] = {
        "tv": float(
            0.5 * np.sum(np.abs(target_frequency - candidate_frequency))
        ),
        "max_absolute_action_share_error": float(
            np.max(np.abs(target_frequency - candidate_frequency))
        ),
        "target": target_frequency.tolist(),
        "candidate": candidate_frequency.tolist(),
    }
    return result


def _cluster_fold(cluster_id: int, folds: int = 5) -> int:
    digest = hashlib.sha256(f"hex-root-cluster:{cluster_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little") % folds


def cluster_disjoint_split(
    cluster_id: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return 60/20/20 fit/validation/test masks grouped by game cluster."""

    fold = np.asarray(
        [_cluster_fold(int(value)) for value in cluster_id],
        dtype=np.int8,
    )
    return {
        "fit": fold <= 2,
        "validation": fold == 3,
        "test": fold == 4,
    }


def _temperature_grid(
    minimum: float,
    maximum: float,
    count: int,
) -> np.ndarray:
    if minimum <= 0.0 or maximum < minimum or count < 2:
        raise ValueError("invalid temperature grid")
    values = np.exp(np.linspace(np.log(minimum), np.log(maximum), count))
    if minimum <= 1.0 <= maximum:
        values = np.unique(np.concatenate((values, np.asarray([1.0]))))
    return values


def _fit_temperature(
    population: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    temperatures: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    curve: list[dict[str, float]] = []
    best_temperature = 0.0
    best_objective = math.inf
    for temperature in temperatures:
        candidate = power_policy(population, float(temperature))
        forward_kl = _kl(target, candidate)
        objective = _weighted_mean(forward_kl[mask], weights[mask])
        curve.append(
            {
                "temperature": float(temperature),
                "inverse_temperature": float(1.0 / temperature),
                "fit_forward_kl_nats": objective,
            }
        )
        if objective < best_objective:
            best_objective = objective
            best_temperature = float(temperature)
    return best_temperature, curve


def _posterior_mean_population(
    alpha: np.ndarray,
    invalid: np.ndarray,
    categorical_outcome: np.ndarray,
) -> np.ndarray:
    """Construct a documented action-mean score proxy for diagnosis only.

    This is not a posterior-best law.  Unresolved scores are posterior mean
    win probabilities; certified wins/losses are assigned one/zero.  Scores
    are normalized over legal actions only so a power transform can be fit.
    """

    alpha = np.asarray(alpha, dtype=np.float64)
    invalid = np.asarray(invalid, dtype=bool)
    categorical_outcome = np.asarray(categorical_outcome, dtype=np.int8)
    concentration = np.sum(alpha, axis=-1)
    unresolved_mean = np.divide(
        alpha[..., 1],
        concentration,
        out=np.zeros_like(concentration),
        where=concentration > 0.0,
    )
    unresolved = categorical_outcome == int(NO_OUTCOME)
    score = np.where(
        unresolved,
        unresolved_mean,
        (categorical_outcome == 1).astype(np.float64),
    )
    score = np.where(~invalid & np.isfinite(score) & (score > 0.0), score, 0.0)
    total = np.sum(score, axis=-1, keepdims=True)
    legal = ~invalid
    legal_count = np.sum(legal, axis=-1, keepdims=True)
    fallback = legal.astype(np.float64) / np.maximum(legal_count, 1)
    return np.where(total > 0.0, score / np.maximum(total, 1e-300), fallback)


def _per_root_optimum_summary(
    population: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    temperatures: np.ndarray,
) -> dict[str, Any]:
    objective = np.stack(
        [
            _kl(target, power_policy(population, float(temperature)))
            for temperature in temperatures
        ],
        axis=1,
    )
    best_index = np.argmin(objective, axis=1)
    best_temperature = temperatures[best_index]
    return {
        "temperature": _scalar_summary(best_temperature, weights),
        "inverse_temperature": _scalar_summary(
            1.0 / best_temperature,
            weights,
        ),
        "fit_grid_boundary_fraction": float(
            np.mean(
                (best_index == 0)
                | (best_index == len(temperatures) - 1)
            )
        ),
        "minimum_forward_kl_nats": _scalar_summary(
            objective[np.arange(len(target)), best_index],
            weights,
        ),
    }


def _binary_theory(num_votes: int) -> dict[str, Any]:
    """Report the binary local-slope approximation and the even-M tie defect."""

    if num_votes % 2:
        n = (num_votes - 1) // 2
        inverse_temperature = (
            num_votes
            * math.comb(num_votes - 1, n)
            / float(2 ** (num_votes - 1))
        )
        tie_probability = 0.0
        lower_index_probability_at_half = 0.5
    else:
        n = num_votes // 2
        tie_probability = math.comb(num_votes, n) / float(2**num_votes)
        # This is the slope of majority voting with a randomized tie.  A
        # lowest-index tie has the same first-order scale but a nonzero
        # intercept defect at q=1/2.
        inverse_temperature = (
            n * math.comb(num_votes, n) / float(2 ** (num_votes - 1))
        )
        lower_index_probability_at_half = 0.5 + 0.5 * tie_probability
    return {
        "num_votes": num_votes,
        "binary_local_slope_matched_inverse_temperature": (
            inverse_temperature
        ),
        "binary_local_slope_matched_temperature": (
            1.0 / inverse_temperature
        ),
        "equal_probability_count_tie_probability": tie_probability,
        "lower_index_win_probability_at_equal_q": (
            lower_index_probability_at_half
        ),
        "power_transform_win_probability_at_equal_q": 0.5,
        "exact_identity": False,
        "caveat": (
            "Even in two actions the binomial-majority CDF is not a logistic "
            "power law. For even M, lowest-index count ties create an "
            "intercept asymmetry that every pure power transform misses."
        ),
    }


def _jax_power_policy(
    policy: jax.Array,
    legal: jax.Array,
    temperature: float,
) -> jax.Array:
    positive = legal & jnp.isfinite(policy) & (policy > 0.0)
    logits = jnp.where(
        positive,
        jnp.log(jnp.where(positive, policy, 1.0)) / temperature,
        -jnp.inf,
    )
    return jax.nn.softmax(logits, axis=-1)


def _make_temperature_sampler(
    *,
    temperature: float,
    repetitions: int,
) -> Any:
    @jax.jit
    def sample(
        key: jax.Array,
        policy: jax.Array,
        legal: jax.Array,
    ) -> jax.Array:
        transformed = _jax_power_policy(policy, legal, temperature)
        keys = jax.random.split(key, repetitions)
        initial = jnp.zeros(policy.shape, dtype=jnp.int32)

        def one(counts, sample_key):
            action = posterior_sample_action(
                sample_key,
                transformed,
                legal,
            )
            return counts + jax.nn.one_hot(
                action,
                policy.shape[-1],
                dtype=jnp.int32,
            ), None

        return jax.lax.scan(one, initial, keys)[0]

    return sample


def _kernel_timing(
    policy: np.ndarray,
    legal: np.ndarray,
    *,
    temperature: float,
    num_votes: int,
    batch_size: int,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    repeats = int(math.ceil(batch_size / len(policy)))
    local_policy = jnp.asarray(
        np.tile(policy, (repeats, 1))[:batch_size],
        dtype=jnp.float32,
    )
    local_legal = jnp.asarray(
        np.tile(legal, (repeats, 1))[:batch_size],
    )

    @jax.jit
    def plurality(key, q, mask):
        return posterior_plurality_action(
            key,
            q,
            mask,
            num_samples=num_votes,
        )

    @jax.jit
    def powered(key, q, mask):
        transformed = _jax_power_policy(q, mask, temperature)
        return posterior_sample_action(key, transformed, mask)

    base_key = jax.random.PRNGKey(seed)
    jax.block_until_ready(plurality(base_key, local_policy, local_legal))
    jax.block_until_ready(powered(base_key, local_policy, local_legal))
    elapsed: dict[str, list[float]] = {"plurality": [], "power": []}
    for repetition in range(repetitions):
        order = (
            ("plurality", plurality, 1, "power", powered, 2)
            if repetition % 2 == 0
            else ("power", powered, 2, "plurality", plurality, 1)
        )
        for offset in (0, 3):
            name = order[offset]
            function = order[offset + 1]
            fold = order[offset + 2]
            key = jax.random.fold_in(base_key, repetition * 10 + fold)
            started = time.perf_counter()
            jax.block_until_ready(function(key, local_policy, local_legal))
            elapsed[name].append(time.perf_counter() - started)
    plurality_median = float(np.median(elapsed["plurality"]))
    power_median = float(np.median(elapsed["power"]))
    return {
        "backend": jax.default_backend(),
        "batch_size": batch_size,
        "paired_interleaved_repetitions": repetitions,
        "includes_power_transform_and_categorical_draw": True,
        "plurality32_seconds_median": plurality_median,
        "power_seconds_median": power_median,
        "power_over_plurality_ratio_of_medians": (
            power_median / plurality_median
        ),
        "warning": (
            "Kernel-only timing is not complete-search timing. CPU ratios "
            "must not be extrapolated numerically to GPU."
        ),
    }


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO / "checkpoints/hex6_prefix_q21_evidence_s0",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO / "experiments/e4/hex6_repair_contexts_v1",
    )
    parser.add_argument("--steps", default="0,50,100")
    parser.add_argument("--num-votes", type=int, default=32)
    parser.add_argument("--temperature-min", type=float, default=0.05)
    parser.add_argument("--temperature-max", type=float, default=2.0)
    parser.add_argument("--temperature-count", type=int, default=401)
    parser.add_argument("--validation-repetitions", type=int, default=512)
    parser.add_argument("--timing-batch-size", type=int, default=8192)
    parser.add_argument("--timing-repetitions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.num_votes < 1:
        parser.error("--num-votes must be positive")
    if args.validation_repetitions < 1:
        parser.error("--validation-repetitions must be positive")

    steps = tuple(int(item) for item in args.steps.split(","))
    started = time.perf_counter()
    arrays, extraction_seconds = _extract_roots(
        checkpoint=args.checkpoint,
        corpus=args.corpus,
        steps=steps,
        seed=args.seed,
    )
    alpha = jnp.asarray(arrays["alpha"], dtype=jnp.float32)
    invalid = jnp.asarray(arrays["invalid"])
    edge_outcome = jnp.asarray(arrays["edge_outcome"])
    root_categorical = arrays["node_outcome"] != int(NO_OUTCOME)

    q321_result = jax.block_until_ready(
        jax.jit(
            lambda a, inv, cat: binary_posterior_best_policy_quadrature(
                a,
                inv,
                cat,
                half_width=160,
                step=0.1,
            )
        )(alpha.astype(jnp.float64), invalid, edge_outcome)
    )
    q21_result = jax.block_until_ready(
        jax.jit(
            lambda a, inv, cat: (
                binary_posterior_best_policy_prefix_quadrature(
                    a,
                    inv,
                    cat,
                    half_width=10,
                    adaptive_range=True,
                    tail_scale=8.0,
                    min_half_range=6.0,
                    max_half_range=11.0,
                    mass_conserving=True,
                )
            )
        )(alpha, invalid, edge_outcome)
    )
    density_error = np.max(
        np.abs(np.asarray(q21_result.density_log_integral)),
        axis=-1,
    )
    unsafe = (
        np.asarray(q21_result.tail_range_clipped)
        | ~np.asarray(q21_result.finite)
        | (
            density_error
            > DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE
        )
    )
    eligible = _eligible_root_mask(root_categorical, unsafe)
    legal = ~arrays["invalid"][eligible]
    q321 = _sanitize_policy(
        np.asarray(q321_result.policy)[eligible],
        legal,
    )
    q21 = _sanitize_policy(
        np.asarray(q21_result.policy)[eligible],
        legal,
    )
    mean_population = _sanitize_policy(
        _posterior_mean_population(
            arrays["alpha"][eligible],
            arrays["invalid"][eligible],
            arrays["edge_outcome"][eligible],
        ),
        legal,
    )
    stacked_law, normalization_error = exact_plurality_law(
        np.concatenate((q21, q321), axis=0),
        num_votes=args.num_votes,
    )
    g21, g321 = np.split(stacked_law, 2, axis=0)
    weights = arrays["root_weight"][eligible].astype(np.float64)
    checkpoint_step = arrays["checkpoint_step"][eligible]
    stage_id = arrays["stage_id"][eligible]
    root_id = arrays["root_id"][eligible].astype(np.int64)

    roots_metadata = np.load(args.corpus / "roots.npz", allow_pickle=False)
    root_to_cluster = {
        int(identifier): int(cluster)
        for identifier, cluster in zip(
            roots_metadata["root_id"],
            roots_metadata["game_cluster_id"],
            strict=True,
        )
    }
    cluster_id = np.asarray(
        [root_to_cluster[int(identifier)] for identifier in root_id],
        dtype=np.int64,
    )
    split = cluster_disjoint_split(cluster_id)
    temperatures = _temperature_grid(
        args.temperature_min,
        args.temperature_max,
        args.temperature_count,
    )

    selected_temperature, q_curve = _fit_temperature(
        q21,
        g21,
        weights,
        split["fit"],
        temperatures,
    )
    mean_temperature, mean_curve = _fit_temperature(
        mean_population,
        g21,
        weights,
        split["fit"],
        temperatures,
    )
    q_candidate = power_policy(q21, selected_temperature)
    mean_candidate = power_policy(mean_population, mean_temperature)

    comparisons: dict[str, Any] = {}
    for split_name, mask in {
        "all": np.ones(len(q21), dtype=bool),
        **split,
    }.items():
        comparisons[split_name] = {
            "power_q21_vs_exact_plurality_q21": _comparison_summary(
                g21[mask],
                q_candidate[mask],
                weights[mask],
            ),
            "power_q21_vs_exact_plurality_q321": _comparison_summary(
                g321[mask],
                q_candidate[mask],
                weights[mask],
            ),
            "power_mean_proxy_vs_exact_plurality_q21": (
                _comparison_summary(
                    g21[mask],
                    mean_candidate[mask],
                    weights[mask],
                )
            ),
            "untransformed_q21_vs_exact_plurality_q21": (
                _comparison_summary(
                    g21[mask],
                    q21[mask],
                    weights[mask],
                )
            ),
        }
    groups: dict[str, Any] = {}
    for step in steps:
        for stage_index, stage_name in enumerate(STAGE_NAMES):
            mask = (
                (checkpoint_step == step)
                & (stage_id == stage_index)
            )
            if np.any(mask):
                groups[f"checkpoint_{step}_stage_{stage_name}"] = (
                    _comparison_summary(
                        g21[mask],
                        q_candidate[mask],
                        weights[mask],
                    )
                )

    validation_sampler = _make_temperature_sampler(
        temperature=selected_temperature,
        repetitions=args.validation_repetitions,
    )
    validation_counts = np.asarray(
        jax.block_until_ready(
            validation_sampler(
                jax.random.PRNGKey(args.seed + 10_001),
                jnp.asarray(q21, dtype=jnp.float32),
                jnp.asarray(legal),
            )
        )
    )
    calibration = _multinomial_calibration(
        validation_counts,
        q_candidate,
    )
    timing = _kernel_timing(
        q21,
        legal,
        temperature=selected_temperature,
        num_votes=args.num_votes,
        batch_size=args.timing_batch_size,
        repetitions=args.timing_repetitions,
        seed=args.seed + 20_001,
    )

    test_summary = comparisons["test"][
        "power_q21_vs_exact_plurality_q21"
    ]
    endpoint_l1 = comparisons["test"][
        "untransformed_q21_vs_exact_plurality_q21"
    ]["l1"]["weighted_mean"]
    gates = {
        "test_weighted_mean_l1_at_most_0_05": (
            test_summary["l1"]["weighted_mean"] <= 0.05
        ),
        "test_weighted_p95_l1_at_most_0_15": (
            test_summary["l1"]["weighted_p95"] <= 0.15
        ),
        "test_top_action_agreement_at_least_0_95": (
            test_summary["top_action_agreement_weighted"] >= 0.95
        ),
        "test_action_frequency_tv_at_most_0_01": (
            test_summary["occupancy_weighted_action_frequency"]["tv"]
            <= 0.01
        ),
        "test_mean_l1_at_most_25pct_untransformed_separation": (
            test_summary["l1"]["weighted_mean"] <= 0.25 * endpoint_l1
        ),
    }
    gates["all_pass"] = bool(all(gates.values()))

    report = {
        "format": "scacchi.hex_plurality_temperature_benchmark.v1",
        "reproduction": shlex.join([sys.executable, *sys.argv]),
        "backend": jax.default_backend(),
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "steps": list(steps),
        "protocol": {
            "target": (
                "exact lowest-index plurality law Phi_32(q_Q21) on one "
                "fixed completed unresolved root"
            ),
            "candidate": "q_Q21 ** (1/T), normalized, then one draw",
            "selection_objective": (
                "occupancy-weighted KL(Phi_32(q_Q21) || Power_T(q_Q21)) "
                "on the game-cluster-disjoint fit split"
            ),
            "split": (
                "SHA256(game_cluster_id) modulo five: folds 0--2 fit, "
                "fold 3 validation, fold 4 untouched test"
            ),
            "categorical_behavior": "native E11 categorical bypass retained",
            "guard_fallback_behavior": "native E11 M32 fallback retained",
            "posterior_mean_proxy": (
                "normalized posterior mean win scores; diagnostic only and "
                "not asserted to be a posterior-best action law"
            ),
        },
        "population": {
            "all_roots": int(len(root_categorical)),
            "eligible_unresolved_guard_accepted": int(np.sum(eligible)),
            "categorical_bypass": int(np.sum(root_categorical)),
            "q21_guard_fallback": int(np.sum(~root_categorical & unsafe)),
            "exact_plurality_normalization_error_max": float(
                np.max(normalization_error)
            ),
        },
        "split": {
            name: {
                "roots": int(np.sum(mask)),
                "clusters": int(np.unique(cluster_id[mask]).size),
                "weighted_root_mass": float(np.sum(weights[mask])),
            }
            for name, mask in split.items()
        },
        "temperature_grid": {
            "minimum": float(np.min(temperatures)),
            "maximum": float(np.max(temperatures)),
            "count": int(len(temperatures)),
        },
        "selected": {
            "q21_temperature": selected_temperature,
            "q21_inverse_temperature": 1.0 / selected_temperature,
            "posterior_mean_proxy_temperature": mean_temperature,
            "posterior_mean_proxy_inverse_temperature": (
                1.0 / mean_temperature
            ),
        },
        "fit_curve_best_20_q21": sorted(
            q_curve,
            key=lambda item: item["fit_forward_kl_nats"],
        )[:20],
        "fit_curve_best_20_posterior_mean_proxy": sorted(
            mean_curve,
            key=lambda item: item["fit_forward_kl_nats"],
        )[:20],
        "comparisons": comparisons,
        "groups": groups,
        "per_root_optimum": {
            "q21": _per_root_optimum_summary(
                q21,
                g21,
                weights,
                temperatures,
            ),
            "posterior_mean_proxy": _per_root_optimum_summary(
                mean_population,
                g21,
                weights,
                temperatures,
            ),
        },
        "binary_theory": _binary_theory(args.num_votes),
        "multiclass_caveat": (
            "Power_T obeys independence of irrelevant alternatives and "
            "preserves q's complete ranking. Phi_M does neither: a third "
            "competitor changes pairwise winning odds, and finite "
            "lowest-index count ties can change the modal action. A single "
            "temperature therefore has no exact multiclass interpretation."
        ),
        "sampling_validation": calibration,
        "kernel_timing": timing,
        "e11_substitution_gates": gates,
        "timing_seconds": {
            "root_extraction_by_step": extraction_seconds,
            "total": time.perf_counter() - started,
        },
        "local_only": True,
    }
    serialized = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
