#!/usr/bin/env python3
"""Fixed-root benchmark for finite-population plurality commitment.

The benchmark never changes or reruns a completed tree after extraction.  On
each unresolved root whose production Q21 guards pass, it compares:

* the exact plurality law ``g_M(q_Q321)`` of the float64 Q321 reference;
* the exact plurality law ``g_M(q_Q21)`` of guarded production Q21; and
* ``g_M(q_impl_hat)``, where ``q_impl_hat`` is estimated with independent
  draws from the production bounded-work Thompson primitive.

``g_M`` itself is evaluated deterministically by the generating-function
coefficient formula, including the production lowest-index count-tie rule.
Separate direct Monte Carlo checks exercise production native
winner-MC-plus-argmax and ``posterior_plurality``.  Categorical roots and Q21
guard-fallback roots are counted and excluded from the conditional-law
comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import chi2


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import jax.numpy as jnp

from e4_repair_context_corpus import (
    _load_checkpoint_config_and_model,
    _replay_roots,
)
from e8_fixed_tree_root_readout_benchmark import (
    _make_tree_extractor,
    _parse_steps,
    _subset,
)
from scacchi.dirichlet_mctx import action_selection
from scacchi.dirichlet_mctx.estimator_diagnostics import (
    binary_posterior_best_policy_prefix_quadrature,
    binary_posterior_best_policy_quadrature,
)
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.posterior_updates import (
    DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE,
)
from scacchi.dirichlet_q_search import posterior_plurality_action


STAGE_NAMES = ("early", "mid", "late")


def _eligible_root_mask(
    root_categorical: np.ndarray,
    q21_unsafe: np.ndarray,
) -> np.ndarray:
    """Select exactly unresolved roots whose Q21 action readout is accepted."""

    root_categorical = np.asarray(root_categorical, dtype=bool)
    q21_unsafe = np.asarray(q21_unsafe, dtype=bool)
    if root_categorical.shape != q21_unsafe.shape:
        raise ValueError(
            "categorical/unsafe shape mismatch: "
            f"{root_categorical.shape} != {q21_unsafe.shape}"
        )
    return ~root_categorical & ~q21_unsafe


def _sanitize_policy(
    policy: np.ndarray,
    legal: np.ndarray,
) -> np.ndarray:
    """Mirror the production plurality input normalization on the host."""

    policy = np.asarray(policy, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    if policy.ndim == 1:
        policy = policy[None, :]
    if legal.ndim == 1:
        legal = legal[None, :]
    if policy.shape != legal.shape:
        raise ValueError(
            f"policy/legal shape mismatch: {policy.shape} != {legal.shape}"
        )
    positive = legal & np.isfinite(policy) & (policy > 0.0)
    normalized = np.where(positive, policy, 0.0)
    total = np.sum(normalized, axis=-1, keepdims=True)
    legal_count = np.sum(legal, axis=-1, keepdims=True)
    fallback = np.divide(
        legal.astype(np.float64),
        np.maximum(legal_count, 1),
    )
    no_legal = np.zeros_like(policy)
    no_legal[:, 0] = 1.0
    fallback = np.where(legal_count > 0, fallback, no_legal)
    return np.where(total > 0.0, normalized / np.maximum(total, 1e-300), fallback)


def _multiply_truncated(
    polynomial: np.ndarray,
    factor: np.ndarray,
) -> np.ndarray:
    """Multiply batched polynomials, retaining the input maximum degree."""

    degree = polynomial.shape[-1] - 1
    output = np.zeros_like(polynomial)
    for offset in range(min(factor.shape[-1] - 1, degree) + 1):
        output[:, offset:] += (
            polynomial[:, : degree + 1 - offset]
            * factor[:, offset, None]
        )
    return output


def exact_plurality_law(
    policy: np.ndarray,
    *,
    num_votes: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``g_M(policy)`` exactly by exponential generating functions.

    For candidate action ``a`` to win with count ``m``, every lower-index
    action must have count at most ``m - 1`` and every higher-index action may
    have count at most ``m``.  The coefficient calculation is therefore the
    exact finite multinomial sum, not a Monte Carlo approximation.

    Returns the normalized law and each row's absolute raw normalization
    error.  The latter should contain only floating-point roundoff.
    """

    if num_votes < 1:
        raise ValueError(f"num_votes must be >= 1, got {num_votes}")
    probability = np.asarray(policy, dtype=np.float64)
    one_dimensional = probability.ndim == 1
    if one_dimensional:
        probability = probability[None, :]
    if probability.ndim != 2:
        raise ValueError(
            "policy must have shape [root, action] or [action], got "
            f"{probability.shape}"
        )
    if not np.all(np.isfinite(probability)):
        raise ValueError("policy contains a non-finite value")
    if np.any(probability < 0.0):
        raise ValueError("policy contains a negative value")
    mass = np.sum(probability, axis=-1)
    if not np.allclose(mass, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(
            "policy rows must sum to one; maximum error is "
            f"{float(np.max(np.abs(mass - 1.0)))}"
        )
    probability = probability / mass[:, None]

    roots, actions = probability.shape
    coefficient = np.empty(
        (roots, actions, num_votes + 1),
        dtype=np.float64,
    )
    coefficient[:, :, 0] = 1.0
    for count in range(1, num_votes + 1):
        coefficient[:, :, count] = (
            coefficient[:, :, count - 1]
            * probability
            / float(count)
        )

    law = np.zeros_like(probability)
    multinomial_factor = float(math.factorial(num_votes))
    for winner_count in range(1, num_votes + 1):
        remaining = num_votes - winner_count

        # prefix[:, a] is the product over j < a with counts <= m - 1.
        prefix = np.zeros(
            (roots, actions + 1, remaining + 1),
            dtype=np.float64,
        )
        prefix[:, 0, 0] = 1.0
        lower_degree = min(winner_count - 1, remaining)
        for action in range(actions):
            prefix[:, action + 1, :] = _multiply_truncated(
                prefix[:, action, :],
                coefficient[:, action, : lower_degree + 1],
            )

        # suffix[:, a] is the product over j >= a with counts <= m.
        suffix = np.zeros_like(prefix)
        suffix[:, actions, 0] = 1.0
        upper_degree = min(winner_count, remaining)
        for action in range(actions - 1, -1, -1):
            suffix[:, action, :] = _multiply_truncated(
                suffix[:, action + 1, :],
                coefficient[:, action, : upper_degree + 1],
            )

        for action in range(actions):
            other_coefficient = np.sum(
                prefix[:, action, :]
                * suffix[:, action + 1, ::-1],
                axis=-1,
            )
            law[:, action] += (
                multinomial_factor
                * coefficient[:, action, winner_count]
                * other_coefficient
            )

    raw_mass = np.sum(law, axis=-1)
    normalization_error = np.abs(raw_mass - 1.0)
    if not np.all(np.isfinite(law)):
        raise FloatingPointError("exact plurality calculation was non-finite")
    if np.any(raw_mass <= 0.0):
        raise FloatingPointError("exact plurality calculation has zero mass")
    if float(np.max(normalization_error)) > 1e-8:
        raise FloatingPointError(
            "exact plurality normalization error exceeds 1e-8: "
            f"{float(np.max(normalization_error))}"
        )
    law = np.maximum(law, 0.0) / raw_mass[:, None]
    if one_dimensional:
        return law[0], normalization_error[0]
    return law, normalization_error


def _entropy(policy: np.ndarray) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64)
    terms = np.zeros_like(policy)
    positive = policy > 0.0
    terms[positive] = policy[positive] * np.log(policy[positive])
    return -np.sum(terms, axis=-1)


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (left + right)

    def kl(probability: np.ndarray, reference: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(probability)
        positive = probability > 0.0
        terms[positive] = probability[positive] * (
            np.log(probability[positive])
            - np.log(reference[positive])
        )
        return np.sum(terms, axis=-1)

    return 0.5 * (kl(left, middle) + kl(right, middle))


def _comparison_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, np.ndarray]:
    difference = np.asarray(candidate) - np.asarray(reference)
    l1 = np.sum(np.abs(difference), axis=-1)
    reference_entropy = _entropy(reference)
    candidate_entropy = _entropy(candidate)
    return {
        "l1": l1,
        "tv": 0.5 * l1,
        "js_nats": _js(reference, candidate),
        "top_action_agreement": (
            np.argmax(reference, axis=-1)
            == np.argmax(candidate, axis=-1)
        ),
        "reference_entropy_nats": reference_entropy,
        "candidate_entropy_nats": candidate_entropy,
        "reference_ess": np.exp(reference_entropy),
        "candidate_ess": np.exp(candidate_entropy),
        "entropy_delta_nats": candidate_entropy - reference_entropy,
        "ess_delta": (
            np.exp(candidate_entropy) - np.exp(reference_entropy)
        ),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        weights = np.broadcast_to(weights, values.shape)
    denominator = float(np.sum(weights))
    if denominator <= 0.0:
        raise ValueError("weighted mean requires positive total weight")
    return float(np.sum(values * weights) / denominator)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"invalid quantile {quantile}")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 0.0:
        raise ValueError("weighted quantile requires positive total weight")
    index = int(
        np.searchsorted(
            cumulative,
            quantile * cumulative[-1],
            side="left",
        )
    )
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _scalar_summary(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "weighted_mean": _weighted_mean(values, weights),
        "unweighted_mean": float(np.mean(values)),
        "weighted_p50": _weighted_quantile(values, weights, 0.50),
        "weighted_p95": _weighted_quantile(values, weights, 0.95),
        "unweighted_p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _comparison_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    arrays = _comparison_arrays(reference, candidate)
    return {
        "roots": int(reference.shape[0]),
        "l1": _scalar_summary(arrays["l1"], weights),
        "tv": _scalar_summary(arrays["tv"], weights),
        "js_nats": _scalar_summary(arrays["js_nats"], weights),
        "top_action_agreement_weighted": _weighted_mean(
            arrays["top_action_agreement"],
            weights,
        ),
        "top_action_agreement_unweighted": float(
            np.mean(arrays["top_action_agreement"])
        ),
        "reference_entropy_nats": _scalar_summary(
            arrays["reference_entropy_nats"],
            weights,
        ),
        "candidate_entropy_nats": _scalar_summary(
            arrays["candidate_entropy_nats"],
            weights,
        ),
        "reference_ess": _scalar_summary(
            arrays["reference_ess"],
            weights,
        ),
        "candidate_ess": _scalar_summary(
            arrays["candidate_ess"],
            weights,
        ),
        "entropy_delta_nats": _scalar_summary(
            arrays["entropy_delta_nats"],
            weights,
        ),
        "ess_delta": _scalar_summary(
            arrays["ess_delta"],
            weights,
        ),
        "optimal_coupling_action_agreement_weighted": (
            1.0 - _weighted_mean(arrays["tv"], weights)
        ),
    }


def _bootstrap_weighted_means(
    arrays: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> dict[str, list[float]]:
    if repetitions < 1:
        return {}
    rng = np.random.default_rng(seed)
    roots = len(weights)
    draws: dict[str, np.ndarray] = {
        name: np.empty(repetitions, dtype=np.float64)
        for name in arrays
    }
    for repetition in range(repetitions):
        index = rng.integers(0, roots, size=roots)
        local_weight = weights[index]
        for name, values in arrays.items():
            draws[name][repetition] = _weighted_mean(
                np.asarray(values)[index],
                local_weight,
            )
    return {
        name: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for name, values in draws.items()
    }


def _multinomial_calibration(
    counts: np.ndarray,
    expected_policy: np.ndarray,
) -> dict[str, Any]:
    """Pearson calibration after pooling cells with expectation below five."""

    counts = np.asarray(counts, dtype=np.int64)
    expected_policy = np.asarray(expected_policy, dtype=np.float64)
    repetitions = np.sum(counts, axis=-1)
    if not np.all(repetitions == repetitions[0]):
        raise ValueError("calibration requires equal repetitions per root")
    sample_count = int(repetitions[0])
    statistic = 0.0
    degrees = 0
    impossible_observations = 0
    maximum_abs_z = 0.0
    for observed, policy in zip(counts, expected_policy, strict=True):
        expected = sample_count * policy
        frequent = expected >= 5.0
        observed_cells = observed[frequent].astype(np.float64).tolist()
        expected_cells = expected[frequent].tolist()
        rare_observed = float(np.sum(observed[~frequent]))
        rare_expected = float(np.sum(expected[~frequent]))
        if rare_expected > 0.0:
            observed_cells.append(rare_observed)
            expected_cells.append(rare_expected)
        elif rare_observed > 0.0:
            impossible_observations += int(rare_observed)
        if len(expected_cells) >= 2:
            observed_array = np.asarray(observed_cells)
            expected_array = np.asarray(expected_cells)
            statistic += float(
                np.sum(
                    (observed_array - expected_array) ** 2
                    / expected_array
                )
            )
            degrees += len(expected_cells) - 1
        variance = sample_count * policy * (1.0 - policy)
        supported = variance > 0.0
        if np.any(supported):
            maximum_abs_z = max(
                maximum_abs_z,
                float(
                    np.max(
                        np.abs(observed[supported] - expected[supported])
                        / np.sqrt(variance[supported])
                    )
                ),
            )
    p_value = (
        float(chi2.sf(statistic, degrees))
        if degrees > 0 and impossible_observations == 0
        else 0.0
    )
    return {
        "repetitions_per_root": sample_count,
        "pearson_chi_square": statistic,
        "degrees_of_freedom": degrees,
        "chi_square_per_degree": (
            statistic / degrees if degrees > 0 else 0.0
        ),
        "asymptotic_p_value": p_value,
        "cells_with_expected_below_5_pooled_per_root": True,
        "impossible_observations_under_supplied_policy": (
            impossible_observations
        ),
        "maximum_absolute_unpooled_binomial_z": maximum_abs_z,
    }


def _make_native_group_sampler(
    *,
    num_groups: int,
    num_votes: int,
) -> Any:
    """Build a compiled production-winner/grouped-plurality sampler."""

    if num_groups < 1 or num_votes < 1:
        raise ValueError("num_groups and num_votes must be >= 1")

    @jax.jit
    def sample(
        key: jax.Array,
        alpha: jax.Array,
        invalid: jax.Array,
        edge_outcome: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        action_count = alpha.shape[-2]
        group_keys = jax.random.split(key, num_groups)
        initial = (
            jnp.zeros(alpha.shape[:-1], dtype=jnp.int32),
            jnp.zeros(alpha.shape[:-1], dtype=jnp.int32),
        )

        def sample_group(carry, group_key):
            vote_keys = jax.random.split(group_key, num_votes)
            winners = jax.vmap(
                lambda vote_key: action_selection.thompson_sample(
                    vote_key,
                    alpha,
                    invalid,
                    edge_outcome,
                )
            )(vote_keys)
            winner_count = jnp.sum(
                jax.nn.one_hot(
                    winners,
                    action_count,
                    dtype=jnp.int32,
                ),
                axis=0,
            ).astype(jnp.int32)
            plurality = jnp.argmax(winner_count, axis=-1).astype(jnp.int32)
            plurality_count = jax.nn.one_hot(
                plurality,
                action_count,
                dtype=jnp.int32,
            ).astype(jnp.int32)
            return (
                carry[0] + winner_count,
                carry[1] + plurality_count,
            ), None

        counts, _ = jax.lax.scan(sample_group, initial, group_keys)
        return counts

    return sample


def _make_q21_plurality_sampler(
    *,
    repetitions: int,
    num_votes: int,
) -> Any:
    """Build a compiled sampler for production posterior_plurality."""

    if repetitions < 1 or num_votes < 1:
        raise ValueError("repetitions and num_votes must be >= 1")

    @jax.jit
    def sample(
        key: jax.Array,
        policy: jax.Array,
        legal: jax.Array,
    ) -> jax.Array:
        keys = jax.random.split(key, repetitions)
        initial = jnp.zeros(policy.shape, dtype=jnp.int32)

        def sample_once(counts, sample_key):
            action = posterior_plurality_action(
                sample_key,
                policy,
                legal,
                num_samples=num_votes,
            )
            return counts + jax.nn.one_hot(
                action,
                policy.shape[-1],
                dtype=jnp.int32,
            ), None

        counts, _ = jax.lax.scan(sample_once, initial, keys)
        return counts

    return sample


def _extract_roots(
    *,
    checkpoint: Path,
    corpus: Path,
    steps: tuple[int, ...],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    roots = dict(np.load(corpus / "roots.npz", allow_pickle=False))
    names = (
        "prior",
        "alpha",
        "invalid",
        "node_outcome",
        "edge_outcome",
        "edge_distance",
        "native_m32",
        "native_action",
        "simulation_active_count",
        "executed_simulation_call_count",
    )
    chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            *names,
            "checkpoint_step",
            "stage_id",
            "root_weight",
            "root_id",
        )
    }
    base_key = jax.random.PRNGKey(seed)
    extraction_seconds: dict[str, float] = {}
    for ordinal, step in enumerate(steps):
        config, env, model, _ = _load_checkpoint_config_and_model(
            checkpoint,
            step,
        )
        search_config = config.selfplay.search.dirichlet_thompson
        indices = np.flatnonzero(roots["checkpoint_step"] == step)
        if len(indices) == 0:
            raise ValueError(f"no corpus roots for checkpoint step {step}")
        state = _replay_roots(env, _subset(roots, indices))
        extract = _make_tree_extractor(env, search_config)
        started = time.perf_counter()
        output = jax.block_until_ready(
            extract(model, state, jax.random.fold_in(base_key, ordinal))
        )
        extraction_seconds[str(step)] = time.perf_counter() - started
        for name, value in zip(names, output, strict=True):
            chunks[name].append(np.asarray(jax.device_get(value)))
        for name in (
            "checkpoint_step",
            "stage_id",
            "root_weight",
            "root_id",
        ):
            chunks[name].append(roots[name][indices])
        del model, extract
    return {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }, extraction_seconds


def _json_scalar(value: Any) -> int | float | str | bool:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _build_group_summary(
    mask: np.ndarray,
    *,
    weights: np.ndarray,
    q321: np.ndarray,
    q21: np.ndarray,
    q_impl: np.ndarray,
    g321: np.ndarray,
    g21: np.ndarray,
    g_impl: np.ndarray,
    g1: np.ndarray,
    deterministic_argmax: np.ndarray,
    g_half_a: np.ndarray,
    g_half_b: np.ndarray,
) -> dict[str, Any]:
    local_weight = weights[mask]
    return {
        "roots": int(np.sum(mask)),
        "weighted_root_mass": float(np.sum(local_weight)),
        "q21_population_vs_q321": _comparison_summary(
            q321[mask],
            q21[mask],
            local_weight,
        ),
        "q_impl_estimate_vs_q321": _comparison_summary(
            q321[mask],
            q_impl[mask],
            local_weight,
        ),
        "plurality_q21_vs_q321": _comparison_summary(
            g321[mask],
            g21[mask],
            local_weight,
        ),
        "plurality_q_impl_estimate_vs_q321": _comparison_summary(
            g321[mask],
            g_impl[mask],
            local_weight,
        ),
        "plurality_q21_vs_q_impl_estimate": _comparison_summary(
            g_impl[mask],
            g21[mask],
            local_weight,
        ),
        "plurality_split_half_native_population": _comparison_summary(
            g_half_a[mask],
            g_half_b[mask],
            local_weight,
        ),
        "endpoint_geometry": {
            "g32_q321_vs_g1_q321": _comparison_summary(
                g1[mask],
                g321[mask],
                local_weight,
            ),
            "g32_q321_vs_deterministic_argmax_q321": _comparison_summary(
                deterministic_argmax[mask],
                g321[mask],
                local_weight,
            ),
        },
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
    parser.add_argument(
        "--population-groups-per-half",
        type=int,
        default=512,
        help=(
            "Each group contains num-votes production Thompson winners; "
            "two independent halves estimate q_impl."
        ),
    )
    parser.add_argument(
        "--validation-repetitions",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=5_000,
    )
    parser.add_argument("--max-roots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.num_votes < 1:
        parser.error("--num-votes must be >= 1")
    if args.population_groups_per_half < 1:
        parser.error("--population-groups-per-half must be >= 1")
    if args.validation_repetitions < 1:
        parser.error("--validation-repetitions must be >= 1")
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap-repetitions must be >= 0")
    if args.max_roots is not None and args.max_roots < 1:
        parser.error("--max-roots must be >= 1")

    started_total = time.perf_counter()
    steps = _parse_steps(args.steps)
    arrays, extraction_seconds = _extract_roots(
        checkpoint=args.checkpoint,
        corpus=args.corpus,
        steps=steps,
        seed=args.seed,
    )
    if args.max_roots is not None:
        arrays = {
            name: values[: args.max_roots]
            for name, values in arrays.items()
        }

    alpha = jnp.asarray(arrays["alpha"], dtype=jnp.float32)
    invalid = jnp.asarray(arrays["invalid"])
    edge_outcome = jnp.asarray(arrays["edge_outcome"])
    root_categorical = (
        arrays["node_outcome"] != int(NO_OUTCOME)
    )

    started_reference = time.perf_counter()
    reference_result = jax.block_until_ready(
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
    reference_seconds = time.perf_counter() - started_reference

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
    if not np.any(eligible):
        raise ValueError("no unresolved guard-accepted roots")

    legal = ~arrays["invalid"][eligible]
    q321 = _sanitize_policy(
        np.asarray(reference_result.policy)[eligible],
        legal,
    )
    q21 = _sanitize_policy(
        np.asarray(q21_result.policy)[eligible],
        legal,
    )
    local_alpha = alpha[eligible]
    local_invalid = invalid[eligible]
    local_edge_outcome = edge_outcome[eligible]
    base_key = jax.random.PRNGKey(args.seed)

    started_population = time.perf_counter()
    population_sampler = _make_native_group_sampler(
        num_groups=args.population_groups_per_half,
        num_votes=args.num_votes,
    )
    winner_a, _ = jax.block_until_ready(
        population_sampler(
            jax.random.fold_in(base_key, 10_001),
            local_alpha,
            local_invalid,
            local_edge_outcome,
        )
    )
    winner_b, _ = jax.block_until_ready(
        population_sampler(
            jax.random.fold_in(base_key, 10_002),
            local_alpha,
            local_invalid,
            local_edge_outcome,
        )
    )
    population_seconds = time.perf_counter() - started_population
    samples_per_half = (
        args.population_groups_per_half * args.num_votes
    )
    q_impl_half_a = _sanitize_policy(
        np.asarray(winner_a, dtype=np.float64) / samples_per_half,
        legal,
    )
    q_impl_half_b = _sanitize_policy(
        np.asarray(winner_b, dtype=np.float64) / samples_per_half,
        legal,
    )
    q_impl = 0.5 * (q_impl_half_a + q_impl_half_b)

    started_exact = time.perf_counter()
    stacked_population = np.concatenate(
        (q321, q21, q_impl, q_impl_half_a, q_impl_half_b),
        axis=0,
    )
    stacked_law, law_normalization_error = exact_plurality_law(
        stacked_population,
        num_votes=args.num_votes,
    )
    root_count = len(q321)
    (
        g321,
        g21,
        g_impl,
        g_half_a,
        g_half_b,
    ) = np.split(
        stacked_law,
        np.arange(1, 5) * root_count,
        axis=0,
    )
    exact_seconds = time.perf_counter() - started_exact

    g1 = q321
    deterministic_argmax = np.eye(
        q321.shape[-1],
        dtype=np.float64,
    )[
        np.argmax(q321, axis=-1)
    ]
    weights = arrays["root_weight"][eligible].astype(np.float64)
    checkpoint_step = arrays["checkpoint_step"][eligible]
    stage_id = arrays["stage_id"][eligible]
    root_id = arrays["root_id"][eligible]

    started_validation = time.perf_counter()
    native_validation_sampler = _make_native_group_sampler(
        num_groups=args.validation_repetitions,
        num_votes=args.num_votes,
    )
    _, native_validation_count = jax.block_until_ready(
        native_validation_sampler(
            jax.random.fold_in(base_key, 20_001),
            local_alpha,
            local_invalid,
            local_edge_outcome,
        )
    )
    q21_validation_sampler = _make_q21_plurality_sampler(
        repetitions=args.validation_repetitions,
        num_votes=args.num_votes,
    )
    q21_validation_count = jax.block_until_ready(
        q21_validation_sampler(
            jax.random.fold_in(base_key, 20_002),
            jnp.asarray(q21, dtype=jnp.float32),
            jnp.asarray(legal),
        )
    )
    validation_seconds = time.perf_counter() - started_validation
    native_validation_count_np = np.asarray(native_validation_count)
    q21_validation_count_np = np.asarray(q21_validation_count)
    native_empirical = (
        native_validation_count_np / args.validation_repetitions
    )
    q21_empirical = (
        q21_validation_count_np / args.validation_repetitions
    )

    comparison_main = _comparison_arrays(g321, g21)
    comparison_qimpl_q21 = _comparison_arrays(g_impl, g21)
    split_population = _comparison_arrays(
        q_impl_half_a,
        q_impl_half_b,
    )
    split_law = _comparison_arrays(g_half_a, g_half_b)
    full_population_samples = 2 * samples_per_half
    expected_q_impl_squared_l2_mc = (
        1.0 - np.sum(q_impl * q_impl, axis=-1)
    ) / max(full_population_samples - 1, 1)
    first_order_law_tv_mc_proxy = 0.5 * split_law["tv"]
    signal_to_mc_proxy = (
        comparison_qimpl_q21["tv"]
        / np.maximum(first_order_law_tv_mc_proxy, 1e-12)
    )

    all_mask = np.ones(root_count, dtype=bool)
    groups = {
        "all_eligible": _build_group_summary(
            all_mask,
            weights=weights,
            q321=q321,
            q21=q21,
            q_impl=q_impl,
            g321=g321,
            g21=g21,
            g_impl=g_impl,
            g1=g1,
            deterministic_argmax=deterministic_argmax,
            g_half_a=g_half_a,
            g_half_b=g_half_b,
        )
    }
    for step in steps:
        step_mask = checkpoint_step == step
        if np.any(step_mask):
            groups[f"checkpoint_{step}"] = _build_group_summary(
                step_mask,
                weights=weights,
                q321=q321,
                q21=q21,
                q_impl=q_impl,
                g321=g321,
                g21=g21,
                g_impl=g_impl,
                g1=g1,
                deterministic_argmax=deterministic_argmax,
                g_half_a=g_half_a,
                g_half_b=g_half_b,
            )
        for stage_index, stage_name in enumerate(STAGE_NAMES):
            mask = step_mask & (stage_id == stage_index)
            if np.any(mask):
                groups[
                    f"checkpoint_{step}_stage_{stage_name}"
                ] = _build_group_summary(
                    mask,
                    weights=weights,
                    q321=q321,
                    q21=q21,
                    q_impl=q_impl,
                    g321=g321,
                    g21=g21,
                    g_impl=g_impl,
                    g1=g1,
                    deterministic_argmax=deterministic_argmax,
                    g_half_a=g_half_a,
                    g_half_b=g_half_b,
                )

    main_summary = groups["all_eligible"]["plurality_q21_vs_q321"]
    endpoint_g1_l1 = groups["all_eligible"]["endpoint_geometry"][
        "g32_q321_vs_g1_q321"
    ]["l1"]["weighted_mean"]
    endpoint_argmax_l1 = groups["all_eligible"][
        "endpoint_geometry"
    ][
        "g32_q321_vs_deterministic_argmax_q321"
    ]["l1"]["weighted_mean"]
    main_mean_l1 = main_summary["l1"]["weighted_mean"]
    gates = {
        "mean_l1_at_most_0_05": main_mean_l1 <= 0.05,
        "weighted_p95_l1_at_most_0_15": (
            main_summary["l1"]["weighted_p95"] <= 0.15
        ),
        "law_mode_agreement_at_least_0_95": (
            main_summary["top_action_agreement_weighted"] >= 0.95
        ),
        "mean_l1_at_most_10pct_g1_separation": (
            main_mean_l1 <= 0.10 * endpoint_g1_l1
        ),
        "mean_l1_at_most_10pct_deterministic_argmax_separation": (
            main_mean_l1 <= 0.10 * endpoint_argmax_l1
        ),
    }
    gates["all_pass"] = bool(all(gates.values()))

    bootstrap = _bootstrap_weighted_means(
        {
            "plurality_q21_vs_q321_l1": comparison_main["l1"],
            "plurality_q21_vs_q321_tv": comparison_main["tv"],
            "plurality_q21_vs_q321_js_nats": (
                comparison_main["js_nats"]
            ),
            "plurality_q21_vs_q321_top_action_agreement": (
                comparison_main["top_action_agreement"]
            ),
        },
        weights,
        seed=args.seed + 30_001,
        repetitions=args.bootstrap_repetitions,
    )

    per_root = []
    for index in range(root_count):
        per_root.append(
            {
                "root_id": _json_scalar(root_id[index]),
                "checkpoint_step": int(checkpoint_step[index]),
                "stage": STAGE_NAMES[int(stage_id[index])],
                "weight": float(weights[index]),
                "q21_vs_q321_population_l1": float(
                    np.sum(np.abs(q21[index] - q321[index]))
                ),
                "plurality_q21_vs_q321_l1": float(
                    comparison_main["l1"][index]
                ),
                "plurality_q21_vs_q321_tv": float(
                    comparison_main["tv"][index]
                ),
                "plurality_q21_vs_q321_js_nats": float(
                    comparison_main["js_nats"][index]
                ),
                "plurality_q21_vs_q321_top_action_agreement": bool(
                    comparison_main["top_action_agreement"][index]
                ),
                "plurality_q_impl_vs_q21_l1": float(
                    comparison_qimpl_q21["l1"][index]
                ),
                "plurality_q_impl_vs_q21_tv": float(
                    comparison_qimpl_q21["tv"][index]
                ),
                "plurality_q_impl_vs_q21_js_nats": float(
                    comparison_qimpl_q21["js_nats"][index]
                ),
                "plurality_q_impl_vs_q21_top_action_agreement": bool(
                    comparison_qimpl_q21[
                        "top_action_agreement"
                    ][index]
                ),
                "plurality_q321_entropy_nats": float(
                    _entropy(g321[index])[()]
                ),
                "plurality_q21_entropy_nats": float(
                    _entropy(g21[index])[()]
                ),
                "plurality_q_impl_entropy_nats": float(
                    _entropy(g_impl[index])[()]
                ),
                "plurality_q321_ess": float(
                    np.exp(_entropy(g321[index]))[()]
                ),
                "plurality_q21_ess": float(
                    np.exp(_entropy(g21[index]))[()]
                ),
                "plurality_q_impl_ess": float(
                    np.exp(_entropy(g_impl[index]))[()]
                ),
                "native_population_split_half_tv": float(
                    split_population["tv"][index]
                ),
                "plurality_split_half_tv": float(
                    split_law["tv"][index]
                ),
                "plurality_full_mc_tv_first_order_proxy": float(
                    first_order_law_tv_mc_proxy[index]
                ),
                "q_impl_vs_q21_signal_to_mc_proxy": float(
                    signal_to_mc_proxy[index]
                ),
                "q321_top_action": int(np.argmax(q321[index])),
                "q21_top_action": int(np.argmax(q21[index])),
                "plurality_q321_top_action": int(
                    np.argmax(g321[index])
                ),
                "plurality_q21_top_action": int(np.argmax(g21[index])),
                "plurality_q_impl_top_action": int(
                    np.argmax(g_impl[index])
                ),
            }
        )

    report = {
        "format": "scacchi.hex_plurality_fixed_root_benchmark.v1",
        "reproduction": shlex.join([sys.executable, *sys.argv]),
        "backend": jax.default_backend(),
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "steps": list(steps),
        "num_votes": args.num_votes,
        "protocol": {
            "estimand": (
                "conditional action law on one fixed completed root; no "
                "readout reruns or mutates tree traversal/repair"
            ),
            "reference": (
                "float64 Q321 exact-Beta quadrature population followed by "
                "deterministic exact generating-function g_M"
            ),
            "candidate": (
                "guard-accepted production float32 mass-conserving Q21 "
                "population followed by the same exact g_M"
            ),
            "production_population": (
                "two independent estimates from the bounded-work production "
                "Thompson primitive; each is grouped only to preserve an "
                "independent split-half uncertainty audit"
            ),
            "tie_rule": (
                "action a wins count m iff all lower-index counts are <m and "
                "all higher-index counts are <=m"
            ),
            "exclusions": (
                "categorical roots and unresolved Q21 guard-fallback roots "
                "are excluded; they retain their documented native bypass"
            ),
            "uncertainty": (
                "g_M is exact conditional on its supplied q. Q21 is "
                "deterministic; q_impl is sampled. Split-half disagreement "
                "and the analytic multinomial q_impl MSE quantify that Monte "
                "Carlo layer. Root bootstrap intervals quantify corpus-root "
                "sampling, not conditional action randomness."
            ),
        },
        "population": {
            "all_extracted_roots": int(len(root_categorical)),
            "categorical_roots_excluded": int(
                np.sum(root_categorical)
            ),
            "unresolved_guard_fallback_roots_excluded": int(
                np.sum(~root_categorical & unsafe)
            ),
            "eligible_unresolved_guard_accepted_roots": root_count,
            "q21_tail_clipped_roots": int(
                np.sum(np.asarray(q21_result.tail_range_clipped))
            ),
            "q21_nonfinite_roots": int(
                np.sum(~np.asarray(q21_result.finite))
            ),
            "q21_density_guard_roots": int(
                np.sum(
                    density_error
                    > DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE
                )
            ),
            "q21_density_error_max": float(np.max(density_error)),
            "q321_normalization_error_max_eligible": float(
                np.max(
                    np.asarray(reference_result.normalization_error)[
                        eligible
                    ]
                )
            ),
            "exact_plurality_raw_normalization_error_max": float(
                np.max(law_normalization_error)
            ),
        },
        "sampling_uncertainty": {
            "production_winner_samples_per_half_per_root": (
                samples_per_half
            ),
            "production_winner_samples_full_per_root": (
                full_population_samples
            ),
            "q_impl_expected_squared_l2_mc_error_plugin": (
                _scalar_summary(
                    expected_q_impl_squared_l2_mc,
                    weights,
                )
            ),
            "q_impl_split_half": _comparison_summary(
                q_impl_half_a,
                q_impl_half_b,
                weights,
            ),
            "plurality_law_split_half": _comparison_summary(
                g_half_a,
                g_half_b,
                weights,
            ),
            "plurality_full_mc_tv_first_order_proxy": _scalar_summary(
                first_order_law_tv_mc_proxy,
                weights,
            ),
            "q_impl_vs_q21_signal_to_mc_proxy": _scalar_summary(
                signal_to_mc_proxy,
                weights,
            ),
            "native_direct_validation": {
                "empirical_vs_exact_g_q_impl_estimate": (
                    _comparison_summary(
                        g_impl,
                        native_empirical,
                        weights,
                    )
                ),
                "calibration_diagnostic": _multinomial_calibration(
                    native_validation_count_np,
                    g_impl,
                ),
                "formal_p_value_valid": False,
                "note": (
                    "g(q_impl_hat) uses an estimated q; this calibration is "
                    "descriptive rather than a formal fixed-null test."
                ),
            },
            "q21_implementation_validation": {
                "empirical_vs_exact_g_q21": _comparison_summary(
                    g21,
                    q21_empirical,
                    weights,
                ),
                "calibration": _multinomial_calibration(
                    q21_validation_count_np,
                    g21,
                ),
                "formal_p_value_valid": True,
            },
        },
        "gates": {
            "definition": {
                "mean_l1": 0.05,
                "weighted_p95_l1": 0.15,
                "law_mode_agreement": 0.95,
                "endpoint_relative_error_fraction": 0.10,
                "endpoint_rule": (
                    "weighted mean L1 candidate error must be at most 10% "
                    "of each weighted mean separation between reference g32 "
                    "and g1 or the deterministic lowest-index argmax "
                    "endpoint; this endpoint is the M->infinity limit only "
                    "when the population mode is unique"
                ),
            },
            "observed": {
                "mean_l1": main_mean_l1,
                "weighted_p95_l1": (
                    main_summary["l1"]["weighted_p95"]
                ),
                "law_mode_agreement": (
                    main_summary["top_action_agreement_weighted"]
                ),
                "g1_separation_mean_l1": endpoint_g1_l1,
                "deterministic_argmax_separation_mean_l1": (
                    endpoint_argmax_l1
                ),
                "error_fraction_of_g1_separation": (
                    main_mean_l1 / max(endpoint_g1_l1, 1e-300)
                ),
                "error_fraction_of_deterministic_argmax_separation": (
                    main_mean_l1
                    / max(endpoint_argmax_l1, 1e-300)
                ),
            },
            "pass": gates,
        },
        "root_bootstrap_ci95": {
            "repetitions": args.bootstrap_repetitions,
            "seed": args.seed + 30_001,
            "intervals": bootstrap,
        },
        "groups": groups,
        "per_root": per_root,
        "implementation_validation": {
            "production_q21_plurality_called_directly": True,
            "production_native_thompson_primitive_called_directly": True,
            "lowest_index_count_argmax_used_in_native_validation": True,
            "exact_law_uses_strict_lower_and_weak_higher_count_bounds": True,
            "categorical_roots_in_exact_comparison": 0,
            "guard_fallback_roots_in_exact_comparison": 0,
        },
        "timing_seconds": {
            "tree_extraction_compile_and_first_by_step": (
                extraction_seconds
            ),
            "q321_and_q21_compile_and_run": reference_seconds,
            "production_population_two_halves": population_seconds,
            "exact_plurality_all_populations": exact_seconds,
            "direct_implementation_validation": validation_seconds,
            "total": time.perf_counter() - started_total,
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
