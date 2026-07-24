#!/usr/bin/env python3
"""Fixed-tree Hex6 root-readout benchmark.

This intentionally lives outside the repository.  It separates the completed
tree from its public policy readout and measures:

* the actual M32 (or one categorical-tie draw) target error;
* guarded mass-conserving prefix-CDF Q21 error against the float64 Q321
  exact-Beta reference on unresolved roots;
* the exact categorical-tie population on solved roots;
* direct policy-logit gradient signal/noise; and
* the incremental warm runtime of retaining M32 for commitment while adding
  Q21 only for the training target.

No trajectory is generated from the Q21 target.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import jax

import jax.numpy as jnp
from flax import nnx

from e4_repair_context_corpus import (
    _load_checkpoint_config_and_model,
    _replay_roots,
)
from scacchi import dirichlet_mctx
from scacchi.dirichlet_mctx import action_selection
from scacchi.dirichlet_mctx.estimator_diagnostics import (
    binary_posterior_best_policy_prefix_quadrature,
    binary_posterior_best_policy_quadrature,
)
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.posterior_updates import (
    DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE,
)
from scacchi.dirichlet_q_search import (
    make_dirichlet_expand_fn,
    terminal_outcome_from_reward,
)
from scacchi.play_search import make_evaluator


def _subset(table: dict[str, np.ndarray], indices: np.ndarray):
    return {key: value[indices] for key, value in table.items()}


def _categorical_population(
    node_outcome: jax.Array,
    edge_outcome: jax.Array,
    edge_distance: jax.Array,
    invalid: jax.Array,
    *,
    num_outcomes: int,
) -> jax.Array:
    """Expectation of production categorical_action's uniform tie draw."""

    legal = ~invalid
    win_index = int(num_outcomes) - 1
    win_candidates = legal & (edge_outcome == win_index)
    loss_candidates = legal & (edge_outcome == 0)
    draw_candidates = legal & (edge_outcome == 1)
    distance = edge_distance.astype(jnp.float32)
    win_scores = jnp.where(win_candidates, -distance, -jnp.inf)
    loss_scores = jnp.where(loss_candidates, distance, -jnp.inf)
    draw_scores = jnp.where(draw_candidates, 0.0, -jnp.inf)
    scores = jnp.where(
        (node_outcome == win_index)[..., None],
        win_scores,
        jnp.where(
            (node_outcome == 0)[..., None],
            loss_scores,
            draw_scores,
        ),
    )
    best = jnp.max(scores, axis=-1, keepdims=True)
    tied = jnp.isfinite(scores) & (scores == best)
    count = jnp.sum(tied, axis=-1, keepdims=True)
    return tied.astype(jnp.float32) / jnp.maximum(count, 1)


def _public_m32_readout(
    key: jax.Array,
    alpha: jax.Array,
    invalid: jax.Array,
    node_outcome: jax.Array,
    edge_outcome: jax.Array,
    edge_distance: jax.Array,
    *,
    num_samples: int = 32,
    chunk_size: int = 32,
) -> jax.Array:
    """Match policies.dirichlet_thompson_policy after its tree is fixed."""

    categorical_action = action_selection.categorical_action(
        jax.random.fold_in(key, 0),
        node_outcome,
        edge_outcome,
        edge_distance,
        invalid,
        num_outcomes=alpha.shape[-1],
    )
    categorical_policy = jax.nn.one_hot(
        categorical_action,
        alpha.shape[-2],
        dtype=alpha.dtype,
    )
    sampled_policy = action_selection.posterior_best_policy(
        key,
        alpha,
        invalid,
        num_samples,
        chunk_size=chunk_size,
        categorical_outcome=edge_outcome,
    )
    return jnp.where(
        (node_outcome != int(NO_OUTCOME))[..., None],
        categorical_policy,
        sampled_policy,
    )


def _guarded_q21(
    alpha: jax.Array,
    invalid: jax.Array,
    edge_outcome: jax.Array,
    fallback: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    estimate = binary_posterior_best_policy_prefix_quadrature(
        alpha,
        invalid,
        edge_outcome,
        half_width=10,
        adaptive_range=True,
        tail_scale=8.0,
        min_half_range=6.0,
        max_half_range=11.0,
        mass_conserving=True,
    )
    density_error = jnp.max(
        jnp.abs(estimate.density_log_integral),
        axis=-1,
    )
    unsafe = (
        estimate.tail_range_clipped
        | ~estimate.finite
        | (
            density_error
            > DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE
        )
    )
    target = jnp.where(unsafe[..., None], fallback, estimate.policy)
    diagnostics = {
        "unsafe": unsafe,
        "tail_range_clipped": estimate.tail_range_clipped,
        "finite": estimate.finite,
        "density_error": density_error,
        "normalization_error": estimate.normalization_error,
        "fallback_interval_count": estimate.fallback_interval_count,
    }
    return target, diagnostics


def _make_tree_extractor(env: Any, search_config: Any):
    """Search once with the configured guarded Q21 internal repair."""

    update = functools.partial(
        dirichlet_mctx.update_posterior_prefix_cdf,
        kappa=float(search_config.kappa),
        half_width=10,
        tail_scale=8.0,
        min_half_range=6.0,
        max_half_range=11.0,
        fallback_policy_samples=max(
            1,
            int(
                search_config.policy_samples
                if search_config.posterior_policy_samples is None
                else search_config.posterior_policy_samples
            ),
        ),
        fallback_policy_sample_chunk_size=max(
            1,
            int(search_config.policy_sample_chunk_size or 4),
        ),
    )

    @nnx.jit
    def extract(model, state, rng_key):
        evaluator = make_evaluator(model)
        prediction = evaluator(state.observation)
        assert prediction.alpha_v is not None
        assert prediction.alpha_q is not None
        invalid = ~state.legal_action_mask
        root_reward = state.rewards[
            jnp.arange(state.rewards.shape[0]),
            state.current_player,
        ]
        terminal_outcome = jnp.where(
            state.terminated,
            terminal_outcome_from_reward(
                root_reward,
                prediction.alpha_v.shape[-1],
            ),
            jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
        )
        root = dirichlet_mctx.RootFnOutput(
            prior_logits=prediction.logits,
            value=prediction.alpha_v,
            action_values=prediction.alpha_q,
            embedding=state,
            terminal_outcome=terminal_outcome,
            to_play=state.current_player,
        )
        output = dirichlet_mctx.dirichlet_thompson_policy(
            params=(),
            rng_key=rng_key,
            root=root,
            recurrent_fn=make_dirichlet_expand_fn(env, evaluator),
            num_simulations=int(search_config.num_simulations),
            invalid_actions=invalid,
            posterior_update=update,
            max_depth=search_config.max_depth,
            policy_samples=32,
            policy_sample_chunk_size=32,
        )
        tree = output.search_tree
        summary = tree.summary()
        masked_logits = jnp.where(
            invalid,
            jnp.finfo(prediction.logits.dtype).min,
            prediction.logits,
        )
        prior = jax.nn.softmax(masked_logits, axis=-1)
        return (
            prior,
            summary.alpha,
            invalid,
            summary.v_categorical_outcome,
            summary.q_categorical_outcome,
            summary.q_categorical_distance,
            output.action_weights,
            output.action,
            tree.simulation_active_count,
            tree.executed_simulation_call_count,
        )

    return extract


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.broadcast_to(
        np.asarray(weights, dtype=np.float64),
        values.shape,
    )
    return float(np.sum(values * weights) / np.sum(weights))


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

    def kl(p, q):
        terms = np.zeros_like(p)
        positive = p > 0.0
        terms[positive] = p[positive] * (
            np.log(p[positive]) - np.log(q[positive])
        )
        return np.sum(terms, axis=-1)

    return 0.5 * (kl(left, middle) + kl(right, middle))


def _cosine(estimate: np.ndarray, exact: np.ndarray) -> np.ndarray:
    numerator = np.sum(estimate * exact, axis=-1)
    denominator = np.sqrt(
        np.sum(estimate * estimate, axis=-1)
        * np.sum(exact * exact, axis=-1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 1e-15,
    )


def _summarize_estimator(
    name: str,
    targets: np.ndarray,
    reference: np.ndarray,
    prior: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Summarize [R,B,A] or [B,A] targets against [B,A] reference."""

    if targets.ndim == 2:
        targets = targets[None, ...]
    reps = targets.shape[0]
    reference_b = reference[None, ...]
    prior_b = prior[None, ...]
    error = targets - reference_b
    squared_l2 = np.sum(error * error, axis=-1)
    l1 = np.sum(np.abs(error), axis=-1)
    exact_gradient = prior - reference
    estimated_gradient = prior_b - targets
    signal_energy = np.sum(exact_gradient * exact_gradient, axis=-1)
    projection = np.sum(
        estimated_gradient * exact_gradient[None, ...],
        axis=-1,
    ) / np.maximum(signal_energy[None, ...], 1e-15)
    rep_weights = np.broadcast_to(weights[None, :], (reps, len(weights)))
    mean_error = np.mean(targets, axis=0) - reference
    supported = reference > 1e-4
    missed_mass = np.sum(
        np.where((targets == 0.0) & supported[None, ...], reference_b, 0.0),
        axis=-1,
    )
    top_ref = np.argmax(reference, axis=-1)
    top_est = np.argmax(targets, axis=-1)
    return {
        "name": name,
        "repetitions": reps,
        "squared_l2_error_mean": _weighted_mean(squared_l2, rep_weights),
        "squared_l2_error_p95": float(np.quantile(squared_l2, 0.95)),
        "l1_error_mean": _weighted_mean(l1, rep_weights),
        "l1_error_p95": float(np.quantile(l1, 0.95)),
        "l1_error_max": float(np.max(l1)),
        "js_to_reference_mean_nats": _weighted_mean(
            _js(targets, reference_b),
            rep_weights,
        ),
        "argmax_disagreement_fraction": _weighted_mean(
            top_est != top_ref[None, :],
            rep_weights,
        ),
        "target_entropy_mean": _weighted_mean(
            _entropy(targets),
            rep_weights,
        ),
        "reference_entropy_mean": _weighted_mean(
            _entropy(reference),
            weights,
        ),
        "entropy_bias_mean": _weighted_mean(
            _entropy(targets) - _entropy(reference)[None, :],
            rep_weights,
        ),
        "reference_supported_mass_missed_mean": _weighted_mean(
            missed_mass,
            rep_weights,
        ),
        "mean_estimator_bias_squared_l2": _weighted_mean(
            np.sum(mean_error * mean_error, axis=-1),
            weights,
        ),
        "exact_logit_gradient_signal_energy": _weighted_mean(
            signal_energy,
            weights,
        ),
        "target_error_to_gradient_signal_ratio": (
            _weighted_mean(squared_l2, rep_weights)
            / max(_weighted_mean(signal_energy, weights), 1e-15)
        ),
        "logit_gradient_cosine_mean": _weighted_mean(
            _cosine(
                estimated_gradient,
                exact_gradient[None, ...],
            ),
            rep_weights,
        ),
        "logit_gradient_projection_mean": _weighted_mean(
            projection,
            rep_weights,
        ),
    }


def _paired_bootstrap_geomean(
    ratios: np.ndarray,
    *,
    seed: int,
    repetitions: int = 10_000,
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(ratios),
        size=(repetitions, len(ratios)),
    )
    draws = np.exp(np.mean(np.log(ratios)[indices], axis=-1))
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def _time_readouts(
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    repetitions: int,
    warmups: int,
    seed: int,
) -> dict[str, Any]:
    count = len(arrays["alpha"])
    indices = np.resize(np.arange(count), batch_size)
    alpha = jnp.asarray(arrays["alpha"][indices])
    invalid = jnp.asarray(arrays["invalid"][indices])
    node_outcome = jnp.asarray(arrays["node_outcome"][indices])
    edge_outcome = jnp.asarray(arrays["edge_outcome"][indices])
    edge_distance = jnp.asarray(arrays["edge_distance"][indices])

    @jax.jit
    def baseline(key):
        return _public_m32_readout(
            key,
            alpha,
            invalid,
            node_outcome,
            edge_outcome,
            edge_distance,
        )

    @jax.jit
    def q21_only(key):
        fallback = jnp.zeros(alpha.shape[:-1], dtype=alpha.dtype)
        target, diagnostics = _guarded_q21(
            alpha,
            invalid,
            edge_outcome,
            fallback,
        )
        return target, diagnostics["unsafe"]

    @jax.jit
    def combined(key):
        commitment = _public_m32_readout(
            key,
            alpha,
            invalid,
            node_outcome,
            edge_outcome,
            edge_distance,
        )
        target, diagnostics = _guarded_q21(
            alpha,
            invalid,
            edge_outcome,
            commitment,
        )
        categorical = _categorical_population(
            node_outcome,
            edge_outcome,
            edge_distance,
            invalid,
            num_outcomes=alpha.shape[-1],
        )
        target = jnp.where(
            (node_outcome != int(NO_OUTCOME))[..., None],
            categorical,
            target,
        )
        return commitment, target, diagnostics["unsafe"]

    methods = {
        "m32_commitment_only": baseline,
        "q21_target_only": q21_only,
        "m32_commitment_plus_q21_target": combined,
    }
    base_key = jax.random.PRNGKey(seed)
    compile_seconds: dict[str, float] = {}
    for name, method in methods.items():
        started = time.perf_counter()
        jax.block_until_ready(method(base_key))
        compile_seconds[name] = time.perf_counter() - started
    for index in range(warmups):
        key = jax.random.fold_in(base_key, index + 1)
        order = list(methods)
        if index % 2:
            order.reverse()
        for name in order:
            jax.block_until_ready(methods[name](key))
    samples = {
        name: np.empty(repetitions, dtype=np.float64)
        for name in methods
    }
    names = list(methods)
    for repetition in range(repetitions):
        key = jax.random.fold_in(base_key, warmups + repetition + 1)
        order = names[repetition % len(names) :] + names[
            : repetition % len(names)
        ]
        if repetition % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            jax.block_until_ready(methods[name](key))
            samples[name][repetition] = time.perf_counter() - started

    # This directly proves the additional target cannot alter the commitment
    # policy under the proposed API/key discipline.
    check_key = jax.random.fold_in(base_key, 999_999)
    baseline_policy = np.asarray(jax.device_get(baseline(check_key)))
    combined_policy = np.asarray(jax.device_get(combined(check_key)[0]))
    if not np.array_equal(baseline_policy, combined_policy):
        raise AssertionError("combined readout changed the M32 commitment")

    result: dict[str, Any] = {
        "batch_size": batch_size,
        "unique_roots": count,
        "roots_tiled": batch_size > count,
        "repetitions": repetitions,
        "warmups": warmups,
        "commitment_policy_bitwise_equal": True,
        "methods": {},
    }
    for name in names:
        median = float(np.median(samples[name]))
        result["methods"][name] = {
            "compile_and_first_seconds": compile_seconds[name],
            "warm_seconds_mean": float(np.mean(samples[name])),
            "warm_seconds_median": median,
            "warm_seconds_p95": float(np.quantile(samples[name], 0.95)),
            "roots_per_second_at_median": batch_size / median,
        }
    base = samples["m32_commitment_only"]
    combined_samples = samples["m32_commitment_plus_q21_target"]
    ratio = combined_samples / base
    delta = combined_samples - base
    result["incremental"] = {
        "paired_ratio_geometric_mean": float(
            np.exp(np.mean(np.log(ratio)))
        ),
        "paired_ratio_median": float(np.median(ratio)),
        "paired_ratio_p95": float(np.quantile(ratio, 0.95)),
        "paired_geomean_ratio_ci95": _paired_bootstrap_geomean(
            ratio,
            seed=seed + 31,
        ),
        "median_seconds_per_batch": float(np.median(delta)),
        "mean_seconds_per_batch": float(np.mean(delta)),
        "median_microseconds_per_root": float(
            1e6 * np.median(delta) / batch_size
        ),
    }
    return result


def _parse_steps(encoded: str) -> tuple[int, ...]:
    return tuple(int(item) for item in encoded.split(",") if item.strip())


def main() -> None:
    # Enable the float64 Q321 reference only for executable benchmark runs.
    # Importing this module for its small pure helpers must not mutate the
    # process-wide JAX precision mode used by the test suite.
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
    parser.add_argument("--readout-repetitions", type=int, default=128)
    parser.add_argument("--timing-batch-size", type=int, default=8192)
    parser.add_argument("--timing-repetitions", type=int, default=30)
    parser.add_argument("--timing-warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    roots = dict(
        np.load(args.corpus / "roots.npz", allow_pickle=False)
    )
    chunks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "prior",
            "alpha",
            "invalid",
            "node_outcome",
            "edge_outcome",
            "edge_distance",
            "native_m32",
            "native_action",
            "checkpoint_step",
            "stage_id",
            "root_weight",
            "root_id",
            "simulation_active_count",
            "executed_simulation_call_count",
        )
    }
    base_key = jax.random.PRNGKey(args.seed)
    extraction_seconds: dict[str, float] = {}
    for ordinal, step in enumerate(_parse_steps(args.steps)):
        config, env, model, _ = _load_checkpoint_config_and_model(
            args.checkpoint,
            step,
        )
        search_config = config.selfplay.search.dirichlet_thompson
        indices = np.flatnonzero(roots["checkpoint_step"] == step)
        if len(indices) == 0:
            raise ValueError(f"no corpus roots for checkpoint step {step}")
        state = _replay_roots(env, _subset(roots, indices))
        extract = _make_tree_extractor(env, search_config)
        key = jax.random.fold_in(base_key, ordinal)
        started = time.perf_counter()
        output = jax.block_until_ready(extract(model, state, key))
        extraction_seconds[str(step)] = time.perf_counter() - started
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
        for name, value in zip(names, output, strict=True):
            chunks[name].append(np.asarray(jax.device_get(value)))
        for name in ("checkpoint_step", "stage_id", "root_weight", "root_id"):
            chunks[name].append(roots[name][indices])
        del model, extract

    arrays = {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }
    alpha = jnp.asarray(arrays["alpha"], dtype=jnp.float64)
    invalid = jnp.asarray(arrays["invalid"])
    edge_outcome = jnp.asarray(arrays["edge_outcome"])
    node_outcome = jnp.asarray(arrays["node_outcome"])
    edge_distance = jnp.asarray(arrays["edge_distance"])

    reference_result = jax.block_until_ready(
        jax.jit(
            lambda a, inv, cat: binary_posterior_best_policy_quadrature(
                a,
                inv,
                cat,
                half_width=160,
                step=0.1,
            )
        )(alpha, invalid, edge_outcome)
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
        )(alpha.astype(jnp.float32), invalid, edge_outcome)
    )
    categorical_reference = jax.block_until_ready(
        jax.jit(
            lambda n, e, d, inv: _categorical_population(
                n,
                e,
                d,
                inv,
                num_outcomes=2,
            )
        )(node_outcome, edge_outcome, edge_distance, invalid)
    )
    reference = np.asarray(
        reference_result.policy,
        dtype=np.float64,
    ).copy()
    categorical_reference_np = np.asarray(
        categorical_reference,
        dtype=np.float64,
    )
    root_categorical = arrays["node_outcome"] != int(NO_OUTCOME)
    reference[root_categorical] = categorical_reference_np[root_categorical]

    q21 = np.asarray(q21_result.policy, dtype=np.float64)
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

    # Repeated fixed-tree targets.  These use independent readout keys and do
    # not rerun traversal or repair.
    @jax.jit
    def repeated_readouts(keys):
        return jax.vmap(
            lambda key: _public_m32_readout(
                key,
                alpha.astype(jnp.float32),
                invalid,
                node_outcome,
                edge_outcome,
                edge_distance,
            )
        )(keys)

    readout_keys = jax.random.split(
        jax.random.fold_in(base_key, 12345),
        args.readout_repetitions,
    )
    m32 = np.asarray(
        jax.device_get(repeated_readouts(readout_keys)),
        dtype=np.float64,
    )
    guarded_q21 = np.broadcast_to(q21[None, ...], m32.shape).copy()
    guarded_q21[:, unsafe, :] = m32[:, unsafe, :]
    # Population-consistent target: exact expected categorical tie policy,
    # guarded Q21 on unresolved roots.
    population_q21 = guarded_q21.copy()
    population_q21[:, root_categorical, :] = categorical_reference_np[
        None,
        root_categorical,
        :,
    ]
    # Conservative variant leaves categorical target exactly as production.
    conservative_q21 = guarded_q21.copy()
    conservative_q21[:, root_categorical, :] = m32[
        :,
        root_categorical,
        :,
    ]

    prior = arrays["prior"].astype(np.float64)
    weights = arrays["root_weight"].astype(np.float64)

    def summarize_mask(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"roots": 0}
        local_weights = weights[mask]
        local_reference = reference[mask]
        local_prior = prior[mask]
        m = _summarize_estimator(
            "m32",
            m32[:, mask, :],
            local_reference,
            local_prior,
            local_weights,
        )
        q = _summarize_estimator(
            "guarded_q21_population_categorical",
            population_q21[:, mask, :],
            local_reference,
            local_prior,
            local_weights,
        )
        c = _summarize_estimator(
            "guarded_q21_preserve_categorical_draw",
            conservative_q21[:, mask, :],
            local_reference,
            local_prior,
            local_weights,
        )
        return {
            "roots": int(np.sum(mask)),
            "weighted_root_mass": float(np.sum(local_weights)),
            "reference_search_information": {
                "prior_to_reference_squared_l2": _weighted_mean(
                    np.sum(
                        (local_prior - local_reference) ** 2,
                        axis=-1,
                    ),
                    local_weights,
                ),
                "prior_to_reference_js_nats": _weighted_mean(
                    _js(local_prior, local_reference),
                    local_weights,
                ),
            },
            "m32": m,
            "q21_population_categorical": q,
            "q21_preserve_categorical_draw": c,
            "q21_to_m32_squared_l2_error_ratio": (
                q["squared_l2_error_mean"]
                / max(m["squared_l2_error_mean"], 1e-30)
            ),
        }

    groups: dict[str, Any] = {
        "overall": summarize_mask(np.ones(len(reference), dtype=bool)),
        "unresolved": summarize_mask(~root_categorical),
        "categorical": summarize_mask(root_categorical),
    }
    for step in _parse_steps(args.steps):
        mask = arrays["checkpoint_step"] == step
        groups[f"checkpoint_{step}"] = summarize_mask(mask)
        for stage_id, stage_name in enumerate(("early", "mid", "late")):
            groups[f"checkpoint_{step}_stage_{stage_name}"] = summarize_mask(
                mask & (arrays["stage_id"] == stage_id)
            )

    # Multinomial formula is exact if the root population equals the
    # float64 reference.  Compare it to implemented repeated M32.
    unresolved = ~root_categorical
    analytic_m32_l2 = (
        1.0 - np.sum(reference[unresolved] ** 2, axis=-1)
    ) / 32.0
    empirical_m32_l2 = np.mean(
        np.sum(
            (
                m32[:, unresolved, :]
                - reference[None, unresolved, :]
            )
            ** 2,
            axis=-1,
        ),
        axis=0,
    )

    timing = _time_readouts(
        arrays,
        batch_size=args.timing_batch_size,
        repetitions=args.timing_repetitions,
        warmups=args.timing_warmups,
        seed=args.seed + 19,
    )
    report = {
        "format": "scacchi.fixed_tree_root_readout_benchmark.v1",
        "backend": jax.default_backend(),
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.corpus),
        "steps": list(_parse_steps(args.steps)),
        "root_count": len(reference),
        "readout_repetitions": args.readout_repetitions,
        "protocol": {
            "tree": (
                "one completed tree per root with guarded internal Q21; "
                "all target estimates reuse that fixed tree"
            ),
            "reference": (
                "float64 exact-Beta Q321 (half_width=160, step=0.1) on "
                "unresolved roots; exact expected categorical_action tie "
                "population on categorical roots"
            ),
            "m32": (
                "32 implemented bounded-work Thompson winners on unresolved "
                "roots; one production categorical tie draw on solved roots"
            ),
            "q21": (
                "mass-conserving adaptive prefix-CDF Q21, production density/"
                "tail/finite guard; M32 fallback"
            ),
            "gradient_metric": (
                "for policy cross-entropy, dL/dlogit = softmax(logit)-target, "
                "so target squared-L2 error is exactly local policy-logit "
                "gradient squared error"
            ),
        },
        "tree_extraction_compile_and_first_seconds": extraction_seconds,
        "tree_work": {
            "simulation_active_count_mean": float(
                np.mean(arrays["simulation_active_count"])
            ),
            "executed_simulation_call_count_mean": float(
                np.mean(arrays["executed_simulation_call_count"])
            ),
        },
        "population": {
            "unresolved_roots": int(np.sum(~root_categorical)),
            "categorical_roots": int(np.sum(root_categorical)),
            "categorical_fraction": float(np.mean(root_categorical)),
            "q21_guard_fallback_roots": int(np.sum(unsafe)),
            "q21_guard_fallback_fraction": float(np.mean(unsafe)),
            "q21_tail_clipped_roots": int(
                np.sum(np.asarray(q21_result.tail_range_clipped))
            ),
            "q21_nonfinite_roots": int(
                np.sum(~np.asarray(q21_result.finite))
            ),
            "q21_density_error_max": float(np.max(density_error)),
            "q21_normalization_error_max": float(
                np.max(np.asarray(q21_result.normalization_error))
            ),
            "q21_fallback_intervals_total": int(
                np.sum(np.asarray(q21_result.fallback_interval_count))
            ),
            "reference_normalization_error_max_unresolved": float(
                np.max(
                    np.asarray(reference_result.normalization_error)[
                        unresolved
                    ]
                )
                if np.any(unresolved)
                else 0.0
            ),
        },
        "implemented_m32_vs_multinomial_formula_unresolved": {
            "analytic_squared_l2_mean": _weighted_mean(
                analytic_m32_l2,
                weights[unresolved],
            ),
            "empirical_squared_l2_mean": _weighted_mean(
                empirical_m32_l2,
                weights[unresolved],
            ),
            "empirical_to_analytic_ratio": (
                _weighted_mean(empirical_m32_l2, weights[unresolved])
                / _weighted_mean(analytic_m32_l2, weights[unresolved])
            ),
        },
        "groups": groups,
        "fixed_tree_timing": timing,
        "invariants": {
            "q21_never_used_for_tree_traversal_after_tree_fixed": True,
            "q21_never_used_for_commitment_policy": True,
            "combined_commitment_bitwise_matches_m32_baseline": timing[
                "commitment_policy_bitwise_equal"
            ],
            "all_reference_policies_finite": bool(
                np.all(np.isfinite(reference))
            ),
            "all_m32_policies_simplex": bool(
                np.allclose(np.sum(m32, axis=-1), 1.0, atol=1e-6)
            ),
            "all_population_q21_policies_simplex": bool(
                np.allclose(
                    np.sum(population_q21, axis=-1),
                    1.0,
                    atol=1e-6,
                )
            ),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
