"""Replaceable bottom-up posterior repair rules."""

from __future__ import annotations

import math
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int8, Int32

from . import base
from .action_selection import posterior_best_policy
from .estimator_diagnostics import (
    binary_posterior_best_policy_prefix_quadrature,
)
from .outcomes import NO_OUTCOME, align_outcome
from .tree import (
    PosteriorUpdate,
    PosteriorUpdateContext,
    PosteriorUpdateDiagnostics,
)


DEFAULT_KAPPA = 4.0
DEFAULT_POLICY_SAMPLES = 32
DEFAULT_POLICY_SAMPLE_CHUNK_SIZE = 4
DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE = 0.01


@chex.dataclass(frozen=True)
class PosteriorEstimatorSnapshot:
    """Deterministic operands immediately before node-policy estimation.

    This is a benchmark surface, not persistent search state.  In particular,
    ``effective_alpha`` and ``categorical_outcome`` are the native objects
    passed to :func:`posterior_best_policy`, while ``cache_alpha`` is the
    Dirichlet-shaped projection mixed into the node cache after that policy is
    estimated.  Keeping both avoids reconstructing either object from a
    completed tree after earlier estimator noise has already propagated.
    """

    effective_alpha: Float[Array, "batch action outcome"]
    cache_alpha: Float[Array, "batch action outcome"]
    invalid_actions: Bool[Array, "batch action"]
    categorical_mask: Bool[Array, "batch action"]
    categorical_outcome: Int8[Array, "batch action"]
    n_down: Int32[Array, "batch"]
    gamma: Float[Array, "batch"]
    kappa: Float[Array, ""]
    value_prior: Float[Array, "batch outcome"]
    previous_value_alpha: Float[Array, "batch outcome"]
    active: Bool[Array, "batch"]


PosteriorPolicyEstimator = Callable[
    [base.PRNGKey, PosteriorEstimatorSnapshot],
    Float[Array, "batch action"],
]


def _validate_kappa(kappa: float) -> None:
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError(f"kappa must be finite and > 0, got {kappa}")


def _mix_weights(
    n_down: Int32[Array, "*batch"],
    *,
    dtype: jnp.dtype,
    kappa: float,
) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
    """Return the numerically stable prior and descendant cache weights."""

    n_down = n_down.astype(dtype)
    # Compute the normalized weights before multiplying either posterior.
    # The ratio form for kappa >= 1 also avoids overflowing the search dtype
    # when a very large (but finite) Python float is supplied.
    if kappa >= 1.0:
        relative_count = n_down * jnp.asarray(1.0 / kappa, dtype=dtype)
        denominator = 1.0 + relative_count
        prior_weight = 1.0 / denominator
        descendant_weight = relative_count / denominator
    else:
        kappa_array = jnp.asarray(kappa, dtype=dtype)
        denominator = kappa_array + n_down
        prior_weight = kappa_array / denominator
        descendant_weight = n_down / denominator
    return prior_weight, descendant_weight


def mix_value_prior(value_prior: Float[Array, "*batch outcome"], effective_action_alpha: Float[Array, "*batch action outcome"], search_policy: Float[Array, "*batch action"], n_down: Int32[Array, "*batch"], *, kappa: float = DEFAULT_KAPPA) -> Float[Array, "*batch outcome"]:
    """Mix a node prior with its policy-weighted current edge posteriors."""

    _validate_kappa(kappa)
    weighted_alpha = jnp.sum(search_policy[..., None] * effective_action_alpha, axis=-2)
    prior_weight, descendant_weight = _mix_weights(
        n_down,
        dtype=value_prior.dtype,
        kappa=kappa,
    )
    return (
        prior_weight[..., None] * value_prior
        + descendant_weight[..., None] * weighted_alpha
    )


def _prepare_posterior_estimator_snapshot(
    context: PosteriorUpdateContext,
    *,
    kappa: float,
) -> tuple[
    PosteriorEstimatorSnapshot,
    Float[Array, "batch action outcome"],
    Int32[Array, "batch action"],
]:
    """Apply deterministic message repairs and expose estimator operands."""

    node = context.node
    children = context.children
    leaf = context.leaf
    active = context.active
    edge_alpha = node.edge_alpha
    edge_payload = node.edge_payload
    edge_outcome = node.edge_categorical_outcome
    unresolved = edge_outcome == int(NO_OUTCOME)

    # The direct final-edge message is applied only at the deepest node.
    # Terminal expansion has already folded support into the parent and
    # replaced this payload with distance, so categorical edges are untouched.
    batch = jnp.arange(edge_payload.shape[0])
    leaf_action = jnp.where(leaf.active, leaf.action, 0)
    direct_alpha = align_outcome(leaf.value_alpha, leaf.to_play, node.to_play)
    old_direct_alpha = edge_alpha[batch, leaf_action]
    old_direct_count = edge_payload[batch, leaf_action]
    direct_is_dirichlet = leaf.active & unresolved[batch, leaf_action]
    edge_alpha = edge_alpha.at[batch, leaf_action].set(jnp.where(direct_is_dirichlet[..., None], direct_alpha, old_direct_alpha))
    edge_payload = edge_payload.at[batch, leaf_action].set(old_direct_count + direct_is_dirichlet.astype(edge_payload.dtype))

    child_target_player = jnp.broadcast_to(node.to_play[:, None], children.to_play.shape)
    child_value = align_outcome(children.value_alpha, children.to_play, child_target_player)
    refresh = (
        active[:, None]
        & children.visited
        & (children.categorical_outcome == int(NO_OUTCOME))
        & (children.node_payload > 0)
        & unresolved
    )
    edge_alpha = jnp.where(refresh[..., None], child_value, edge_alpha)
    edge_payload = jnp.where(refresh, 1 + children.node_payload, edge_payload)

    # Rebuild the effective edge posterior after the direct and child repairs.
    child_prior = align_outcome(children.value_prior, children.to_play, child_target_player)
    fallback = jnp.where(children.visited[..., None], child_prior, edge_alpha)
    unresolved_count = jnp.where(unresolved, edge_payload, 0)
    use_stored = ~unresolved | (unresolved_count > 0)
    effective_alpha = jnp.where(use_stored[..., None], edge_alpha, fallback)
    categorical = ~unresolved
    safe_categorical_outcome = jnp.where(categorical, edge_outcome, 0)
    categorical_mean = jax.nn.one_hot(safe_categorical_outcome, effective_alpha.shape[-1], dtype=effective_alpha.dtype)
    # A mixed unresolved node still needs a Dirichlet-shaped value cache.
    # Project exact edges to their categorical mean while preserving their
    # existing learned mass; selection and training continue to use sidecars.
    categorical_alpha = (
        jnp.sum(effective_alpha, axis=-1, keepdims=True) * categorical_mean
    )
    cache_alpha = jnp.where(categorical[..., None], categorical_alpha, effective_alpha)
    old_unresolved_count = jnp.where(unresolved, node.edge_payload, 0)
    legal = ~node.invalid_actions
    count_delta = jnp.sum(jnp.where(legal, unresolved_count - old_unresolved_count, 0), axis=-1)
    n_down = node.node_payload + count_delta
    _, gamma = _mix_weights(
        n_down,
        dtype=node.value_prior.dtype,
        kappa=kappa,
    )
    snapshot = PosteriorEstimatorSnapshot(
        effective_alpha=effective_alpha,
        cache_alpha=cache_alpha,
        invalid_actions=node.invalid_actions,
        categorical_mask=categorical,
        categorical_outcome=edge_outcome,
        n_down=n_down,
        gamma=gamma,
        kappa=jnp.asarray(kappa, dtype=node.value_prior.dtype),
        value_prior=node.value_prior,
        previous_value_alpha=node.value_alpha,
        active=active,
    )
    return snapshot, edge_alpha, edge_payload


def posterior_estimator_snapshot(
    context: PosteriorUpdateContext,
    *,
    kappa: float = DEFAULT_KAPPA,
) -> PosteriorEstimatorSnapshot:
    """Return the actual pre-estimator operands for offline benchmarking.

    The deterministic direct-edge and child-cache repairs are identical to
    those performed by :func:`update_posterior`.  No policy samples are drawn
    and calling this function cannot mutate a tree or affect search behavior.
    """

    _validate_kappa(kappa)
    snapshot, _, _ = _prepare_posterior_estimator_snapshot(
        context,
        kappa=kappa,
    )
    return snapshot


def update_posterior_with_estimator(
    rng_key: base.PRNGKey,
    context: PosteriorUpdateContext,
    estimator: PosteriorPolicyEstimator,
    *,
    kappa: float = DEFAULT_KAPPA,
) -> PosteriorUpdate:
    """Repair one path node using an injected posterior-policy estimator.

    ``estimator`` receives the original node-local PRNG key and the exact
    post-message, pre-estimator snapshot.  It must return one action
    distribution per batch lane.  This benchmark hook changes neither
    traversal nor persistent tree state; only the policy used in the existing
    cache mixture is replaceable.
    """

    _validate_kappa(kappa)
    snapshot, edge_alpha, edge_payload = _prepare_posterior_estimator_snapshot(
        context,
        kappa=kappa,
    )
    search_policy = estimator(rng_key, snapshot)
    repaired_value = mix_value_prior(
        snapshot.value_prior,
        snapshot.cache_alpha,
        search_policy,
        snapshot.n_down,
        kappa=kappa,
    )
    # Instrument the exact operands consumed by the cache mix.  These values
    # are returned only as diagnostics; no search decision reads them.
    weighted_alpha = jnp.sum(
        search_policy[..., None] * snapshot.cache_alpha,
        axis=-2,
    )
    innovation = weighted_alpha - snapshot.value_prior
    value_concentration = jnp.sum(
        snapshot.value_prior,
        axis=-1,
        keepdims=True,
    )
    weighted_concentration = jnp.sum(
        weighted_alpha,
        axis=-1,
        keepdims=True,
    )
    positive_concentration = (
        (value_concentration[..., 0] > 0)
        & (weighted_concentration[..., 0] > 0)
    )
    semantic_innovation = (
        weighted_alpha
        / jnp.where(
            weighted_concentration > 0,
            weighted_concentration,
            jnp.ones_like(weighted_concentration),
        )
        - snapshot.value_prior
        / jnp.where(
            value_concentration > 0,
            value_concentration,
            jnp.ones_like(value_concentration),
        )
    )
    dcache_dlogkappa = (
        -snapshot.gamma * (1.0 - snapshot.gamma)
    )[..., None] * innovation
    repaired_concentration = jnp.sum(
        repaired_value,
        axis=-1,
        keepdims=True,
    )
    repaired_mean = repaired_value / jnp.where(
        repaired_concentration > 0,
        repaired_concentration,
        jnp.ones_like(repaired_concentration),
    )
    mean_dcache_dlogkappa = (
        dcache_dlogkappa
        - repaired_mean
        * jnp.sum(
            dcache_dlogkappa,
            axis=-1,
            keepdims=True,
        )
    ) / jnp.where(
        repaired_concentration > 0,
        repaired_concentration,
        jnp.ones_like(repaired_concentration),
    )
    finite = (
        jnp.all(jnp.isfinite(weighted_alpha), axis=-1)
        & jnp.all(jnp.isfinite(snapshot.value_prior), axis=-1)
        & jnp.all(jnp.isfinite(innovation), axis=-1)
        & jnp.all(jnp.isfinite(semantic_innovation), axis=-1)
        & jnp.isfinite(weighted_concentration[..., 0])
        & jnp.isfinite(value_concentration[..., 0])
        & jnp.all(jnp.isfinite(mean_dcache_dlogkappa), axis=-1)
        & (repaired_concentration[..., 0] > 0)
        & jnp.isfinite(snapshot.gamma)
    )
    numeric = (
        snapshot.active
        & (snapshot.n_down > 0)
        & positive_concentration
        & finite
    )
    raw_innovation_l2 = jnp.linalg.vector_norm(innovation, axis=-1)
    semantic_innovation_l2 = jnp.linalg.vector_norm(
        semantic_innovation,
        axis=-1,
    )
    concentration_innovation_abs = jnp.abs(
        weighted_concentration[..., 0] - value_concentration[..., 0]
    )
    raw_dcache_dlogkappa_l2 = jnp.linalg.vector_norm(
        dcache_dlogkappa,
        axis=-1,
    )
    mean_dcache_dlogkappa_l2 = jnp.linalg.vector_norm(
        mean_dcache_dlogkappa,
        axis=-1,
    )
    log_concentration_dcache_dlogkappa_abs = jnp.abs(
        jnp.sum(dcache_dlogkappa, axis=-1)
        / jnp.where(
            repaired_concentration[..., 0] > 0,
            repaired_concentration[..., 0],
            jnp.ones_like(repaired_concentration[..., 0]),
        )
    )

    def numeric_value(value: jax.Array) -> jax.Array:
        return jnp.where(numeric, value, jnp.zeros_like(value))

    has_message = snapshot.n_down > 0
    value_alpha = jnp.where(
        (snapshot.active & has_message)[..., None],
        repaired_value,
        snapshot.previous_value_alpha,
    )
    return PosteriorUpdate(
        edge_alpha=edge_alpha,
        edge_payload=edge_payload,
        value_alpha=value_alpha,
        diagnostics=PosteriorUpdateDiagnostics(
            numeric=numeric,
            gamma=numeric_value(snapshot.gamma),
            raw_innovation_l2=numeric_value(raw_innovation_l2),
            semantic_innovation_l2=numeric_value(
                semantic_innovation_l2
            ),
            concentration_innovation_abs=numeric_value(
                concentration_innovation_abs
            ),
            raw_dcache_dlogkappa_l2=numeric_value(
                raw_dcache_dlogkappa_l2
            ),
            mean_dcache_dlogkappa_l2=numeric_value(
                mean_dcache_dlogkappa_l2
            ),
            log_concentration_dcache_dlogkappa_abs=numeric_value(
                log_concentration_dcache_dlogkappa_abs
            ),
        ),
    )


def update_posterior_prefix_cdf(
    rng_key: base.PRNGKey,
    context: PosteriorUpdateContext,
    *,
    kappa: float = DEFAULT_KAPPA,
    half_width: int = 20,
    tail_scale: float = 8.0,
    min_half_range: float = 6.0,
    max_half_range: float = 11.0,
    fallback_policy_samples: int = DEFAULT_POLICY_SAMPLES,
    fallback_policy_sample_chunk_size: int = (
        DEFAULT_POLICY_SAMPLE_CHUNK_SIZE
    ),
    density_log_integral_tolerance: float = (
        DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE
    ),
) -> PosteriorUpdate:
    """Repair a binary-outcome cache with mass-conserving prefix-CDF Q41.

    Only the policy estimate inside the existing bottom-up cache mixture is
    replaced.  Tree traversal and the public root-policy readout remain native
    Thompson sampling.

    The deterministic estimator is used while every batch lane stays inside
    its validated numerical envelope.  If any lane clips the adaptive tail
    range, produces a non-finite result, or has excessive pre-normalization
    density-integral error, the whole repair batch falls back to the original
    winner-count estimator.  A batch-level ``lax.cond`` keeps the safe hot
    path from paying the Monte Carlo cost.
    """

    _validate_kappa(kappa)
    if half_width < 1:
        raise ValueError(f"half_width must be >= 1, got {half_width}")
    if fallback_policy_samples < 1:
        raise ValueError(
            "fallback_policy_samples must be >= 1, got "
            f"{fallback_policy_samples}"
        )
    if fallback_policy_sample_chunk_size < 1:
        raise ValueError(
            "fallback_policy_sample_chunk_size must be >= 1, got "
            f"{fallback_policy_sample_chunk_size}"
        )
    if (
        not math.isfinite(density_log_integral_tolerance)
        or density_log_integral_tolerance <= 0.0
    ):
        raise ValueError(
            "density_log_integral_tolerance must be finite and > 0, got "
            f"{density_log_integral_tolerance}"
        )

    def prefix_estimator(
        estimator_rng_key: base.PRNGKey,
        snapshot: PosteriorEstimatorSnapshot,
    ) -> Float[Array, "batch action"]:
        if snapshot.effective_alpha.shape[-1] != 2:
            raise ValueError(
                "prefix-CDF posterior repair requires exactly two "
                "outcomes; got "
                f"{snapshot.effective_alpha.shape[-1]}"
            )
        estimate = binary_posterior_best_policy_prefix_quadrature(
            snapshot.effective_alpha,
            snapshot.invalid_actions,
            snapshot.categorical_outcome,
            half_width=half_width,
            adaptive_range=True,
            tail_scale=tail_scale,
            min_half_range=min_half_range,
            max_half_range=max_half_range,
            mass_conserving=True,
        )
        density_error = jnp.max(
            jnp.abs(estimate.density_log_integral),
            axis=-1,
        )
        unsafe = (
            estimate.tail_range_clipped
            | ~estimate.finite
            | (density_error > density_log_integral_tolerance)
        )

        def fallback(_: None) -> Float[Array, "batch action"]:
            return posterior_best_policy(
                estimator_rng_key,
                snapshot.effective_alpha,
                snapshot.invalid_actions,
                fallback_policy_samples,
                chunk_size=fallback_policy_sample_chunk_size,
                categorical_outcome=snapshot.categorical_outcome,
            )

        return jax.lax.cond(
            jnp.any(unsafe),
            fallback,
            lambda _: estimate.policy,
            operand=None,
        )

    return update_posterior_with_estimator(
        rng_key,
        context,
        prefix_estimator,
        kappa=kappa,
    )


def update_posterior(rng_key: base.PRNGKey, context: PosteriorUpdateContext, *, kappa: float = DEFAULT_KAPPA, policy_samples: int = DEFAULT_POLICY_SAMPLES, policy_sample_chunk_size: int = DEFAULT_POLICY_SAMPLE_CHUNK_SIZE) -> PosteriorUpdate:
    """Repair one path node using the Tic-Tac-Toe message-passing rule.

    This function owns the posterior mathematics. Search only gathers the
    current node and all child summaries, invokes the function bottom-up, and
    stores its complete return value. A different rule with the same two-arg
    contract can therefore replace this implementation directly.
    """

    _validate_kappa(kappa)
    if policy_samples < 1:
        raise ValueError(f"policy_samples must be >= 1, got {policy_samples}")
    if policy_sample_chunk_size < 1:
        raise ValueError(f"policy_sample_chunk_size must be >= 1, got {policy_sample_chunk_size}")

    def default_estimator(
        estimator_rng_key: base.PRNGKey,
        snapshot: PosteriorEstimatorSnapshot,
    ) -> Float[Array, "batch action"]:
        # This is computeStateSearchPosterior from the demo: pi_search is a
        # fresh population from the node's current post-repair action
        # posteriors. It is neither a visit policy nor a running mean.
        return posterior_best_policy(
            estimator_rng_key,
            snapshot.effective_alpha,
            snapshot.invalid_actions,
            policy_samples,
            chunk_size=policy_sample_chunk_size,
            categorical_outcome=snapshot.categorical_outcome,
        )

    return update_posterior_with_estimator(
        rng_key,
        context,
        default_estimator,
        kappa=kappa,
    )
