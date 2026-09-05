"""Guardable binary posterior-best probabilities from prefix CDFs.

This module contains only the production estimator used by optional
bottom-up cache repair.  Reference quadratures and benchmark estimators live
outside the search package.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import betaln
from jaxtyping import Array, Bool, Float, Int8, Int32

from .outcomes import NO_OUTCOME


class BinaryPrefixQuadraturePolicy(NamedTuple):
    """Policy plus numerical metadata.

    ``fallback_interval_count`` is descriptive only: endpoint underflow can
    make it nonzero even when the affected maximum-CDF mass is negligible. It
    is deliberately not a repair guard.
    """

    policy: Float[Array, "*batch action"]
    raw_policy: Float[Array, "*batch action"]
    raw_mass: Float[Array, "*batch"]
    normalization_error: Float[Array, "*batch"]
    finite: Bool[Array, "*batch"]
    grid_half_range: Float[Array, "*batch"]
    tail_range_clipped: Bool[Array, "*batch"]
    density_log_integral: Float[Array, "*batch action"]
    fallback_interval_count: Int32[Array, "*batch"]


class _FinalizedPolicy(NamedTuple):
    policy: jax.Array
    raw_policy: jax.Array
    raw_mass: jax.Array
    normalization_error: jax.Array
    finite: jax.Array


def _log_cosh(value: jax.Array) -> jax.Array:
    absolute = jnp.abs(value)
    return (
        absolute
        + jnp.log1p(jnp.exp(-2.0 * absolute))
        - jnp.log(jnp.asarray(2.0, dtype=value.dtype))
    )


def _sinh_logit_transform(
    coordinate: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    logit = jnp.sinh(coordinate)
    log_x = -jax.nn.softplus(-logit)
    log_one_minus_x = -jax.nn.softplus(logit)
    # dx/dt = x(1-x) cosh(t); x and (1-x) are absorbed by
    # the endpoint exponents below.
    return log_x, log_one_minus_x, _log_cosh(coordinate)


def _adaptive_grid_half_range(
    alpha: jax.Array,
    unresolved: jax.Array,
    *,
    tail_scale: float,
    min_half_range: float,
    max_half_range: float,
) -> tuple[jax.Array, jax.Array]:
    if not math.isfinite(tail_scale) or tail_scale <= 0.0:
        raise ValueError(
            f"tail_scale must be finite and > 0, got {tail_scale}"
        )
    if not math.isfinite(min_half_range) or min_half_range <= 0.0:
        raise ValueError(
            "min_half_range must be finite and > 0, "
            f"got {min_half_range}"
        )
    if (
        not math.isfinite(max_half_range)
        or max_half_range < min_half_range
    ):
        raise ValueError(
            "max_half_range must be finite and >= min_half_range; "
            f"got {max_half_range} and {min_half_range}"
        )

    dtype = alpha.dtype
    has_unresolved = jnp.any(unresolved, axis=-1)
    safe_component = jnp.where(
        unresolved[..., None],
        jnp.maximum(alpha, jnp.finfo(dtype).tiny),
        jnp.inf,
    )
    minimum_alpha = jnp.min(safe_component, axis=(-2, -1))
    requested = jnp.arcsinh(
        jnp.asarray(tail_scale, dtype=dtype) / minimum_alpha
    )
    requested = jnp.where(
        has_unresolved,
        requested,
        jnp.asarray(min_half_range, dtype=dtype),
    )
    half_range = jnp.clip(
        requested,
        jnp.asarray(min_half_range, dtype=dtype),
        jnp.asarray(max_half_range, dtype=dtype),
    )
    clipped = has_unresolved & (
        requested > jnp.asarray(max_half_range, dtype=dtype)
    )
    return half_range, clipped


def _finalize_policy(
    unresolved_probability: jax.Array,
    legal: jax.Array,
    unresolved: jax.Array,
    certified_win: jax.Array,
    categorical_outcome: jax.Array,
    dtype: jnp.dtype,
) -> _FinalizedPolicy:
    """Apply the native binary categorical precedence and tie semantics."""

    has_unresolved = jnp.any(unresolved, axis=-1)
    has_certified_win = jnp.any(certified_win, axis=-1)
    has_legal = jnp.any(legal, axis=-1)
    num_actions = legal.shape[-1]

    first_certified_win = jnp.argmax(
        certified_win.astype(jnp.int32),
        axis=-1,
    )

    # Without a certified win or unresolved legal action, match masked_argmax:
    # highest exact binary outcome, with the lowest index resolving ties.
    categorical_score = jnp.where(
        legal,
        categorical_outcome.astype(dtype),
        -jnp.inf,
    )
    categorical_best = jnp.argmax(categorical_score, axis=-1)
    categorical_best = jnp.where(
        has_certified_win, first_certified_win, categorical_best
    )
    categorical_policy = jax.nn.one_hot(
        categorical_best,
        num_actions,
        dtype=dtype,
    )

    raw_policy = jnp.where(
        (has_certified_win | ~has_unresolved)[..., None],
        categorical_policy,
        unresolved_probability,
    )
    raw_policy = jnp.where(legal, raw_policy, 0.0)
    raw_mass = jnp.sum(raw_policy, axis=-1)
    normalization_error = jnp.abs(
        raw_mass - has_legal.astype(dtype)
    )
    safe_mass = jnp.where(raw_mass > 0.0, raw_mass, 1.0)
    policy = raw_policy / safe_mass[..., None]
    policy = jnp.where(has_legal[..., None], policy, 0.0)
    finite = (
        jnp.all(jnp.isfinite(policy), axis=-1)
        & jnp.all(jnp.isfinite(raw_policy), axis=-1)
        & jnp.isfinite(raw_mass)
        & jnp.isfinite(normalization_error)
    )
    return _FinalizedPolicy(
        policy=policy,
        raw_policy=raw_policy,
        raw_mass=raw_mass,
        normalization_error=normalization_error,
        finite=finite,
    )


def binary_posterior_best_policy_prefix_quadrature(
    alpha: Float[Array, "*batch action 2"],
    invalid_actions: Bool[Array, "*batch action"],
    categorical_outcome: Int8[Array, "*batch action"] | None = None,
    *,
    half_width: int = 10,
    tail_scale: float = 8.0,
    min_half_range: float = 6.0,
    max_half_range: float = 11.0,
) -> BinaryPrefixQuadraturePolicy:
    """Approximate binary posterior-best probabilities in ``O(AQ)``.

    The estimator uses a shared adaptive sinh-logit grid with
    ``Q = 2 * half_width + 1`` points.  Each interval's increment in the CDF
    of the maximum is allocated among actions in proportion to their local
    nonnegative winner contribution.  Those increments telescope, making the
    raw policy mass-conserving up to floating-point roundoff.

    ``tail_range_clipped``, ``finite``, and ``density_log_integral`` are the
    production safety signals. The caller decides whether to accept the
    deterministic estimate or fall back to its native winner population.
    They guard numerical integration failures, not the Q21 discretization
    error relative to a denser reference.
    """

    alpha = jnp.asarray(alpha)
    invalid_actions = jnp.asarray(invalid_actions)
    if alpha.shape[-1] != 2:
        raise ValueError(
            "binary prefix quadrature requires exactly two outcomes; "
            f"got shape {alpha.shape}"
        )
    if invalid_actions.shape != alpha.shape[:-1]:
        raise ValueError(
            "invalid_actions must match alpha without its outcome axis; "
            f"got {invalid_actions.shape} and {alpha.shape}"
        )
    if categorical_outcome is None:
        categorical_outcome = jnp.full(
            invalid_actions.shape,
            int(NO_OUTCOME),
            dtype=jnp.int8,
        )
    else:
        categorical_outcome = jnp.asarray(
            categorical_outcome,
            dtype=jnp.int8,
        )
        if categorical_outcome.shape != invalid_actions.shape:
            raise ValueError(
                "categorical_outcome must match invalid_actions; "
                f"got {categorical_outcome.shape} and "
                f"{invalid_actions.shape}"
            )
    if half_width < 1:
        raise ValueError(f"half_width must be >= 1, got {half_width}")

    dtype = jnp.result_type(alpha.dtype, jnp.float32)
    alpha = alpha.astype(dtype)
    legal = ~invalid_actions
    unresolved = legal & (categorical_outcome == int(NO_OUTCOME))
    certified_win = legal & (categorical_outcome == 1)
    grid_half_range, tail_range_clipped = _adaptive_grid_half_range(
        alpha,
        unresolved,
        tail_scale=tail_scale,
        min_half_range=min_half_range,
        max_half_range=max_half_range,
    )

    unit_grid = (
        jnp.arange(-half_width, half_width + 1, dtype=dtype)
        / jnp.asarray(half_width, dtype=dtype)
    )
    coordinate = grid_half_range[..., None] * unit_grid
    log_x, log_one_minus_x, log_jacobian = _sinh_logit_transform(
        coordinate
    )
    # Insert the action axis between arbitrary batch axes and the grid.
    log_x = log_x[..., None, :]
    log_one_minus_x = log_one_minus_x[..., None, :]
    log_jacobian = log_jacobian[..., None, :]
    safe_alpha = jnp.where(
        unresolved[..., None],
        jnp.maximum(alpha, jnp.finfo(dtype).tiny),
        1.0,
    )
    loss_alpha = safe_alpha[..., 0, None]
    win_alpha = safe_alpha[..., 1, None]
    log_density = (
        win_alpha * log_x
        + loss_alpha * log_one_minus_x
        - betaln(win_alpha, loss_alpha)
        + log_jacobian
    )
    density_scale = jnp.max(log_density, axis=-1, keepdims=True)
    density = jnp.exp(log_density - density_scale)
    density = jnp.where(unresolved[..., None], density, 0.0)

    grid_step = (
        grid_half_range / jnp.asarray(half_width, dtype=dtype)
    )
    cdf_increment = 0.5 * grid_step[..., None, None] * (
        density[..., :-1] + density[..., 1:]
    )
    unnormalized_cdf = jnp.concatenate(
        (
            jnp.zeros_like(density[..., :1]),
            jnp.cumsum(cdf_increment, axis=-1),
        ),
        axis=-1,
    )
    density_integral = unnormalized_cdf[..., -1:]
    density_log_integral = (
        density_scale[..., 0] + jnp.log(density_integral[..., 0])
    )
    density_log_integral = jnp.where(
        unresolved,
        density_log_integral,
        0.0,
    )
    safe_integral = jnp.where(
        density_integral > 0.0,
        density_integral,
        1.0,
    )
    cdf = unnormalized_cdf / safe_integral
    normalized_density = density / safe_integral

    # Prefix CDFs begin at exact zero.  Explicit zero accounting avoids
    # forming log(0) - log(0) in leave-one-action-out products.
    positive_cdf = cdf > 0.0
    finite_log_cdf = jnp.where(positive_cdf, jnp.log(cdf), 0.0)
    zero_factor = unresolved[..., None] & ~positive_cdf
    zero_count = jnp.sum(zero_factor, axis=-2, keepdims=True)
    finite_log_product = jnp.sum(
        jnp.where(unresolved[..., None], finite_log_cdf, 0.0),
        axis=-2,
        keepdims=True,
    )
    other_has_zero = (
        zero_count - zero_factor.astype(jnp.int32)
    ) > 0
    self_log = jnp.where(zero_factor, 0.0, finite_log_cdf)
    other_log_product = jnp.where(
        other_has_zero,
        -jnp.inf,
        finite_log_product - self_log,
    )
    integrand = normalized_density * jnp.exp(other_log_product)
    interval_contribution = (
        0.5
        * grid_step[..., None, None]
        * (integrand[..., :-1] + integrand[..., 1:])
    )
    interval_contribution = jnp.maximum(interval_contribution, 0.0)

    # The independently normalized CDFs define a discrete maximum
    # distribution.  Pin its endpoints so increments telescope exactly.
    cdf = jnp.clip(cdf, 0.0, 1.0)
    cdf = cdf.at[..., 0].set(0.0)
    cdf = cdf.at[..., -1].set(
        jnp.where(unresolved, 1.0, 0.0)
    )
    positive_cdf = cdf > 0.0
    log_joint_cdf = jnp.sum(
        jnp.where(
            unresolved[..., None],
            jnp.where(positive_cdf, jnp.log(cdf), -jnp.inf),
            0.0,
        ),
        axis=-2,
    )
    joint_cdf = jnp.exp(log_joint_cdf)
    joint_cdf = jnp.maximum.accumulate(joint_cdf, axis=-1)
    has_unresolved = jnp.any(unresolved, axis=-1)
    joint_cdf = joint_cdf.at[..., 0].set(
        jnp.where(has_unresolved, 0.0, 1.0)
    )
    joint_cdf = joint_cdf.at[..., -1].set(1.0)
    joint_increment = jnp.maximum(
        joint_cdf[..., 1:] - joint_cdf[..., :-1],
        0.0,
    )

    score_sum = jnp.sum(
        interval_contribution,
        axis=-2,
        keepdims=True,
    )
    score_positive = score_sum > 0.0
    unresolved_count = jnp.sum(
        unresolved,
        axis=-1,
        keepdims=True,
    )
    uniform = unresolved.astype(dtype) / jnp.maximum(
        unresolved_count,
        1,
    ).astype(dtype)
    allocation_weight = jnp.where(
        score_positive,
        interval_contribution
        / jnp.where(score_positive, score_sum, 1.0),
        uniform[..., None],
    )
    unresolved_probability = jnp.sum(
        allocation_weight * joint_increment[..., None, :],
        axis=-1,
    )
    unresolved_probability = jnp.where(
        unresolved,
        unresolved_probability,
        0.0,
    )
    fallback_interval_count = jnp.sum(
        (~jnp.squeeze(score_positive, axis=-2))
        & (joint_increment > 0.0),
        axis=-1,
        dtype=jnp.int32,
    )

    finalized = _finalize_policy(
        unresolved_probability,
        legal,
        unresolved,
        certified_win,
        categorical_outcome,
        dtype,
    )
    density_integral_finite = jnp.all(
        ~unresolved | jnp.isfinite(density_log_integral),
        axis=-1,
    )
    finite = (
        finalized.finite
        & jnp.isfinite(grid_half_range)
        & density_integral_finite
    )
    return BinaryPrefixQuadraturePolicy(
        policy=finalized.policy,
        raw_policy=finalized.raw_policy,
        raw_mass=finalized.raw_mass,
        normalization_error=finalized.normalization_error,
        finite=finite,
        grid_half_range=grid_half_range,
        tail_range_clipped=tail_range_clipped,
        density_log_integral=density_log_integral,
        fallback_interval_count=fallback_interval_count,
    )


__all__ = [
    "BinaryPrefixQuadraturePolicy",
    "binary_posterior_best_policy_prefix_quadrature",
]
