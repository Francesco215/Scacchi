"""Diagnostics and validated estimators for posterior-policy cache repair.

It provides four pieces developed for the E4 materiality benchmark:

* a deterministic binary-outcome posterior-best reference based on a fixed
  sinh-logit quadrature grid;
* a faster prefix-CDF approximation on the same kind of grid;
* a simplex-valued one-coordinate Rao--Blackwell estimator; and
* the exact conditional covariance of an ``M``-winner Monte Carlo cache
  estimate once the posterior-best probabilities are known.

The mass-conserving adaptive prefix-CDF estimator is optionally used by
production binary-outcome repair after passing the frozen-corpus and
complete-search gates.  The exact reference and Rao--Blackwell routines remain
offline diagnostics.  The production sampler has a vanishingly rare
bounded-work fallback, so benchmark reports must keep "exact Beta" and
"implemented sampler" results distinct.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, betaln, logsumexp
from jaxtyping import Array, Bool, Float, Int8, Int32

from .action_selection import sample_dirichlet
from .outcomes import NO_OUTCOME


class BinaryQuadraturePolicy(NamedTuple):
    """Raw and normalized posterior-best probabilities from quadrature."""

    policy: Float[Array, "*batch action"]
    raw_policy: Float[Array, "*batch action"]
    raw_mass: Float[Array, "*batch"]
    normalization_error: Float[Array, "*batch"]
    finite: Bool[Array, "*batch"]


class BinaryPrefixQuadraturePolicy(NamedTuple):
    """Prefix-CDF policy plus mass, range, and density-integral diagnostics.

    ``density_log_integral`` is the log trapezoidal integral of each legal
    unresolved transformed Beta density *before* its per-action
    normalization.  Large absolute values expose finite-range or coarse-grid
    leakage that the normalized winner policy and ``raw_mass`` can hide.
    Non-unresolved entries are zero.
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


class BinaryRaoBlackwellPolicy(NamedTuple):
    """Simplex-valued one-coordinate Rao--Blackwell policy estimate.

    ``coordinate_counts`` records how often each unresolved action was chosen
    as the analytically integrated coordinate.  It is exposed so the offline
    benchmark can verify that the deterministic cycle is balanced.
    """

    policy: Float[Array, "*batch action"]
    coordinate_counts: Int32[Array, "*batch action"]
    normalization_error: Float[Array, "*batch"]
    finite: Bool[Array, "*batch"]


class AnalyticCacheNoise(NamedTuple):
    """Conditional moments of the finite-population cache estimator."""

    posterior_mean_action_alpha: Float[Array, "*batch outcome"]
    exact_cache_alpha: Float[Array, "*batch outcome"]
    winner_action_covariance: Float[Array, "*batch outcome outcome2"]
    cache_covariance: Float[Array, "*batch outcome outcome2"]
    raw_alpha_mse: Float[Array, "*batch"]
    concentration_mse: Float[Array, "*batch"]
    semantic_mean_delta_mse: Float[Array, "*batch"]
    utility_delta_mse: Float[Array, "*batch"]
    raw_update_squared_l2: Float[Array, "*batch"]
    semantic_update_squared_l2: Float[Array, "*batch"]
    concentration_update_squared: Float[Array, "*batch"]
    repair_squared_l2: Float[Array, "*batch"]
    semantic_repair_squared_l2: Float[Array, "*batch"]
    concentration_repair_squared: Float[Array, "*batch"]
    raw_noise_to_update_ratio: Float[Array, "*batch"]
    semantic_noise_to_update_ratio: Float[Array, "*batch"]
    concentration_noise_to_update_ratio: Float[Array, "*batch"]
    raw_noise_to_repair_ratio: Float[Array, "*batch"]
    semantic_noise_to_repair_ratio: Float[Array, "*batch"]
    concentration_noise_to_repair_ratio: Float[Array, "*batch"]
    noise_fraction_of_expected_repair_step: Float[Array, "*batch"]
    gamma: Float[Array, "*batch"]


def _log_cosh(x: jax.Array) -> jax.Array:
    """Stable ``log(cosh(x))`` for the tails of a tanh--sinh grid."""

    absolute = jnp.abs(x)
    return absolute + jnp.log1p(jnp.exp(-2.0 * absolute)) - jnp.log(2.0)


def _sinh_logit_transform(
    t: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map arbitrary sinh-grid coordinates into stable endpoint logs."""

    z = jnp.sinh(t)
    log_x = -jax.nn.softplus(-z)
    log_one_minus_x = -jax.nn.softplus(z)
    # dz/dt = cosh(t); dx/dz is absorbed into the transformed Beta density.
    log_jacobian = _log_cosh(t)
    return z, log_x, log_one_minus_x, log_jacobian


def _sinh_logit_coordinates(
    dtype: jnp.dtype,
    *,
    half_width: int,
    step: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return logit coordinates and their sinh-map log Jacobian.

    Integrating in ``z=logit(x)`` avoids rounding x to an endpoint for small
    Beta components.  The additional ``z=sinh(t)`` map reaches both long
    exponential tails with a compact, fixed trapezoidal grid.
    """

    if half_width < 1:
        raise ValueError(f"half_width must be >= 1, got {half_width}")
    if not 0.0 < step:
        raise ValueError(f"step must be > 0, got {step}")

    t = jnp.arange(-half_width, half_width + 1, dtype=dtype)
    t = t * jnp.asarray(step, dtype=dtype)
    return _sinh_logit_transform(t)


def _sinh_logit_grid(
    dtype: jnp.dtype,
    *,
    half_width: int,
    step: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return z, endpoint logs, and fixed trapezoidal log weights."""

    z, log_x, log_one_minus_x, log_jacobian = (
        _sinh_logit_coordinates(
            dtype,
            half_width=half_width,
            step=step,
        )
    )
    log_weight = (
        jnp.log(jnp.asarray(step, dtype=dtype))
        + log_jacobian
    )
    return z, log_x, log_one_minus_x, log_weight


def _log1mexp(log_probability: jax.Array) -> jax.Array:
    """Stable log(1-exp(x)) for x<=0."""

    cutoff = -jnp.log(2.0)
    return jnp.where(
        log_probability < cutoff,
        jnp.log1p(-jnp.exp(log_probability)),
        jnp.log(-jnp.expm1(log_probability)),
    )


def _beta_log_cdf_from_logit(
    win_alpha: jax.Array,
    loss_alpha: jax.Array,
    z: jax.Array,
    log_x: jax.Array,
    log_one_minus_x: jax.Array,
) -> jax.Array:
    """Evaluate log Beta CDF without losing endpoint tail probabilities."""

    threshold = jnp.asarray(30.0, dtype=z.dtype)
    x = jnp.exp(log_x)
    direct = jnp.log(betainc(win_alpha, loss_alpha, x))

    # I_x(a,b) = x^a / (a B(a,b)) * (1 + O(x)).
    lower_tail = (
        win_alpha * log_x
        - jnp.log(win_alpha)
        - betaln(win_alpha, loss_alpha)
    )
    # 1-I_x(a,b) = I_(1-x)(b,a), with the analogous endpoint expansion.
    log_survival = (
        loss_alpha * log_one_minus_x
        - jnp.log(loss_alpha)
        - betaln(loss_alpha, win_alpha)
    )
    upper_tail = _log1mexp(jnp.minimum(log_survival, 0.0))
    return jnp.where(
        z < -threshold,
        lower_tail,
        jnp.where(z > threshold, upper_tail, direct),
    )


def _finalize_binary_quadrature_policy(
    unresolved_probability: jax.Array,
    legal: jax.Array,
    unresolved: jax.Array,
    certified_win: jax.Array,
    categorical_outcome: jax.Array,
    dtype: jnp.dtype,
) -> BinaryQuadraturePolicy:
    """Apply native categorical precedence and expose quadrature mass."""

    has_unresolved = jnp.any(unresolved, axis=-1)
    has_certified_win = jnp.any(certified_win, axis=-1)
    has_legal = jnp.any(legal, axis=-1)
    num_actions = legal.shape[-1]

    first_certified_win = jnp.argmax(
        certified_win.astype(jnp.int32),
        axis=-1,
    )
    win_policy = jax.nn.one_hot(
        first_certified_win,
        num_actions,
        dtype=dtype,
    )
    win_policy = jnp.where(legal, win_policy, 0.0)

    # This branch is relevant only when no unresolved action remains.  For
    # binary outcomes, choose the highest categorical outcome and preserve
    # jnp.argmax's first-index tie rule, exactly as the production selector.
    categorical_score = jnp.where(
        legal,
        categorical_outcome.astype(dtype),
        -jnp.inf,
    )
    categorical_best = jnp.argmax(categorical_score, axis=-1)
    categorical_policy = jax.nn.one_hot(
        categorical_best,
        num_actions,
        dtype=dtype,
    )
    categorical_policy = jnp.where(legal, categorical_policy, 0.0)

    raw_policy = jnp.where(
        has_certified_win[..., None],
        win_policy,
        jnp.where(
            has_unresolved[..., None],
            unresolved_probability,
            categorical_policy,
        ),
    )
    raw_policy = jnp.where(legal, raw_policy, 0.0)
    raw_mass = jnp.sum(raw_policy, axis=-1)
    normalization_error = jnp.abs(raw_mass - has_legal.astype(dtype))
    safe_mass = jnp.where(raw_mass > 0, raw_mass, 1.0)
    policy = raw_policy / safe_mass[..., None]
    policy = jnp.where(has_legal[..., None], policy, 0.0)
    finite = (
        jnp.all(jnp.isfinite(policy), axis=-1)
        & jnp.all(jnp.isfinite(raw_policy), axis=-1)
        & jnp.isfinite(raw_mass)
        & jnp.isfinite(normalization_error)
    )
    return BinaryQuadraturePolicy(
        policy=policy,
        raw_policy=raw_policy,
        raw_mass=raw_mass,
        normalization_error=normalization_error,
        finite=finite,
    )


def _adaptive_prefix_grid_half_range(
    alpha: jax.Array,
    unresolved: jax.Array,
    *,
    tail_scale: float,
    min_half_range: float,
    max_half_range: float,
) -> tuple[jax.Array, jax.Array]:
    """Choose one shared symmetric t range per action context."""

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
    tail_range_clipped = has_unresolved & (
        requested > jnp.asarray(max_half_range, dtype=dtype)
    )
    return half_range, tail_range_clipped


def binary_posterior_best_policy_quadrature(
    alpha: Float[Array, "*batch action 2"],
    invalid_actions: Bool[Array, "*batch action"],
    categorical_outcome: Int8[Array, "*batch action"] | None = None,
    *,
    half_width: int = 160,
    step: float = 0.1,
) -> BinaryQuadraturePolicy:
    """Integrate posterior-best probabilities for independent Beta actions.

    Outcome index zero is loss and index one is win, matching the native Hex
    representation.  A certified win deterministically dominates unresolved
    continuous actions.  Certified losses have zero winning probability while
    at least one unresolved action remains.  If every legal action is
    categorical, the production selector's first-index tie rule is preserved.

    The returned ``raw_policy`` is never silently normalized.  ``policy`` is
    its normalized version for downstream moment calculations, while
    ``normalization_error`` remains an explicit quadrature-accuracy diagnostic.
    """

    alpha = jnp.asarray(alpha)
    invalid_actions = jnp.asarray(invalid_actions)
    if alpha.shape[-1] != 2:
        raise ValueError(
            "binary quadrature requires exactly two outcomes; "
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

    dtype = jnp.result_type(alpha.dtype, jnp.float32)
    alpha = alpha.astype(dtype)
    legal = ~invalid_actions
    unresolved = legal & (categorical_outcome == int(NO_OUTCOME))
    certified_win = legal & (categorical_outcome == 1)

    z, log_x, log_one_minus_x, log_weight = _sinh_logit_grid(
        dtype,
        half_width=half_width,
        step=step,
    )
    loss_alpha = alpha[..., 0, None]
    win_alpha = alpha[..., 1, None]
    log_cdf = _beta_log_cdf_from_logit(
        win_alpha,
        loss_alpha,
        z,
        log_x,
        log_one_minus_x,
    )

    # Products of all other CDFs need explicit zero accounting.  Subtracting
    # one action from a total log-product would otherwise form -inf - -inf at
    # endpoint nodes.
    positive_cdf = jnp.isfinite(log_cdf)
    finite_log_cdf = jnp.where(positive_cdf, log_cdf, 0.0)
    cdf_factor = unresolved[..., None]
    zero_factor = cdf_factor & ~positive_cdf
    zero_count = jnp.sum(zero_factor, axis=-2, keepdims=True)
    finite_log_product = jnp.sum(
        jnp.where(cdf_factor, finite_log_cdf, 0.0),
        axis=-2,
        keepdims=True,
    )
    self_zero = zero_factor.astype(jnp.int32)
    other_has_zero = (zero_count - self_zero) > 0
    self_log = jnp.where(zero_factor, 0.0, finite_log_cdf)
    other_log_product = jnp.where(
        other_has_zero,
        -jnp.inf,
        finite_log_product - self_log,
    )

    # Beta density times dx/dz simplifies by adding one to both endpoint
    # exponents.
    log_density = (
        win_alpha * log_x
        + loss_alpha * log_one_minus_x
        - betaln(win_alpha, loss_alpha)
    )
    log_integrand = log_density + other_log_product + log_weight
    log_integrand = jnp.where(
        unresolved[..., None],
        log_integrand,
        -jnp.inf,
    )
    unresolved_probability = jnp.exp(logsumexp(log_integrand, axis=-1))
    unresolved_probability = jnp.where(
        unresolved,
        unresolved_probability,
        0.0,
    )
    return _finalize_binary_quadrature_policy(
        unresolved_probability,
        legal,
        unresolved,
        certified_win,
        categorical_outcome,
        dtype,
    )


def binary_posterior_best_policy_prefix_quadrature(
    alpha: Float[Array, "*batch action 2"],
    invalid_actions: Bool[Array, "*batch action"],
    categorical_outcome: Int8[Array, "*batch action"] | None = None,
    *,
    half_width: int = 20,
    adaptive_range: bool = True,
    fixed_step: float = 0.3,
    tail_scale: float = 8.0,
    min_half_range: float = 6.0,
    max_half_range: float = 11.0,
    mass_conserving: bool = True,
) -> BinaryPrefixQuadraturePolicy:
    """Approximate binary posterior-best probabilities with prefix CDFs.

    Each unresolved action's transformed Beta density is evaluated on one
    fixed sinh-logit grid.  A trapezoidal prefix sum supplies every CDF value,
    after which the usual

    ``P(a is best) = integral f_a(x) product_{b != a} F_b(x) dx``

    is evaluated on that same grid.  Densities are independently normalized
    over the finite grid before forming CDFs.  This makes the fixed-order
    approximation robust in float32 while ``raw_mass`` and
    ``normalization_error`` continue to expose winner-integral discretization
    error.

    The grid has ``2 * half_width + 1`` points.  By default, one symmetric
    half-range ``T`` is shared by every unresolved action in a context:

    ``T = clip(asinh(tail_scale / min_alpha), min_half_range, max_half_range)``.

    The Q41 defaults use ``tail_scale=8`` and clip ``T`` to ``[6, 11]``.
    ``tail_range_clipped`` reports contexts that requested more than the
    maximum; the upper clip begins around alpha ``8 / sinh(11) = 2.67e-4``.
    Current synthetic validation covers positive components down to ``1e-3``,
    not every alpha above the clipping threshold.  Set ``adaptive_range=False``
    to recover a caller-controlled fixed grid with
    ``T = half_width * fixed_step``.

    Point count and range are independent accuracy controls: refining a fixed
    range cannot recover omitted endpoint tails.  Prefix normalization can
    hide density mass outside the grid, so neither finite output nor small
    winner ``raw_mass`` error proves safety outside a measured alpha envelope.
    Any proposed use on a new corpus must first gate on its minimum legal alpha
    and compare against the exact reference.

    With the default ``mass_conserving=True``, each interval's increment in
    the CDF of the maximum, ``Delta product_a F_a``, is allocated among
    actions in proportion to their nonnegative local trapezoidal winner
    contributions.  These increments telescope to one, so the raw winner
    vector conserves probability up to floating-point roundoff.  The rule is
    permutation equivariant and remains ``O(AQ)``.  A symmetric uniform
    allocation is used only when every local contribution underflows while a
    positive maximum-CDF increment remains; ``fallback_interval_count``
    exposes that event.  Set ``mass_conserving=False`` to reproduce the
    original globally normalized prefix-integral prototype.

    Native categorical and invalid-action behavior exactly matches
    :func:`binary_posterior_best_policy_quadrature`: certified wins dominate,
    certified losses cannot beat a remaining continuous action, all-
    categorical rows use the production first-index tie rule, and all-invalid
    rows return the zero policy.

    The mass-conserving adaptive default is the validated optional production
    estimator for binary-outcome repair.  Fixed-range and non-conserving modes
    remain benchmark controls.
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

    dtype = jnp.result_type(alpha.dtype, jnp.float32)
    alpha = alpha.astype(dtype)
    legal = ~invalid_actions
    unresolved = legal & (categorical_outcome == int(NO_OUTCOME))
    certified_win = legal & (categorical_outcome == 1)

    if half_width < 1:
        raise ValueError(f"half_width must be >= 1, got {half_width}")
    if adaptive_range:
        grid_half_range, tail_range_clipped = (
            _adaptive_prefix_grid_half_range(
                alpha,
                unresolved,
                tail_scale=tail_scale,
                min_half_range=min_half_range,
                max_half_range=max_half_range,
            )
        )
    else:
        if not math.isfinite(fixed_step) or fixed_step <= 0.0:
            raise ValueError(
                f"fixed_step must be finite and > 0, got {fixed_step}"
            )
        grid_half_range = jnp.full(
            alpha.shape[:-2],
            float(half_width) * fixed_step,
            dtype=dtype,
        )
        tail_range_clipped = jnp.zeros(
            alpha.shape[:-2],
            dtype=jnp.bool_,
        )

    unit_grid = (
        jnp.arange(-half_width, half_width + 1, dtype=dtype)
        / jnp.asarray(half_width, dtype=dtype)
    )
    t = grid_half_range[..., None] * unit_grid
    _, log_x, log_one_minus_x, log_jacobian = (
        _sinh_logit_transform(t)
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
    log_transformed_density = (
        win_alpha * log_x
        + loss_alpha * log_one_minus_x
        - betaln(win_alpha, loss_alpha)
        + log_jacobian
    )
    density_scale = jnp.max(
        log_transformed_density,
        axis=-1,
        keepdims=True,
    )
    density = jnp.exp(log_transformed_density - density_scale)
    density = jnp.where(unresolved[..., None], density, 0.0)

    grid_step = grid_half_range / jnp.asarray(half_width, dtype=dtype)
    cdf_increments = 0.5 * grid_step[..., None, None] * (
        density[..., :-1] + density[..., 1:]
    )
    unnormalized_cdf = jnp.concatenate(
        (
            jnp.zeros_like(density[..., :1]),
            jnp.cumsum(cdf_increments, axis=-1),
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
        density_integral > 0,
        density_integral,
        1.0,
    )
    cdf = unnormalized_cdf / safe_integral
    normalized_density = density / safe_integral

    # Prefix CDFs start at exact zero.  Track zero factors explicitly so
    # leave-one-action-out products never form log(0) - log(0).
    positive_cdf = cdf > 0
    finite_log_cdf = jnp.where(positive_cdf, jnp.log(cdf), 0.0)
    cdf_factor = unresolved[..., None]
    zero_factor = cdf_factor & ~positive_cdf
    zero_count = jnp.sum(zero_factor, axis=-2, keepdims=True)
    finite_log_product = jnp.sum(
        jnp.where(cdf_factor, finite_log_cdf, 0.0),
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
    fallback_interval_count = jnp.zeros(
        alpha.shape[:-2],
        dtype=jnp.int32,
    )
    if mass_conserving:
        # The independently normalized CDFs define a discrete approximation
        # to the maximum distribution.  Pin its mathematical endpoints so
        # the interval increments telescope exactly in float32.
        cdf = jnp.clip(cdf, 0.0, 1.0)
        cdf = cdf.at[..., 0].set(0.0)
        cdf = cdf.at[..., -1].set(
            jnp.where(unresolved, 1.0, 0.0)
        )
        positive_cdf = cdf > 0.0
        log_joint_cdf = jnp.sum(
            jnp.where(
                unresolved[..., None],
                jnp.where(
                    positive_cdf,
                    jnp.log(cdf),
                    -jnp.inf,
                ),
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
        fallback_interval_count = jnp.sum(
            (~jnp.squeeze(score_positive, axis=-2))
            & (joint_increment > 0.0),
            axis=-1,
            dtype=jnp.int32,
        )
    else:
        unresolved_probability = jnp.sum(
            interval_contribution,
            axis=-1,
        )
    unresolved_probability = jnp.where(
        unresolved,
        unresolved_probability,
        0.0,
    )
    result = _finalize_binary_quadrature_policy(
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
        result.finite
        & jnp.isfinite(grid_half_range)
        & density_integral_finite
    )
    return BinaryPrefixQuadraturePolicy(
        policy=result.policy,
        raw_policy=result.raw_policy,
        raw_mass=result.raw_mass,
        normalization_error=result.normalization_error,
        finite=finite,
        grid_half_range=grid_half_range,
        tail_range_clipped=tail_range_clipped,
        density_log_integral=density_log_integral,
        fallback_interval_count=fallback_interval_count,
    )


def binary_posterior_best_policy_rao_blackwell(
    rng_key: jax.Array,
    alpha: Float[Array, "*batch action 2"],
    invalid_actions: Bool[Array, "*batch action"],
    categorical_outcome: Int8[Array, "*batch action"] | None = None,
    *,
    num_samples: int,
    sample_chunk_size: int | None = None,
) -> BinaryRaoBlackwellPolicy:
    """Estimate a binary posterior-best policy by conditional integration.

    For each sample, choose one unresolved action ``j`` independently of the
    Dirichlet draws.  Draw every competing unresolved action, let ``b`` be the
    sampled best competitor with value ``m``, and emit the simplex vector

    ``q_j = 1 - F_j(m), q_b = F_j(m)``.

    Conditional on the competitors this exactly integrates the Beta random
    variable for action ``j``.  Every coordinate of ``q`` is unbiased for its
    posterior-best probability, while ``sum(q) == 1`` sample by sample.  The
    integrated coordinate follows a balanced deterministic cycle; a random
    cycle offset is drawn from a key split that is independent of all
    competitor draws.

    This function is benchmark-only.  It intentionally is not imported by
    ``posterior_updates`` and does not alter traversal or production repair.
    Competitors use the same bounded-work Dirichlet primitive as production,
    so reports must continue to distinguish this implemented-sampler estimate
    from the exact-Beta quadrature reference.
    """

    if num_samples < 1:
        raise ValueError(
            f"num_samples must be >= 1, got {num_samples}"
        )
    if sample_chunk_size is None:
        sample_chunk_size = num_samples
    if sample_chunk_size < 1:
        raise ValueError(
            "sample_chunk_size must be >= 1, "
            f"got {sample_chunk_size}"
        )
    sample_chunk_size = min(sample_chunk_size, num_samples)

    alpha = jnp.asarray(alpha)
    invalid_actions = jnp.asarray(invalid_actions)
    if alpha.shape[-1] != 2:
        raise ValueError(
            "binary Rao--Blackwell estimation requires exactly two outcomes; "
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

    dtype = jnp.result_type(alpha.dtype, jnp.float32)
    alpha = alpha.astype(dtype)
    legal = ~invalid_actions
    unresolved = legal & (categorical_outcome == int(NO_OUTCOME))
    unresolved_count = jnp.sum(unresolved, axis=-1)
    safe_unresolved_count = jnp.maximum(unresolved_count, 1)
    has_unresolved = unresolved_count > 0
    certified_win = legal & (categorical_outcome == 1)
    has_certified_win = jnp.any(certified_win, axis=-1)
    has_legal = jnp.any(legal, axis=-1)
    estimate_active = has_unresolved & ~has_certified_win
    num_actions = alpha.shape[-2]

    sample_key, offset_key = jax.random.split(rng_key)
    cycle_offset = jax.random.randint(
        offset_key,
        unresolved_count.shape,
        minval=0,
        maxval=safe_unresolved_count,
        dtype=jnp.int32,
    )
    unresolved_rank = jnp.cumsum(
        unresolved.astype(jnp.int32),
        axis=-1,
    ) - 1

    def one_sample(
        key: jax.Array,
        sample_index: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        target_rank = (
            cycle_offset + sample_index.astype(jnp.int32)
        ) % safe_unresolved_count
        integrated = (
            estimate_active[..., None]
            & unresolved
            & (unresolved_rank == target_rank[..., None])
        )
        integrated_index = jnp.argmax(
            integrated.astype(jnp.int32),
            axis=-1,
        )

        sampled_win_probability = sample_dirichlet(key, alpha)[..., 1]
        competitor = unresolved & ~integrated
        competitor_score = jnp.where(
            competitor,
            sampled_win_probability,
            -jnp.inf,
        )
        best_competitor = jnp.argmax(competitor_score, axis=-1)
        best_value = jnp.take_along_axis(
            competitor_score,
            best_competitor[..., None],
            axis=-1,
        )[..., 0]

        integrated_alpha = jnp.take_along_axis(
            alpha,
            integrated_index[..., None, None],
            axis=-2,
        )[..., 0, :]
        cdf_at_best = betainc(
            integrated_alpha[..., 1],
            integrated_alpha[..., 0],
            jnp.clip(best_value, 0.0, 1.0),
        )
        has_competitor = unresolved_count > 1
        cdf_at_best = jnp.where(has_competitor, cdf_at_best, 0.0)

        integrated_mass = integrated.astype(dtype)
        competitor_mass = jax.nn.one_hot(
            best_competitor,
            num_actions,
            dtype=dtype,
        )
        policy_sample = (
            (1.0 - cdf_at_best)[..., None] * integrated_mass
            + cdf_at_best[..., None] * competitor_mass
        )
        policy_sample = jnp.where(
            estimate_active[..., None],
            policy_sample,
            0.0,
        )
        return policy_sample, integrated.astype(jnp.int32)

    num_chunks = (
        num_samples + sample_chunk_size - 1
    ) // sample_chunk_size
    padded_count = num_chunks * sample_chunk_size
    draw_key, padding_key = jax.random.split(sample_key)
    sample_keys = jax.random.split(draw_key, num_samples)
    pad_count = padded_count - num_samples
    if pad_count:
        sample_keys = jnp.concatenate(
            [
                sample_keys,
                jax.random.split(padding_key, pad_count),
            ],
            axis=0,
        )
    sample_keys = sample_keys.reshape(
        (num_chunks, sample_chunk_size) + sample_keys.shape[1:]
    )
    sample_indices = jnp.arange(padded_count, dtype=jnp.int32).reshape(
        (num_chunks, sample_chunk_size)
    )
    valid_samples = (
        jnp.arange(padded_count) < num_samples
    ).reshape((num_chunks, sample_chunk_size))

    initial_policy = jnp.zeros(alpha.shape[:-1], dtype=dtype)
    initial_counts = jnp.zeros(
        invalid_actions.shape,
        dtype=jnp.int32,
    )

    def sample_chunk(
        totals: tuple[jax.Array, jax.Array],
        chunk: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        total_policy, total_counts = totals
        keys, indices, valid = chunk
        policies, counts = jax.vmap(one_sample)(keys, indices)
        policy_weight = valid.astype(dtype).reshape(
            (sample_chunk_size,)
            + (1,) * (policies.ndim - 1)
        )
        count_weight = valid.astype(jnp.int32).reshape(
            (sample_chunk_size,)
            + (1,) * (counts.ndim - 1)
        )
        return (
            total_policy
            + jnp.sum(policies * policy_weight, axis=0),
            total_counts
            + jnp.sum(
                counts * count_weight,
                axis=0,
                dtype=jnp.int32,
            ),
        ), None

    (total_policy, coordinate_counts), _ = jax.lax.scan(
        sample_chunk,
        (initial_policy, initial_counts),
        (sample_keys, sample_indices, valid_samples),
    )
    estimated_policy = total_policy / jnp.asarray(
        num_samples,
        dtype=dtype,
    )

    first_certified_win = jnp.argmax(
        certified_win.astype(jnp.int32),
        axis=-1,
    )
    win_policy = jax.nn.one_hot(
        first_certified_win,
        num_actions,
        dtype=dtype,
    )
    win_policy = jnp.where(legal, win_policy, 0.0)
    categorical_score = jnp.where(
        legal,
        categorical_outcome.astype(dtype),
        -jnp.inf,
    )
    categorical_best = jnp.argmax(categorical_score, axis=-1)
    categorical_policy = jax.nn.one_hot(
        categorical_best,
        num_actions,
        dtype=dtype,
    )
    categorical_policy = jnp.where(legal, categorical_policy, 0.0)

    policy = jnp.where(
        has_certified_win[..., None],
        win_policy,
        jnp.where(
            estimate_active[..., None],
            estimated_policy,
            categorical_policy,
        ),
    )
    policy = jnp.where(legal, policy, 0.0)
    policy_mass = jnp.sum(policy, axis=-1)
    normalization_error = jnp.abs(
        policy_mass - has_legal.astype(dtype)
    )
    finite = (
        jnp.all(jnp.isfinite(policy), axis=-1)
        & jnp.isfinite(normalization_error)
    )
    return BinaryRaoBlackwellPolicy(
        policy=policy,
        coordinate_counts=coordinate_counts,
        normalization_error=normalization_error,
        finite=finite,
    )


def _quadratic_form(
    vector: jax.Array,
    matrix: jax.Array,
) -> jax.Array:
    return jnp.einsum("...i,...ij,...j->...", vector, matrix, vector)


def analytic_cache_noise(
    posterior_best_policy: Float[Array, "*batch action"],
    cache_action_alpha: Float[Array, "*batch action outcome"],
    value_prior: Float[Array, "*batch outcome"],
    n_down: Array,
    *,
    previous_value_alpha: Float[Array, "*batch outcome"] | None = None,
    kappa: float,
    num_samples: int,
    ratio_epsilon: float = 1e-12,
) -> AnalyticCacheNoise:
    """Return exact conditional MSE for an ``M``-winner cache estimate.

    If ``A^*`` is one posterior-best action and ``A_{A^*}`` is its cache
    alpha, the finite estimator averages ``M`` independent copies.  Therefore

    ``Cov(C_hat | context) = gamma**2 Cov(A_{A^*}) / M``.

    The semantic quantities use a first-order delta method around the exact
    cache alpha.  Raw-alpha and concentration moments are exact.
    """

    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError(f"kappa must be finite and > 0, got {kappa}")
    if num_samples < 1:
        raise ValueError(
            f"num_samples must be >= 1, got {num_samples}"
        )

    policy = jnp.asarray(posterior_best_policy)
    action_alpha = jnp.asarray(cache_action_alpha)
    value_prior = jnp.asarray(value_prior)
    if previous_value_alpha is None:
        previous_value_alpha = value_prior
    else:
        previous_value_alpha = jnp.asarray(previous_value_alpha)
    n_down = jnp.asarray(n_down)
    if action_alpha.shape[:-1] != policy.shape:
        raise ValueError(
            "cache_action_alpha must append one outcome axis to policy; "
            f"got {action_alpha.shape} and {policy.shape}"
        )
    if value_prior.shape != action_alpha.shape[:-2] + (action_alpha.shape[-1],):
        raise ValueError(
            "value_prior must match the batch and outcome axes; "
            f"got {value_prior.shape} and {action_alpha.shape}"
        )
    if previous_value_alpha.shape != value_prior.shape:
        raise ValueError(
            "previous_value_alpha must match value_prior; "
            f"got {previous_value_alpha.shape} and {value_prior.shape}"
        )
    if n_down.shape != policy.shape[:-1]:
        raise ValueError(
            "n_down must match the policy batch axes; "
            f"got {n_down.shape} and {policy.shape}"
        )

    dtype = jnp.result_type(
        policy.dtype,
        action_alpha.dtype,
        value_prior.dtype,
        previous_value_alpha.dtype,
        jnp.float32,
    )
    policy = policy.astype(dtype)
    action_alpha = action_alpha.astype(dtype)
    value_prior = value_prior.astype(dtype)
    previous_value_alpha = previous_value_alpha.astype(dtype)
    n_down = n_down.astype(dtype)
    policy_mass = jnp.sum(policy, axis=-1, keepdims=True)
    policy = policy / jnp.where(policy_mass > 0, policy_mass, 1.0)

    posterior_mean = jnp.sum(
        policy[..., None] * action_alpha,
        axis=-2,
    )
    centered_action_alpha = action_alpha - posterior_mean[..., None, :]
    winner_covariance = jnp.einsum(
        "...a,...ai,...aj->...ij",
        policy,
        centered_action_alpha,
        centered_action_alpha,
    )
    # Remove only roundoff-scale asymmetry; negative variances remain visible
    # to tests and benchmark validity checks rather than being silently
    # clamped.
    winner_covariance = 0.5 * (
        winner_covariance
        + jnp.swapaxes(winner_covariance, -1, -2)
    )

    gamma = n_down / (jnp.asarray(kappa, dtype=dtype) + n_down)
    exact_cache = (
        (1.0 - gamma)[..., None] * value_prior
        + gamma[..., None] * posterior_mean
    )
    cache_covariance = (
        (gamma**2 / jnp.asarray(num_samples, dtype=dtype))[..., None, None]
        * winner_covariance
    )
    raw_alpha_mse = jnp.trace(cache_covariance, axis1=-2, axis2=-1)
    ones = jnp.ones((action_alpha.shape[-1],), dtype=dtype)
    concentration_mse = _quadratic_form(ones, cache_covariance)

    concentration = jnp.sum(exact_cache, axis=-1)
    safe_concentration = jnp.maximum(
        concentration,
        jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype),
    )
    semantic_mean = exact_cache / safe_concentration[..., None]
    identity = jnp.eye(action_alpha.shape[-1], dtype=dtype)
    mean_jacobian = (
        identity
        - semantic_mean[..., :, None] * ones
    ) / safe_concentration[..., None, None]
    semantic_covariance = jnp.einsum(
        "...ij,...jk,...lk->...il",
        mean_jacobian,
        cache_covariance,
        mean_jacobian,
    )
    semantic_mean_delta_mse = jnp.trace(
        semantic_covariance,
        axis1=-2,
        axis2=-1,
    )

    utility_coefficients = jnp.zeros(
        (action_alpha.shape[-1],),
        dtype=dtype,
    ).at[0].set(-1.0).at[-1].set(1.0)
    utility_numerator = jnp.sum(
        exact_cache * utility_coefficients,
        axis=-1,
    )
    utility_gradient = (
        utility_coefficients * safe_concentration[..., None]
        - utility_numerator[..., None] * ones
    ) / safe_concentration[..., None] ** 2
    utility_delta_mse = _quadratic_form(
        utility_gradient,
        cache_covariance,
    )

    raw_update_squared_l2 = jnp.sum(
        (exact_cache - value_prior) ** 2,
        axis=-1,
    )
    prior_mean = value_prior / jnp.sum(
        value_prior,
        axis=-1,
        keepdims=True,
    )
    semantic_update_squared_l2 = jnp.sum(
        (semantic_mean - prior_mean) ** 2,
        axis=-1,
    )
    concentration_update_squared = (
        jnp.sum(exact_cache, axis=-1)
        - jnp.sum(value_prior, axis=-1)
    ) ** 2
    repair_squared_l2 = jnp.sum(
        (exact_cache - previous_value_alpha) ** 2,
        axis=-1,
    )
    previous_mean = previous_value_alpha / jnp.sum(
        previous_value_alpha,
        axis=-1,
        keepdims=True,
    )
    semantic_repair_squared_l2 = jnp.sum(
        (semantic_mean - previous_mean) ** 2,
        axis=-1,
    )
    concentration_repair_squared = (
        jnp.sum(exact_cache, axis=-1)
        - jnp.sum(previous_value_alpha, axis=-1)
    ) ** 2

    epsilon = jnp.asarray(ratio_epsilon, dtype=dtype)

    def ratio(noise: jax.Array, signal: jax.Array) -> jax.Array:
        return jnp.where(signal > epsilon, noise / signal, jnp.nan)

    return AnalyticCacheNoise(
        posterior_mean_action_alpha=posterior_mean,
        exact_cache_alpha=exact_cache,
        winner_action_covariance=winner_covariance,
        cache_covariance=cache_covariance,
        raw_alpha_mse=raw_alpha_mse,
        concentration_mse=concentration_mse,
        semantic_mean_delta_mse=semantic_mean_delta_mse,
        utility_delta_mse=utility_delta_mse,
        raw_update_squared_l2=raw_update_squared_l2,
        semantic_update_squared_l2=semantic_update_squared_l2,
        concentration_update_squared=concentration_update_squared,
        repair_squared_l2=repair_squared_l2,
        semantic_repair_squared_l2=semantic_repair_squared_l2,
        concentration_repair_squared=concentration_repair_squared,
        raw_noise_to_update_ratio=ratio(
            raw_alpha_mse,
            raw_update_squared_l2,
        ),
        semantic_noise_to_update_ratio=ratio(
            semantic_mean_delta_mse,
            semantic_update_squared_l2,
        ),
        concentration_noise_to_update_ratio=ratio(
            concentration_mse,
            concentration_update_squared,
        ),
        raw_noise_to_repair_ratio=ratio(
            raw_alpha_mse,
            repair_squared_l2,
        ),
        semantic_noise_to_repair_ratio=ratio(
            semantic_mean_delta_mse,
            semantic_repair_squared_l2,
        ),
        concentration_noise_to_repair_ratio=ratio(
            concentration_mse,
            concentration_repair_squared,
        ),
        noise_fraction_of_expected_repair_step=ratio(
            raw_alpha_mse,
            repair_squared_l2 + raw_alpha_mse,
        ),
        gamma=gamma,
    )


__all__ = [
    "AnalyticCacheNoise",
    "BinaryPrefixQuadraturePolicy",
    "BinaryQuadraturePolicy",
    "BinaryRaoBlackwellPolicy",
    "analytic_cache_noise",
    "binary_posterior_best_policy_prefix_quadrature",
    "binary_posterior_best_policy_quadrature",
    "binary_posterior_best_policy_rao_blackwell",
]
