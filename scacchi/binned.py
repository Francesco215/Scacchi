"""Categorical beliefs over win probability on evenly spaced atoms.

These are masses over a latent probability, not probabilities of W/L outcomes.
All tree arithmetic uses float32 even when the neural trunk uses bfloat16.
"""

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .types import BinConfig


class BinnedOutput(NamedTuple):
    logits: jax.Array
    value_probs: jax.Array
    action_probs: jax.Array
    support: jax.Array
    # Curiosity head: for each move, the bits expected at the position it leads
    # to and the log-variance of that prediction.
    next_bits: jax.Array | None = None
    next_bits_log_variance: jax.Array | None = None


def support(config: BinConfig) -> np.ndarray:
    """Evenly spaced atoms on [0, 1], including 0, 0.5 and 1."""
    return np.linspace(0.0, 1.0, config.count, dtype=np.float64).astype(np.float32)


def initial_logits(config: BinConfig) -> np.ndarray:
    """Logits of a uniform density over win probability.

    Each atom stands for the cell of points nearest to it, so its mass is the
    cell width: the two endpoint cells are half as wide as the others.
    """
    points = support(config).astype(np.float64)
    edges = np.concatenate(([0.0], (points[:-1] + points[1:]) / 2, [1.0]))
    return np.log(np.diff(edges)).astype(np.float32)


def from_logits(logits: jax.Array) -> jax.Array:
    return jax.nn.softmax(logits.astype(jnp.float32), axis=-1)


def mean(probs: jax.Array, points: jax.Array) -> jax.Array:
    return jnp.sum(probs * points, axis=-1)


def categorical_edges(probs: jax.Array, outcomes: jax.Array) -> jax.Array:
    """Write proven edges as point masses on the first (loss) or last (win) atom."""
    index = jnp.asarray(outcomes, dtype=jnp.int32) * (probs.shape[-1] - 1)
    exact = jax.nn.one_hot(index, probs.shape[-1], dtype=probs.dtype)
    return jnp.where((outcomes >= 0)[..., None], exact, probs)


def _rank_cells(num_actions: int, lattice: int) -> list[tuple[int, int, int]]:
    """``(first, last, cells)`` for moves ranked by weight, largest first.

    Weights that sum to at most one put the r-th largest at most at 1/r, so
    ``weight * X`` for ranks ``first..last`` reaches at most cell
    ``(lattice - 1) / first``. Two spare cells cover the split onto the next
    cell and a sum that exceeds one by rounding.
    """
    groups, first = [], 2
    while first <= num_actions:
        last = min(2 * first - 2, num_actions)
        groups.append((first, last, min(lattice, (lattice - 1) // first + 3)))
        first = last + 1
    return groups


def weighted_sum(probs: jax.Array, policy: jax.Array, points: jax.Array, *, lattice: int = 129) -> jax.Array:
    """Law of ``sum_a policy[a] * X_a`` for independent edges ``X_a ~ probs[a]``.

    This is the law of total probability for the win probability under the
    policy, read as an identity between random variables rather than as a
    mixture of their distributions. The mean is exact for any dependence
    between edges; the shape assumes they are independent.

    Each ``policy[a] * X_a`` is first written on a fine uniform lattice, then
    the lattice laws are convolved and the result is moved onto ``points``.
    Both moves split a point mass between its two neighbours, which keeps the
    mean and adds at most a quarter of the squared spacing to the variance.
    ``policy`` must sum to at most one.

    Convolution does not depend on the order of the edges, so they are taken
    by decreasing weight: the r-th law then spans at most ``(lattice - 1) / r``
    cells and only that many shifts of the partial sum are needed.
    """
    dtype = jnp.result_type(probs, jnp.float32)
    points = jnp.asarray(points, dtype=dtype)
    fine = jnp.linspace(0.0, 1.0, lattice, dtype=dtype)
    # Moves each lattice cell to its two neighbouring atoms: [lattice, atoms].
    upper = jnp.clip(jnp.searchsorted(points, fine, side="right"), 1, points.shape[0] - 1)
    fraction = jnp.clip((fine - points[upper - 1]) / (points[upper] - points[upper - 1]), 0.0, 1.0)
    onto_points = (
        jax.nn.one_hot(upper - 1, points.shape[0], dtype=dtype) * (1.0 - fraction)[:, None]
        + jax.nn.one_hot(upper, points.shape[0], dtype=dtype) * fraction[:, None]
    )
    groups = _rank_cells(probs.shape[-2], lattice)

    def on_lattice(edges, weights, cells):
        """Laws of ``weight * X`` on the first ``cells`` lattice cells."""
        # weight * points[j] is a real number of cells; split it between the
        # two neighbouring cells. The last cell keeps anything at or past it.
        shift = weights[:, None] * points * (lattice - 1)
        whole = jnp.floor(shift).astype(jnp.int32)
        part = shift - whole
        cell = jnp.arange(cells)
        lower = jnp.where(jnp.minimum(whole, cells - 1)[..., None] == cell, (1.0 - part)[..., None], 0.0)
        higher = jnp.where(jnp.minimum(whole + 1, cells - 1)[..., None] == cell, part[..., None], 0.0)
        return jnp.sum(edges[..., None] * (lower + higher), axis=-2)

    def single(edges, weights):
        order = jnp.argsort(-weights)
        edges, weights = edges[order], weights[order]
        partial = on_lattice(edges[:1], weights[:1], lattice)[0]
        for first, last, cells in groups:
            def add(partial, law, cells=cells):
                spare = jnp.zeros(cells - 1, dtype=dtype)
                padded = jnp.concatenate((spare, partial, spare))
                moved = sum(law[m] * padded[cells - 1 - m: 2 * (cells - 1) - m + lattice] for m in range(cells))
                # Sums at or past one stay on the last cell.
                return jnp.concatenate((moved[:lattice - 1], jnp.sum(moved[lattice - 1:], keepdims=True))), None

            partial, _ = jax.lax.scan(add, partial, on_lattice(edges[first - 1:last], weights[first - 1:last], cells))
        # A plain sum, not a matrix product: float32 matrix products on GPU
        # may run at reduced (TF32) precision, about 1e-4 relative.
        result = jnp.sum(partial[:, None] * onto_points, axis=0)
        return result / jnp.maximum(jnp.sum(result), jnp.finfo(dtype).tiny)

    batch = probs.shape[:-2]
    flat = jax.vmap(single)(probs.astype(dtype).reshape((-1, *probs.shape[-2:])), policy.astype(dtype).reshape((-1, policy.shape[-1])))
    return flat.reshape((*batch, probs.shape[-1]))


@lru_cache(maxsize=32)
def _tie_quadrature(num_actions: int) -> tuple[np.ndarray, np.ndarray]:
    # The tie-priority integrand is a polynomial of degree A-1. Gauss-Legendre
    # with ceil(A/2) nodes integrates it exactly (apart from floating point).
    nodes, weights = np.polynomial.legendre.leggauss((num_actions + 1) // 2)
    return ((nodes + 1) / 2).astype(np.float32), (weights / 2).astype(np.float32)


def best_policy(probs: jax.Array, invalid: jax.Array) -> jax.Array:
    """Exact discrete posterior-best probabilities, with uniform tie breaks.

    P(a wins) = sum_i h[a,i] integral_0^1 product_{b!=a}
    (F[b,i-1] + t h[b,i]) dt. The product over the other edges is the product
    over all edges divided by edge a's own factor, so work is O(A K) per
    quadrature node and live memory is O(A K) per lane.
    """
    probs = jnp.where(invalid[..., None], 0.0, probs.astype(jnp.float32))
    below = jnp.maximum(jnp.cumsum(probs, axis=-1) - probs, 0.0)
    nodes, weights = _tie_quadrature(probs.shape[-2])
    nodes = jnp.asarray(nodes)[:, None]
    # CDF of every legal edge at each tie point, [..., node, atom, action];
    # the action axis is last so that the product is a plain row reduction.
    cdf = jnp.swapaxes(below, -1, -2)[..., None, :, :] + nodes[..., None] * jnp.swapaxes(probs, -1, -2)[..., None, :, :]
    joint = jnp.prod(jnp.where(invalid[..., None, None, :], 1.0, cdf), axis=-1)
    # Edge a's own factor, [..., action, node, atom]. It can only be zero
    # where the edge has no mass there (h = 0, or t * h below float32
    # resolution); the joint product holds the same factor and is zero too,
    # so the guarded quotient contributes nothing, as it should.
    own = below[..., None, :] + nodes * probs[..., None, :]
    share = probs[..., None, :] / jnp.maximum(own, jnp.finfo(jnp.float32).tiny)
    # Sum the atoms of each node first, then the nodes, as two separate
    # reductions: short sums keep float32 rounding small on every backend.
    by_node = jax.lax.optimization_barrier(jnp.sum(joint[..., None, :, :] * share, axis=-1))
    policy = jnp.sum(by_node * jnp.asarray(weights), axis=-1)
    total = jnp.sum(policy, axis=-1, keepdims=True)
    fallback = (~invalid).astype(policy.dtype) / jnp.maximum(jnp.sum(~invalid, axis=-1, keepdims=True), 1)
    return jnp.where(total > 0.0, policy / jnp.maximum(total, jnp.finfo(policy.dtype).tiny), fallback)


def sample_action(key: jax.Array, probs: jax.Array, invalid: jax.Array) -> jax.Array:
    sample_key, tie_key = jax.random.split(key)
    u = jax.random.uniform(sample_key, probs.shape[:-1])
    cdf = jnp.cumsum(probs.astype(jnp.float32), axis=-1)
    index = jnp.minimum(jnp.sum(u[..., None] >= cdf, axis=-1), probs.shape[-1] - 1)
    best = jnp.max(jnp.where(invalid, -1, index), axis=-1, keepdims=True)
    tied = ~invalid & (index == best)
    return jax.random.categorical(tie_key, jnp.where(tied, 0.0, -jnp.inf), axis=-1).astype(jnp.int32)
