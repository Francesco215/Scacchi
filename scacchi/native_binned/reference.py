"""float64 NumPy references written from the definitions, for checking only.

``best_policy64`` evaluates the probability that each move is the best under
independent categorical edge beliefs with uniform tie breaks::

    P(a best) = sum_i h[a,i] integral_0^1 prod_{b != a} (F[b,i-1] + t h[b,i]) dt

with the product taken directly (no division by a's own factor) and the
integral by Gauss-Legendre with A nodes, exact for the polynomial integrand.
``best_policy_enumerated`` is a brute-force check of that formula for tiny
shapes. ``weighted_sum64`` follows the specification of the law of
``sum_a w_a X_a``: each ``w_a X_a`` is split between the two neighbouring cells
of a 129-cell lattice on [0, 1], the laws are convolved in full (mass past the
last cell is kept on the last cell), then split onto the atoms and normalised.
"""

from __future__ import annotations

import itertools

import numpy as np

LATTICE = 129


def best_policy64(probs: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """``probs`` ``[A, K]``, ``invalid`` ``[A]`` -> ``[A]``."""
    probs = np.where(invalid[:, None], 0.0, probs.astype(np.float64))
    actions, atoms = probs.shape
    legal = ~invalid
    below = np.cumsum(probs, axis=-1) - probs
    nodes, weights = np.polynomial.legendre.leggauss(max(actions, 1))
    nodes, weights = (nodes + 1) / 2, weights / 2
    policy = np.zeros(actions)
    others = ~np.eye(actions, dtype=bool) & legal[None, :]  # others[a, b]: b counts against a
    for t, weight in zip(nodes, weights):
        factor = below + t * probs                                            # [b, i]
        excluded = np.prod(np.where(others[:, :, None], factor[None], 1.0), axis=1)  # [a, i]
        policy += weight * np.sum(probs * excluded, axis=-1)
    policy = np.where(legal, policy, 0.0)
    total = policy.sum()
    if total > 0.0:
        return policy / total
    return legal / max(int(legal.sum()), 1)


def best_policy_enumerated(probs: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Brute force over all outcomes (tiny shapes only), ties broken uniformly."""
    probs = np.where(invalid[:, None], 0.0, probs.astype(np.float64))
    actions, atoms = probs.shape
    legal = [a for a in range(actions) if not invalid[a]]
    policy = np.zeros(actions)
    for outcome in itertools.product(range(atoms), repeat=len(legal)):
        mass = np.prod([probs[a, i] for a, i in zip(legal, outcome)])
        top = max(outcome)
        winners = [a for a, i in zip(legal, outcome) if i == top]
        for a in winners:
            policy[a] += mass / len(winners)
    total = policy.sum()
    return policy / total if total > 0 else np.asarray(~invalid, dtype=np.float64) / max(len(legal), 1)


def onto_atoms64(lattice_law: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float64)
    fine = np.linspace(0.0, 1.0, len(lattice_law))
    upper = np.clip(np.searchsorted(points, fine, side="right"), 1, len(points) - 1)
    fraction = np.clip((fine - points[upper - 1]) / (points[upper] - points[upper - 1]), 0.0, 1.0)
    result = np.zeros(len(points))
    np.add.at(result, upper - 1, lattice_law * (1.0 - fraction))
    np.add.at(result, upper, lattice_law * fraction)
    return result / max(result.sum(), np.finfo(np.float64).tiny)


def weighted_sum64(probs: np.ndarray, weights: np.ndarray, points: np.ndarray, lattice: int = LATTICE) -> np.ndarray:
    """Law of ``sum_a weights[a] X_a`` on ``points``; ``probs`` ``[A, K]``, ``weights`` ``[A]``."""
    probs, weights, points = probs.astype(np.float64), weights.astype(np.float64), points.astype(np.float64)
    top = lattice - 1
    law = np.zeros(lattice)
    law[0] = 1.0
    for a in range(len(weights)):
        shift = weights[a] * points * top
        whole = np.floor(shift).astype(np.int64)
        part = shift - whole
        edge = np.zeros(lattice)
        np.add.at(edge, np.minimum(whole, top), probs[a] * (1.0 - part))
        np.add.at(edge, np.minimum(whole + 1, top), probs[a] * part)
        full = np.convolve(law, edge)
        law = full[:lattice].copy()
        law[top] = full[top:].sum()
    return onto_atoms64(law, points)


def backup64(edges: np.ndarray, invalid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """One lane: ``edges`` ``[A, K]`` -> ``[K]``."""
    return weighted_sum64(edges, best_policy64(edges, invalid), points)
