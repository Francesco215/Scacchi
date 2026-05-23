from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .types import outcome_mean, outcome_utility


@jax.jit
def thompson_select_jax(
    rng_key: jax.Array,
    edge_post_alpha: jax.Array,
    legal_actions: jax.Array,
    action_mask: jax.Array,
) -> jax.Array:
    gamma = jax.random.gamma(rng_key, edge_post_alpha)
    phi = gamma / jnp.sum(gamma, axis=-1, keepdims=True)
    utility = phi[..., 2] - phi[..., 0]
    utility = jnp.where(action_mask, utility, -jnp.inf)
    idx = jnp.argmax(utility, axis=-1)
    return jnp.take_along_axis(legal_actions, idx[:, None], axis=1)[:, 0].astype(jnp.int32)


def thompson_select_np(
    rng: np.random.Generator,
    edge_post_alpha: np.ndarray,
    legal_actions: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.maximum(np.asarray(edge_post_alpha, dtype=np.float32), np.float32(1e-6))
    mask = np.asarray(action_mask, dtype=bool)
    gamma = rng.gamma(alpha, 1.0).astype(np.float32, copy=False)
    phi = gamma / np.maximum(np.sum(gamma, axis=-1, keepdims=True), np.float32(1e-12))
    utility = np.where(mask, phi[..., -1] - phi[..., 0], -np.inf)
    idx = np.argmax(utility, axis=-1).astype(np.int32)
    actions = np.take_along_axis(
        np.asarray(legal_actions, dtype=np.int32),
        idx[:, None],
        axis=1,
    )[:, 0]
    return actions.astype(np.int32, copy=False), idx


def posterior_best_policy_target_np(
    rng: np.random.Generator,
    alpha: np.ndarray,
    legal_action_mask: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    target = np.zeros((alpha.shape[0],), dtype=np.float32)
    legal_actions = np.flatnonzero(np.asarray(legal_action_mask, dtype=bool))
    if legal_actions.size == 0:
        return target
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    alpha = np.maximum(np.asarray(alpha, dtype=np.float32), np.float32(1e-6))
    hits = np.zeros((legal_actions.shape[0],), dtype=np.float32)
    for _ in range(int(num_samples)):
        sampled = np.stack([rng.dirichlet(alpha[action]) for action in legal_actions], axis=0)
        hits[int(np.argmax(sampled[:, 2] - sampled[:, 0]))] += 1.0
    target[legal_actions] = hits / float(num_samples)
    total = float(np.sum(target))
    if total <= 0.0:
        target[legal_actions] = 1.0 / float(legal_actions.shape[0])
    else:
        target /= total
    return target


def greedy_q_action(alpha: np.ndarray, legal_action_mask: np.ndarray) -> int:
    scores = outcome_utility(outcome_mean(alpha))
    return int(np.argmax(np.where(legal_action_mask, scores, -np.inf)))
