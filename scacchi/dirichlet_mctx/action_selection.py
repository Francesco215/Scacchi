"""Shared Thompson action selection for every search-tree node."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp

from .tree import Tree


def flip_outcome(outcome: jax.Array) -> jax.Array:
    return outcome[..., ::-1]


def align_outcome(
    outcome: jax.Array,
    source_player: jax.Array,
    target_player: jax.Array,
) -> jax.Array:
    return jnp.where(
        (source_player == target_player)[..., None],
        outcome,
        flip_outcome(outcome),
    )


def outcome_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def outcome_utility(outcome: jax.Array) -> jax.Array:
    return outcome[..., -1] - outcome[..., 0]


def masked_argmax(scores: jax.Array, invalid_actions: jax.Array) -> jax.Array:
    return jnp.argmax(jnp.where(invalid_actions, -jnp.inf, scores), axis=-1).astype(
        jnp.int32
    )


def sample_dirichlet(
    rng_key: chex.PRNGKey,
    alpha: jax.Array,
) -> jax.Array:
    """Draw a fixed-work approximation to a Dirichlet sample.

    ``jax.random.dirichlet`` uses a rejection-based gamma sampler.  Small
    terminal components (for example ``0.01``) make that sampler prohibitively
    slow when it is nested inside every bottom-up tree repair.  This uses the
    Wilson--Hilferty gamma transform, with the exact shape-augmentation identity
    below one and the exact exponential draw at shape one.  Normalizing in log
    space keeps concentrated terminal messages finite.

    The same primitive is used for traversal, node-local posterior-best
    populations, and the public root population; there is still only one
    Thompson action-selection rule.
    """

    dtype = jnp.result_type(alpha, jnp.float32)
    alpha = alpha.astype(dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    alpha = jnp.maximum(alpha, tiny)
    normal_key, augment_key, exponential_key = jax.random.split(rng_key, 3)

    augmented_shape = jnp.where(alpha < 1.0, alpha + 1.0, alpha)
    d = augmented_shape - jnp.asarray(1.0 / 3.0, dtype=dtype)
    z = jax.random.normal(normal_key, alpha.shape, dtype=dtype)
    base = jnp.maximum(
        1.0 + z / jnp.sqrt(9.0 * d),
        jnp.sqrt(tiny),
    )
    log_gamma = jnp.log(d) + 3.0 * jnp.log(base)

    augment_u = jax.random.uniform(
        augment_key,
        alpha.shape,
        dtype=dtype,
        minval=tiny,
        maxval=1.0,
    )
    log_gamma = log_gamma + jnp.where(
        alpha < 1.0,
        jnp.log(augment_u) / alpha,
        0.0,
    )

    exponential_u = jax.random.uniform(
        exponential_key,
        alpha.shape,
        dtype=dtype,
        minval=tiny,
        maxval=1.0,
    )
    log_exponential = jnp.log(jnp.maximum(-jnp.log(exponential_u), tiny))
    log_gamma = jnp.where(alpha == 1.0, log_exponential, log_gamma)
    return jax.nn.softmax(log_gamma, axis=-1)


def thompson_sample(
    rng_key: chex.PRNGKey,
    alpha: jax.Array,
    invalid_actions: jax.Array,
) -> jax.Array:
    """Apply the one action-selection rule used throughout this backend."""

    sampled = sample_dirichlet(rng_key, alpha)
    return masked_argmax(outcome_utility(sampled), invalid_actions)


def thompson_policy(
    rng_key: chex.PRNGKey,
    alpha: jax.Array,
    invalid_actions: jax.Array,
    num_samples: int,
    *,
    chunk_size: int | None = None,
) -> jax.Array:
    """Estimate the posterior-best policy by repeating one Thompson rule.

    This is ``posteriorBestPolicy`` from the Tic-Tac-Toe demo.  Drawing keys
    up front makes the result independent of ``chunk_size`` so callers can
    trade peak memory for launch overhead without changing the algorithm.
    """

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if chunk_size is None:
        chunk_size = num_samples
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    chunk_size = min(chunk_size, num_samples)
    num_actions = alpha.shape[-2]

    if num_samples == 1:
        best = thompson_sample(rng_key, alpha, invalid_actions)
        policy = jax.nn.one_hot(best, num_actions, dtype=alpha.dtype)
        return jnp.where(invalid_actions, 0.0, policy)

    def sample_chunk(total_hits, chunk):
        keys, valid_samples = chunk
        best = jax.vmap(
            lambda key: thompson_sample(key, alpha, invalid_actions)
        )(keys)
        hits = jax.nn.one_hot(best, num_actions, dtype=alpha.dtype)
        weight = valid_samples.astype(alpha.dtype).reshape(
            (chunk_size,) + (1,) * (hits.ndim - 1)
        )
        return total_hits + jnp.sum(hits * weight, axis=0), None

    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    padded_count = num_chunks * chunk_size
    sample_key, padding_key = jax.random.split(rng_key)
    keys = jax.random.split(sample_key, num_samples)
    pad_count = padded_count - num_samples
    if pad_count:
        keys = jnp.concatenate(
            [keys, jax.random.split(padding_key, pad_count)],
            axis=0,
        )
    keys = keys.reshape((num_chunks, chunk_size) + keys.shape[1:])
    valid = (jnp.arange(padded_count) < num_samples).reshape(
        (num_chunks, chunk_size)
    )
    initial_hits = jnp.zeros(alpha.shape[:-1], dtype=alpha.dtype)
    total_hits, _ = jax.lax.scan(sample_chunk, initial_hits, (keys, valid))
    policy = jnp.where(
        ~invalid_actions,
        total_hits / jnp.asarray(num_samples, dtype=alpha.dtype),
        0.0,
    )
    total = jnp.sum(policy, axis=-1, keepdims=True)
    legal_count = jnp.sum(~invalid_actions, axis=-1, keepdims=True)
    fallback = (~invalid_actions).astype(alpha.dtype) / jnp.maximum(legal_count, 1)
    return jnp.where(total > 0, policy / jnp.maximum(total, 1.0), fallback)


def effective_action_alpha(tree: Tree, node_index: chex.Array) -> jax.Array:
    """Return all action Dirichlets for one unbatched node.

    This is ``edgePosterior`` from the Tic-Tac-Toe demo: use an edge message
    when present, otherwise an expanded child's V prior, otherwise the node's
    Q-head fallback stored in ``action_alpha``.
    """

    child_index = tree.children_index[node_index]
    visited = child_index != Tree.UNVISITED
    safe_child = jnp.where(visited, child_index, Tree.ROOT_INDEX)
    child_value = tree.node_value_priors[safe_child]
    child_player = tree.node_to_play[safe_child]
    child_fallback = align_outcome(
        child_value,
        child_player,
        tree.node_to_play[node_index],
    )
    posterior = tree.posterior
    stored = posterior.action_alpha[node_index]
    fallback = jnp.where(visited[..., None], child_fallback, stored)
    return jnp.where(
        (posterior.action_count[node_index] > 0)[..., None],
        stored,
        fallback,
    )


def root_action_alpha(tree: Tree) -> jax.Array:
    root_index = jnp.asarray(Tree.ROOT_INDEX, dtype=jnp.int32)
    return jax.vmap(effective_action_alpha, in_axes=(0, None))(tree, root_index)


def thompson_action_selection(
    rng_key: chex.PRNGKey,
    tree: Tree,
    node_index: chex.Array,
) -> jax.Array:
    """Take one Thompson draw for every legal action at ``node_index``."""

    return thompson_sample(
        rng_key,
        effective_action_alpha(tree, node_index),
        tree.invalid_actions[node_index],
    )
