"""Shared Thompson action selection for every search-tree node."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Int8, Int32, Key, UInt32

from . import base
from .categorical import NO_OUTCOME
from .tree import Tree, UnbatchedTree


# Marsaglia--Tsang accepts a proposal with very high probability for every
# shape >= 1. Four vectorized proposals reproduce exact-Dirichlet moments while
# keeping the sampler affordable inside every node repair. The rare rejected
# lane still has a finite, positive fallback below.
_GAMMA_PROPOSALS = 4


type _SamplePRNGKeys = Key[Array, "sample"] | UInt32[Array, "sample 2"]


def flip_outcome(outcome: Float[Array, "*batch outcome"]) -> Float[Array, "*batch outcome"]:
    return outcome[..., ::-1]


def align_outcome(outcome: Float[Array, "*batch outcome"], source_player: Int32[Array, "*batch"], target_player: Int32[Array, "*batch"]) -> Float[Array, "*batch outcome"]:
    return jnp.where((source_player == target_player)[..., None], outcome, flip_outcome(outcome))


def outcome_mean(alpha: Float[Array, "*batch outcome"]) -> Float[Array, "*batch outcome"]:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def outcome_utility(outcome: Float[Array, "*batch outcome"]) -> Float[Array, "*batch"]:
    return outcome[..., -1] - outcome[..., 0]


def categorical_utility(outcome: Int[Array, "*batch"], num_outcomes: int) -> Float[Array, "*batch"]:
    """Return exact scalar utility for a categorical outcome index."""

    outcome = jnp.asarray(outcome)
    dtype = jnp.result_type(outcome, jnp.float32)
    return jnp.where(outcome == int(num_outcomes) - 1, jnp.asarray(1.0, dtype=dtype), jnp.where(outcome == 0, jnp.asarray(-1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)))


def categorical_action(rng_key: base.PRNGKey, node_outcome: Int8[Array, "*batch"], edge_outcome: Int8[Array, "*batch action"], edge_distance: Int32[Array, "*batch action"], invalid_actions: Bool[Array, "*batch action"], *, num_outcomes: int) -> Int32[Array, "*batch"]:
    """Sample a certified action uniformly among equally good candidates.

    Win certificates prefer the shortest edge and loss certificates the
    longest. All certified draw actions are equivalent. Ties at each of those
    outcomes are sampled uniformly instead of consulting policy logits or
    action order.
    """

    legal = ~invalid_actions
    win_index = int(num_outcomes) - 1
    win_candidates = legal & (edge_outcome == win_index)
    loss_candidates = legal & (edge_outcome == 0)
    draw_candidates = legal & (edge_outcome == 1)

    distance = edge_distance.astype(jnp.float32)
    win_scores = jnp.where(win_candidates, -distance, -jnp.inf)
    loss_scores = jnp.where(loss_candidates, distance, -jnp.inf)
    draw_scores = jnp.where(draw_candidates, 0.0, -jnp.inf)

    is_win = node_outcome == win_index
    is_loss = node_outcome == 0
    scores = jnp.where(is_win[..., None], win_scores, jnp.where(is_loss[..., None], loss_scores, draw_scores))
    best = jnp.max(scores, axis=-1, keepdims=True)
    tied = jnp.isfinite(scores) & (scores == best)
    tie_logits = jnp.where(tied, 0.0, -jnp.inf)
    sampled = jax.random.categorical(rng_key, tie_logits, axis=-1)
    has_candidate = jnp.any(tied, axis=-1)
    return jnp.where(has_candidate, sampled, 0).astype(jnp.int32)


def masked_argmax(scores: Float[Array, "*batch action"], invalid_actions: Bool[Array, "*batch action"]) -> Int32[Array, "*batch"]:
    return jnp.argmax(jnp.where(invalid_actions, -jnp.inf, scores), axis=-1).astype(jnp.int32)


def sample_dirichlet(rng_key: base.PRNGKey, alpha: Float[Array, "*batch outcome"]) -> Float[Array, "*batch outcome"]:
    """Draw a bounded-work Marsaglia--Tsang Dirichlet sample.

    The north-star implementation in ``tictactoe-demo/app.js`` samples each
    gamma variate with Marsaglia--Tsang acceptance/rejection and applies the
    exact shape-augmentation identity below one. A single uncorrected
    Wilson--Hilferty proposal is measurably biased, especially when Thompson
    selection takes an extreme over many actions. Here we evaluate a small
    fixed population of proposals in parallel and take the first accepted one.
    Only the vanishingly rare all-rejected lane uses the old transform as a
    finite-work fallback. Normalizing in log space keeps concentrated terminal
    messages finite.

    The same primitive is used for traversal, node-local posterior-best
    populations, and the public root population; there is still only one
    Thompson action-selection rule.
    """

    dtype = jnp.result_type(alpha, jnp.float32)
    alpha = alpha.astype(dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    alpha = jnp.maximum(alpha, tiny)
    proposal_key, augment_key = jax.random.split(rng_key)
    normal_key, accept_key = jax.random.split(proposal_key)

    augmented_shape = jnp.where(alpha < 1.0, alpha + 1.0, alpha)
    d = augmented_shape - jnp.asarray(1.0 / 3.0, dtype=dtype)
    c = jax.lax.rsqrt(9.0 * d)
    proposal_shape = (_GAMMA_PROPOSALS, *alpha.shape)
    z = jax.random.normal(normal_key, proposal_shape, dtype=dtype)
    base = 1.0 + c * z
    positive = base > 0.0
    cube = jnp.where(positive, base**3, 1.0)
    accept_u = jax.random.uniform(accept_key, proposal_shape, dtype=dtype, minval=tiny, maxval=1.0)
    accepted = positive & (
        (accept_u < 1.0 - 0.0331 * z**4)
        | (
            jnp.log(accept_u)
            < 0.5 * z**2 + d * (1.0 - cube + jnp.log(cube))
        )
    )
    first_accepted = jnp.argmax(accepted, axis=0)
    chosen_cube = jnp.take_along_axis(cube, first_accepted[None, ...], axis=0)[0]
    fallback_base = jnp.maximum(1.0 + c * z[0], jnp.sqrt(tiny))
    chosen_cube = jnp.where(jnp.any(accepted, axis=0), chosen_cube, fallback_base**3)
    log_gamma = jnp.log(d) + jnp.log(chosen_cube)

    augment_u = jax.random.uniform(augment_key, alpha.shape, dtype=dtype, minval=tiny, maxval=1.0)
    log_gamma = log_gamma + jnp.where(alpha < 1.0, jnp.log(augment_u) / alpha, 0.0)

    return jax.nn.softmax(log_gamma, axis=-1)


def thompson_sample(rng_key: base.PRNGKey, alpha: Float[Array, "*batch action outcome"], invalid_actions: Bool[Array, "*batch action"], categorical_outcome: Int8[Array, "*batch action"] | None = None) -> Int32[Array, "*batch"]:
    """Apply the one action-selection rule used throughout this backend."""

    sampled = sample_dirichlet(rng_key, alpha)
    utility = outcome_utility(sampled)
    if categorical_outcome is not None:
        categorical_outcome = jnp.asarray(categorical_outcome)
        utility = jnp.where(categorical_outcome != int(NO_OUTCOME), categorical_utility(categorical_outcome, alpha.shape[-1]), utility)
    return masked_argmax(utility, invalid_actions)


def thompson_policy(rng_key: base.PRNGKey, alpha: Float[Array, "*batch action outcome"], invalid_actions: Bool[Array, "*batch action"], num_samples: int, *, chunk_size: int | None = None, categorical_outcome: Int8[Array, "*batch action"] | None = None) -> Float[Array, "*batch action"]:
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
        best = thompson_sample(rng_key, alpha, invalid_actions, categorical_outcome)
        policy = jax.nn.one_hot(best, num_actions, dtype=alpha.dtype)
        return jnp.where(invalid_actions, 0.0, policy)

    def sample_chunk(total_hits: Float[Array, "*batch action"], chunk: tuple[_SamplePRNGKeys, Bool[Array, "sample"]]) -> tuple[Float[Array, "*batch action"], None]:
        keys, valid_samples = chunk

        def sample_action(key: base.PRNGKey) -> Int32[Array, "*batch"]:
            return thompson_sample(key, alpha, invalid_actions, categorical_outcome)

        best = jax.vmap(sample_action)(keys)
        hits = jax.nn.one_hot(best, num_actions, dtype=alpha.dtype)
        weight = valid_samples.astype(alpha.dtype).reshape((chunk_size,) + (1,) * (hits.ndim - 1))
        return total_hits + jnp.sum(hits * weight, axis=0), None

    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    padded_count = num_chunks * chunk_size
    sample_key, padding_key = jax.random.split(rng_key)
    keys = jax.random.split(sample_key, num_samples)
    pad_count = padded_count - num_samples
    if pad_count:
        keys = jnp.concatenate([keys, jax.random.split(padding_key, pad_count)], axis=0)
    keys = keys.reshape((num_chunks, chunk_size) + keys.shape[1:])
    valid = (jnp.arange(padded_count) < num_samples).reshape((num_chunks, chunk_size))
    initial_hits = jnp.zeros(alpha.shape[:-1], dtype=alpha.dtype)
    total_hits, _ = jax.lax.scan(sample_chunk, initial_hits, (keys, valid))
    policy = jnp.where(~invalid_actions, total_hits / jnp.asarray(num_samples, dtype=alpha.dtype), 0.0)
    total = jnp.sum(policy, axis=-1, keepdims=True)
    legal_count = jnp.sum(~invalid_actions, axis=-1, keepdims=True)
    fallback = (~invalid_actions).astype(alpha.dtype) / jnp.maximum(legal_count, 1)
    return jnp.where(total > 0, policy / jnp.maximum(total, 1.0), fallback)


def effective_action_alpha(tree: UnbatchedTree, node_index: Int32[Array, ""]) -> Float[Array, "action outcome"]:
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
    target_player = jnp.broadcast_to(tree.node_to_play[node_index], child_player.shape)
    child_fallback = align_outcome(child_value, child_player, target_player)
    stored = tree.edge_alpha[node_index]
    edge_outcome = tree.edge_categorical_outcome[node_index]
    unresolved = edge_outcome == int(NO_OUTCOME)
    count = jnp.where(unresolved, tree.edge_payload[node_index], 0)
    fallback = jnp.where(visited[..., None], child_fallback, stored)
    return jnp.where(((~unresolved) | (count > 0))[..., None], stored, fallback)


def root_action_alpha(tree: Tree) -> Float[Array, "batch action outcome"]:
    root = Tree.ROOT_INDEX
    child_index = tree.children_index[:, root]
    visited = child_index != Tree.UNVISITED
    safe_child = jnp.where(visited, child_index, root)
    batch = jnp.arange(tree.parents.shape[0])[:, None]
    child_value = tree.node_value_priors[batch, safe_child]
    child_player = tree.node_to_play[batch, safe_child]
    target_player = jnp.broadcast_to(tree.node_to_play[:, root, None], child_player.shape)
    child_fallback = align_outcome(child_value, child_player, target_player)
    stored = tree.edge_alpha[:, root]
    edge_outcome = tree.edge_categorical_outcome[:, root]
    unresolved = edge_outcome == int(NO_OUTCOME)
    count = jnp.where(unresolved, tree.edge_payload[:, root], 0)
    fallback = jnp.where(visited[..., None], child_fallback, stored)
    return jnp.where(((~unresolved) | (count > 0))[..., None], stored, fallback)


def thompson_action_selection(rng_key: base.PRNGKey, tree: UnbatchedTree, node_index: Int32[Array, ""]) -> Int32[Array, ""]:
    """Take one Thompson draw for every legal action at ``node_index``."""

    categorical = tree.edge_categorical_outcome[node_index]
    return thompson_sample(rng_key, effective_action_alpha(tree, node_index), tree.invalid_actions[node_index] | (categorical != int(NO_OUTCOME)))
