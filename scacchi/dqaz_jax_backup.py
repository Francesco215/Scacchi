from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from scacchi.dirichlet_tree.native import NO_OUTCOME


class BackupArrays(NamedTuple):
    edge_b: jax.Array
    edge_completed: jax.Array
    edge_r_count: jax.Array
    c_v: jax.Array
    n_down: jax.Array
    policy: jax.Array


class BackupBlockArrays(NamedTuple):
    edge_b: jax.Array
    edge_completed: jax.Array
    edge_r_count: jax.Array
    c_v: jax.Array
    n_down: jax.Array
    policy: jax.Array
    beta: jax.Array
    beta_players: jax.Array


def has_categorical_outcome(*outcome_arrays: object | None) -> bool:
    """Return true when the batch must stay on Rust's categorical solver path."""

    for outcome_array in outcome_arrays:
        if outcome_array is None:
            continue
        outcomes = np.asarray(jax.device_get(outcome_array))
        if outcomes.size and bool(np.any(outcomes != int(NO_OUTCOME))):
            return True
    return False


def should_use_jax_backup(
    *,
    node_cat_outcome: object | None = None,
    edge_cat_outcome: object | None = None,
    leaf_cat_outcome: object | None = None,
) -> bool:
    """Host-side guard for the non-categorical GPU backup prototype."""

    return not has_categorical_outcome(
        node_cat_outcome,
        edge_cat_outcome,
        leaf_cat_outcome,
    )


def flip_wdl(alpha: jax.Array) -> jax.Array:
    return alpha[..., jnp.array([2, 1, 0])]


def align_wdl(alpha: jax.Array, from_player: jax.Array, to_player: jax.Array) -> jax.Array:
    same_player = from_player == to_player
    return jnp.where(same_player[..., None], alpha, flip_wdl(alpha))


def posterior_best_policy_from_samples(
    wdl_samples: jax.Array,
    legal_mask: jax.Array,
) -> jax.Array:
    """Estimate posterior-best policy from pre-sampled WDL points.

    Args:
        wdl_samples: [S, N, K, 3] WDL probability samples.
        legal_mask: [N, K] valid compact-edge mask.

    Returns:
        [N, K] probability that each legal edge had the best sampled utility.
    """

    sample_count = wdl_samples.shape[0]
    max_actions = wdl_samples.shape[2]
    utility = wdl_samples[..., 2] - wdl_samples[..., 0]
    utility = jnp.where(legal_mask[None, :, :], utility, -jnp.inf)
    best = jnp.argmax(utility, axis=-1)
    counts = jnp.sum(
        jax.nn.one_hot(best, max_actions, dtype=wdl_samples.dtype),
        axis=0,
    )
    counts = jnp.where(legal_mask, counts, 0.0)
    return counts / jnp.asarray(sample_count, dtype=wdl_samples.dtype)


def posterior_best_policy_from_alpha(
    rng_key: jax.Array,
    edge_alpha: jax.Array,
    legal_mask: jax.Array,
    sample_count: int,
) -> jax.Array:
    samples = jax.random.gamma(
        rng_key,
        edge_alpha[None, :, :, :],
        shape=(sample_count, *edge_alpha.shape),
        dtype=edge_alpha.dtype,
    )
    samples = samples / jnp.sum(samples, axis=-1, keepdims=True)
    return posterior_best_policy_from_samples(samples, legal_mask)


def recompute_node_cache_from_policy(
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    policy: jax.Array,
    kappa_n: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    edge_posterior = jnp.where(edge_completed[..., None], edge_b, q_alpha)
    policy = jnp.where(legal_mask, policy, 0.0)
    evidence = jnp.sum(policy[..., None] * edge_posterior, axis=1)
    n_down = jnp.sum(
        jnp.where(legal_mask, edge_r_count, 0),
        axis=1,
        dtype=jnp.int32,
    )
    n_down_f = n_down.astype(value_alpha.dtype)
    gamma = jnp.where(n_down > 0, n_down_f / (jnp.asarray(kappa_n) + n_down_f), 0.0)
    c_v = (1.0 - gamma[:, None]) * value_alpha + gamma[:, None] * evidence
    return c_v, n_down


def recompute_node_cache_from_key(
    rng_key: jax.Array,
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    kappa_n: float | jax.Array,
    sample_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    edge_posterior = jnp.where(edge_completed[..., None], edge_b, q_alpha)
    policy = posterior_best_policy_from_alpha(
        rng_key,
        edge_posterior,
        legal_mask,
        sample_count,
    )
    c_v, n_down = recompute_node_cache_from_policy(
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        policy,
        kappa_n,
    )
    return policy, c_v, n_down


def recompute_node_cache_from_samples(
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    wdl_samples: jax.Array,
    kappa_n: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    policy = posterior_best_policy_from_samples(wdl_samples, legal_mask)
    c_v, n_down = recompute_node_cache_from_policy(
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        policy,
        kappa_n,
    )
    return policy, c_v, n_down


def apply_batched_backup(
    rng_key: jax.Array,
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    node_players: jax.Array,
    path_nodes: jax.Array,
    path_edges: jax.Array,
    path_mask: jax.Array,
    leaf_alpha: jax.Array,
    leaf_players: jax.Array,
    kappa_n: float | jax.Array,
    sample_count: int,
) -> BackupArrays:
    """Apply the non-categorical backup and sample posterior-best policy on GPU."""

    num_nodes = edge_b.shape[0]
    dummy_edge_b = jnp.zeros_like(edge_b[:1])
    dummy_completed = jnp.zeros_like(edge_completed[:1])
    dummy_r_count = jnp.zeros_like(edge_r_count[:1])
    dummy_q_alpha = jnp.ones_like(q_alpha[:1])
    dummy_value_alpha = jnp.ones_like(value_alpha[:1])
    dummy_legal = jnp.zeros_like(legal_mask[:1])
    dummy_player = jnp.zeros_like(node_players[:1])

    edge_b = jnp.concatenate([edge_b, dummy_edge_b], axis=0)
    edge_completed = jnp.concatenate([edge_completed, dummy_completed], axis=0)
    edge_r_count = jnp.concatenate([edge_r_count, dummy_r_count], axis=0)
    q_alpha = jnp.concatenate([q_alpha, dummy_q_alpha], axis=0)
    value_alpha = jnp.concatenate([value_alpha, dummy_value_alpha], axis=0)
    legal_mask = jnp.concatenate([legal_mask, dummy_legal], axis=0)
    node_players = jnp.concatenate([node_players, dummy_player], axis=0)

    max_depth = path_nodes.shape[1]
    dummy_node = jnp.asarray(num_nodes, dtype=path_nodes.dtype)
    depth_keys = jax.random.split(rng_key, max_depth)
    initial_c_v = value_alpha
    initial_n_down = jnp.sum(
        jnp.where(legal_mask, edge_r_count, 0),
        axis=1,
        dtype=jnp.int32,
    )
    initial_policy = jnp.zeros_like(legal_mask, dtype=value_alpha.dtype)

    def body(depth_offset: jax.Array, carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players = carry
        depth = max_depth - 1 - depth_offset
        active = path_mask[:, depth]
        parent_nodes = jnp.where(active, path_nodes[:, depth], dummy_node)
        edge_indices = jnp.where(active, path_edges[:, depth], 0)
        parent_players = node_players[parent_nodes]
        aligned_beta = align_wdl(beta, beta_players, parent_players)

        old_b = edge_b[parent_nodes, edge_indices]
        old_completed = edge_completed[parent_nodes, edge_indices]
        edge_b = edge_b.at[parent_nodes, edge_indices].set(
            jnp.where(active[:, None], aligned_beta, old_b)
        )
        edge_completed = edge_completed.at[parent_nodes, edge_indices].set(
            jnp.where(active, True, old_completed)
        )
        edge_r_count = edge_r_count.at[parent_nodes, edge_indices].add(
            active.astype(edge_r_count.dtype)
        )

        policy, c_v, n_down = recompute_node_cache_from_key(
            depth_keys[depth],
            edge_b,
            edge_completed,
            edge_r_count,
            q_alpha,
            value_alpha,
            legal_mask,
            kappa_n,
            sample_count,
        )
        parent_c_v = c_v[parent_nodes]
        beta = jnp.where(active[:, None], parent_c_v, beta)
        beta_players = jnp.where(active, parent_players, beta_players)
        return edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players

    (
        edge_b,
        edge_completed,
        edge_r_count,
        c_v,
        n_down,
        policy,
        _,
        _,
    ) = jax.lax.fori_loop(
        0,
        max_depth,
        body,
        (
            edge_b,
            edge_completed,
            edge_r_count,
            initial_c_v,
            initial_n_down,
            initial_policy,
            leaf_alpha,
            leaf_players,
        ),
    )

    return BackupArrays(
        edge_b=edge_b[:num_nodes],
        edge_completed=edge_completed[:num_nodes],
        edge_r_count=edge_r_count[:num_nodes],
        c_v=c_v[:num_nodes],
        n_down=n_down[:num_nodes],
        policy=policy[:num_nodes],
    )


def apply_batched_backup_block(
    rng_key: jax.Array,
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    node_players: jax.Array,
    path_nodes: jax.Array,
    path_edges: jax.Array,
    path_mask: jax.Array,
    c_v: jax.Array,
    n_down: jax.Array,
    policy: jax.Array,
    beta: jax.Array,
    beta_players: jax.Array,
    block_start: jax.Array,
    kappa_n: float | jax.Array,
    sample_count: int,
) -> BackupBlockArrays:
    """Apply one fixed-depth reverse backup block.

    The caller processes blocks from deepest to shallowest. Keeping the block
    depth fixed lets the JIT reuse the same executable while `path_mask` carries
    the variable-depth part of the tree.
    """

    num_nodes = edge_b.shape[0]
    dummy_edge_b = jnp.zeros_like(edge_b[:1])
    dummy_completed = jnp.zeros_like(edge_completed[:1])
    dummy_r_count = jnp.zeros_like(edge_r_count[:1])
    dummy_q_alpha = jnp.ones_like(q_alpha[:1])
    dummy_value_alpha = jnp.ones_like(value_alpha[:1])
    dummy_legal = jnp.zeros_like(legal_mask[:1])
    dummy_player = jnp.zeros_like(node_players[:1])
    dummy_n_down = jnp.zeros_like(n_down[:1])
    dummy_policy = jnp.zeros_like(policy[:1])

    edge_b = jnp.concatenate([edge_b, dummy_edge_b], axis=0)
    edge_completed = jnp.concatenate([edge_completed, dummy_completed], axis=0)
    edge_r_count = jnp.concatenate([edge_r_count, dummy_r_count], axis=0)
    q_alpha = jnp.concatenate([q_alpha, dummy_q_alpha], axis=0)
    value_alpha = jnp.concatenate([value_alpha, dummy_value_alpha], axis=0)
    legal_mask = jnp.concatenate([legal_mask, dummy_legal], axis=0)
    node_players = jnp.concatenate([node_players, dummy_player], axis=0)
    c_v = jnp.concatenate([c_v, dummy_value_alpha], axis=0)
    n_down = jnp.concatenate([n_down, dummy_n_down], axis=0)
    policy = jnp.concatenate([policy, dummy_policy], axis=0)

    block_depth = path_nodes.shape[1]
    dummy_node = jnp.asarray(num_nodes, dtype=path_nodes.dtype)

    def body(depth_offset: jax.Array, carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        local_depth = block_depth - 1 - depth_offset
        active = path_mask[:, local_depth]

        def active_body(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players = carry
            parent_nodes = jnp.where(active, path_nodes[:, local_depth], dummy_node)
            edge_indices = jnp.where(active, path_edges[:, local_depth], 0)
            parent_players = node_players[parent_nodes]
            aligned_beta = align_wdl(beta, beta_players, parent_players)

            old_b = edge_b[parent_nodes, edge_indices]
            old_completed = edge_completed[parent_nodes, edge_indices]
            edge_b = edge_b.at[parent_nodes, edge_indices].set(
                jnp.where(active[:, None], aligned_beta, old_b)
            )
            edge_completed = edge_completed.at[parent_nodes, edge_indices].set(
                jnp.where(active, True, old_completed)
            )
            edge_r_count = edge_r_count.at[parent_nodes, edge_indices].add(
                active.astype(edge_r_count.dtype)
            )

            depth_key = jax.random.fold_in(rng_key, block_start + local_depth)
            policy, c_v, n_down = recompute_node_cache_from_key(
                depth_key,
                edge_b,
                edge_completed,
                edge_r_count,
                q_alpha,
                value_alpha,
                legal_mask,
                kappa_n,
                sample_count,
            )
            parent_c_v = c_v[parent_nodes]
            beta = jnp.where(active[:, None], parent_c_v, beta)
            beta_players = jnp.where(active, parent_players, beta_players)
            return edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players

        return jax.lax.cond(jnp.any(active), active_body, lambda x: x, carry)

    (
        edge_b,
        edge_completed,
        edge_r_count,
        c_v,
        n_down,
        policy,
        beta,
        beta_players,
    ) = jax.lax.fori_loop(
        0,
        block_depth,
        body,
        (
            edge_b,
            edge_completed,
            edge_r_count,
            c_v,
            n_down,
            policy,
            beta,
            beta_players,
        ),
    )

    return BackupBlockArrays(
        edge_b=edge_b[:num_nodes],
        edge_completed=edge_completed[:num_nodes],
        edge_r_count=edge_r_count[:num_nodes],
        c_v=c_v[:num_nodes],
        n_down=n_down[:num_nodes],
        policy=policy[:num_nodes],
        beta=beta,
        beta_players=beta_players,
    )


def apply_batched_backup_from_samples(
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    node_players: jax.Array,
    path_nodes: jax.Array,
    path_edges: jax.Array,
    path_mask: jax.Array,
    leaf_alpha: jax.Array,
    leaf_players: jax.Array,
    policy_samples_by_depth: jax.Array,
    kappa_n: float | jax.Array,
) -> BackupArrays:
    """Apply reverse path backups with JAX scatter/scan semantics.

    This is only the non-categorical fast path. Callers should use
    `should_use_jax_backup` and leave any tree with categorical outcomes on the
    existing Rust path.

    This prototype assumes no duplicate active `(node, edge)` updates at the
    same reverse depth. Duplicate edge updates require explicit conflict
    resolution before production use because `scatter.set` is not a stable
    last-write primitive.
    """

    num_nodes = edge_b.shape[0]
    dummy_edge_b = jnp.zeros_like(edge_b[:1])
    dummy_completed = jnp.zeros_like(edge_completed[:1])
    dummy_r_count = jnp.zeros_like(edge_r_count[:1])
    dummy_q_alpha = jnp.zeros_like(q_alpha[:1])
    dummy_value_alpha = jnp.zeros_like(value_alpha[:1])
    dummy_legal = jnp.zeros_like(legal_mask[:1])
    dummy_player = jnp.zeros_like(node_players[:1])

    edge_b = jnp.concatenate([edge_b, dummy_edge_b], axis=0)
    edge_completed = jnp.concatenate([edge_completed, dummy_completed], axis=0)
    edge_r_count = jnp.concatenate([edge_r_count, dummy_r_count], axis=0)
    q_alpha = jnp.concatenate([q_alpha, dummy_q_alpha], axis=0)
    value_alpha = jnp.concatenate([value_alpha, dummy_value_alpha], axis=0)
    legal_mask = jnp.concatenate([legal_mask, dummy_legal], axis=0)
    node_players = jnp.concatenate([node_players, dummy_player], axis=0)
    policy_samples_by_depth = jnp.concatenate(
        [policy_samples_by_depth, jnp.zeros_like(policy_samples_by_depth[:, :, :1])],
        axis=2,
    )

    initial_policy, initial_c_v, initial_n_down = recompute_node_cache_from_samples(
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        policy_samples_by_depth[0],
        kappa_n,
    )

    max_depth = path_nodes.shape[1]
    dummy_node = jnp.asarray(num_nodes, dtype=path_nodes.dtype)

    def body(depth_offset: jax.Array, carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players = carry
        depth = max_depth - 1 - depth_offset
        active = path_mask[:, depth]
        parent_nodes = jnp.where(active, path_nodes[:, depth], dummy_node)
        edge_indices = jnp.where(active, path_edges[:, depth], 0)
        parent_players = node_players[parent_nodes]
        aligned_beta = align_wdl(beta, beta_players, parent_players)

        old_b = edge_b[parent_nodes, edge_indices]
        old_completed = edge_completed[parent_nodes, edge_indices]
        edge_b = edge_b.at[parent_nodes, edge_indices].set(
            jnp.where(active[:, None], aligned_beta, old_b)
        )
        edge_completed = edge_completed.at[parent_nodes, edge_indices].set(
            jnp.where(active, True, old_completed)
        )
        edge_r_count = edge_r_count.at[parent_nodes, edge_indices].add(
            active.astype(edge_r_count.dtype)
        )

        policy, c_v, n_down = recompute_node_cache_from_samples(
            edge_b,
            edge_completed,
            edge_r_count,
            q_alpha,
            value_alpha,
            legal_mask,
            policy_samples_by_depth[depth],
            kappa_n,
        )
        parent_c_v = c_v[parent_nodes]
        beta = jnp.where(active[:, None], parent_c_v, beta)
        beta_players = jnp.where(active, parent_players, beta_players)
        return edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players

    (
        edge_b,
        edge_completed,
        edge_r_count,
        c_v,
        n_down,
        policy,
        _,
        _,
    ) = jax.lax.fori_loop(
        0,
        max_depth,
        body,
        (
            edge_b,
            edge_completed,
            edge_r_count,
            initial_c_v,
            initial_n_down,
            initial_policy,
            leaf_alpha,
            leaf_players,
        ),
    )

    return BackupArrays(
        edge_b=edge_b[:num_nodes],
        edge_completed=edge_completed[:num_nodes],
        edge_r_count=edge_r_count[:num_nodes],
        c_v=c_v[:num_nodes],
        n_down=n_down[:num_nodes],
        policy=policy[:num_nodes],
    )
