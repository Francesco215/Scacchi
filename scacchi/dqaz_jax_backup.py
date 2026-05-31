from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


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


def _flatten_path_batch(
    path_slots: jax.Array,
    path_edges: jax.Array,
    path_mask: jax.Array,
    leaf_alpha: jax.Array,
    leaf_players: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Accept either flat [P, D] or depth-major [B, D, T] trajectory batches."""

    if path_slots.ndim == 2:
        return path_slots, path_edges, path_mask, leaf_alpha, leaf_players
    if path_slots.ndim != 3:
        raise ValueError(f"path tensors must be rank 2 or 3, got rank {path_slots.ndim}")
    path_count = path_slots.shape[0] * path_slots.shape[2]
    return (
        jnp.swapaxes(path_slots, 1, 2).reshape((path_count, path_slots.shape[1])),
        jnp.swapaxes(path_edges, 1, 2).reshape((path_count, path_edges.shape[1])),
        jnp.swapaxes(path_mask, 1, 2).reshape((path_count, path_mask.shape[1])),
        leaf_alpha.reshape((path_count, leaf_alpha.shape[-1])),
        leaf_players.reshape((path_count,)),
    )


def _coalesced_edge_update(
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    parent_nodes: jax.Array,
    edge_indices: jax.Array,
    active: jax.Array,
    aligned_beta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Coalesce duplicate active `(node, edge)` hits before writing edge tables."""

    hit_count = jnp.zeros_like(edge_r_count).at[parent_nodes, edge_indices].add(
        active.astype(edge_r_count.dtype)
    )
    beta_sum = jnp.zeros_like(edge_b).at[parent_nodes, edge_indices].add(
        jnp.where(active[:, None], aligned_beta, 0.0)
    )
    hit_mask = hit_count > 0
    hit_count_f = hit_count.astype(edge_b.dtype)
    beta_out = beta_sum / jnp.maximum(hit_count_f[..., None], 1.0)
    edge_b = jnp.where(hit_mask[..., None], beta_out, edge_b)
    edge_completed = edge_completed | hit_mask
    edge_r_count = edge_r_count + hit_count
    return edge_b, edge_completed, edge_r_count


def _coalesced_depth_edge_update(
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    depth: jax.Array,
    slots: jax.Array,
    edge_indices: jax.Array,
    active: jax.Array,
    aligned_beta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Coalesce active `(root, depth, slot, edge)` hits for depth-bucketed tables."""

    root_count = edge_r_count.shape[0]
    root_ix = jnp.broadcast_to(jnp.arange(root_count)[:, None], slots.shape)
    depth_edge_count = edge_r_count[:, depth]
    depth_edge_b = edge_b[:, depth]
    depth_completed = edge_completed[:, depth]

    hit_count = jnp.zeros_like(depth_edge_count).at[root_ix, slots, edge_indices].add(
        active.astype(edge_r_count.dtype)
    )
    beta_sum = jnp.zeros_like(depth_edge_b).at[root_ix, slots, edge_indices].add(
        jnp.where(active[..., None], aligned_beta, 0.0)
    )
    hit_mask = hit_count > 0
    hit_count_f = hit_count.astype(edge_b.dtype)
    beta_out = beta_sum / jnp.maximum(hit_count_f[..., None], 1.0)

    depth_edge_b = jnp.where(hit_mask[..., None], beta_out, depth_edge_b)
    depth_completed = depth_completed | hit_mask
    depth_edge_count = depth_edge_count + hit_count
    edge_b = edge_b.at[:, depth].set(depth_edge_b)
    edge_completed = edge_completed.at[:, depth].set(depth_completed)
    edge_r_count = edge_r_count.at[:, depth].set(depth_edge_count)
    return edge_b, edge_completed, edge_r_count


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
    max_actions = wdl_samples.shape[-2]
    utility = wdl_samples[..., 2] - wdl_samples[..., 0]
    utility = jnp.where(legal_mask[None, ...], utility, -jnp.inf)
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
    evidence = jnp.sum(policy[..., None] * edge_posterior, axis=-2)
    n_down = jnp.sum(
        jnp.where(legal_mask, edge_r_count, 0),
        axis=-1,
        dtype=jnp.int32,
    )
    n_down_f = n_down.astype(value_alpha.dtype)
    gamma = jnp.where(n_down > 0, n_down_f / (jnp.asarray(kappa_n) + n_down_f), 0.0)
    c_v = (1.0 - gamma[..., None]) * value_alpha + gamma[..., None] * evidence
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

    path_nodes, path_edges, path_mask, leaf_alpha, leaf_players = _flatten_path_batch(
        path_nodes,
        path_edges,
        path_mask,
        leaf_alpha,
        leaf_players,
    )

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

        edge_b, edge_completed, edge_r_count = _coalesced_edge_update(
            edge_b,
            edge_completed,
            edge_r_count,
            parent_nodes,
            edge_indices,
            active,
            aligned_beta,
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

    if edge_b.ndim == 5:
        return _apply_depth_bucketed_backup_block(
            rng_key,
            edge_b,
            edge_completed,
            edge_r_count,
            q_alpha,
            value_alpha,
            legal_mask,
            node_players,
            path_nodes,
            path_edges,
            path_mask,
            c_v,
            n_down,
            policy,
            beta,
            beta_players,
            block_start,
            kappa_n,
            sample_count,
        )

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

            edge_b, edge_completed, edge_r_count = _coalesced_edge_update(
                edge_b,
                edge_completed,
                edge_r_count,
                parent_nodes,
                edge_indices,
                active,
                aligned_beta,
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


def _apply_depth_bucketed_backup_block(
    rng_key: jax.Array,
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    node_players: jax.Array,
    path_slots: jax.Array,
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
    block_depth = path_slots.shape[1]

    def body(depth_offset: jax.Array, carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        local_depth = block_depth - 1 - depth_offset
        depth = block_start + local_depth
        active = path_mask[:, local_depth, :]

        def active_body(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players = carry
            slots = jnp.where(active, path_slots[:, local_depth, :], 0)
            edge_indices = jnp.where(active, path_edges[:, local_depth, :], 0)
            depth_players = node_players[:, depth]
            parent_players = jnp.take_along_axis(depth_players, slots, axis=1)
            aligned_beta = align_wdl(beta, beta_players, parent_players)

            edge_b, edge_completed, edge_r_count = _coalesced_depth_edge_update(
                edge_b,
                edge_completed,
                edge_r_count,
                depth,
                slots,
                edge_indices,
                active,
                aligned_beta,
            )

            depth_key = jax.random.fold_in(rng_key, depth)
            depth_policy, depth_c_v, depth_n_down = recompute_node_cache_from_key(
                depth_key,
                edge_b[:, depth],
                edge_completed[:, depth],
                edge_r_count[:, depth],
                q_alpha[:, depth],
                value_alpha[:, depth],
                legal_mask[:, depth],
                kappa_n,
                sample_count,
            )
            policy = policy.at[:, depth].set(depth_policy)
            c_v = c_v.at[:, depth].set(depth_c_v)
            n_down = n_down.at[:, depth].set(depth_n_down)
            parent_c_v = jnp.take_along_axis(depth_c_v, slots[..., None], axis=1)
            beta = jnp.where(active[..., None], parent_c_v, beta)
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
        edge_b=edge_b,
        edge_completed=edge_completed,
        edge_r_count=edge_r_count,
        c_v=c_v,
        n_down=n_down,
        policy=policy,
        beta=beta,
        beta_players=beta_players,
    )


def _apply_depth_bucketed_backup_from_samples(
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    node_players: jax.Array,
    path_slots: jax.Array,
    path_edges: jax.Array,
    path_mask: jax.Array,
    leaf_alpha: jax.Array,
    leaf_players: jax.Array,
    policy_samples_by_depth: jax.Array,
    kappa_n: float | jax.Array,
) -> BackupArrays:
    max_depth = path_slots.shape[1]
    c_v = value_alpha
    n_down = jnp.sum(
        jnp.where(legal_mask, edge_r_count, 0),
        axis=-1,
        dtype=jnp.int32,
    )
    policy = jnp.zeros_like(legal_mask, dtype=value_alpha.dtype)

    def body(depth_offset: jax.Array, carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players = carry
        depth = max_depth - 1 - depth_offset
        active = path_mask[:, depth, :]

        def active_body(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players = carry
            slots = jnp.where(active, path_slots[:, depth, :], 0)
            edge_indices = jnp.where(active, path_edges[:, depth, :], 0)
            depth_players = node_players[:, depth]
            parent_players = jnp.take_along_axis(depth_players, slots, axis=1)
            aligned_beta = align_wdl(beta, beta_players, parent_players)

            edge_b, edge_completed, edge_r_count = _coalesced_depth_edge_update(
                edge_b,
                edge_completed,
                edge_r_count,
                depth,
                slots,
                edge_indices,
                active,
                aligned_beta,
            )

            depth_policy, depth_c_v, depth_n_down = recompute_node_cache_from_samples(
                edge_b[:, depth],
                edge_completed[:, depth],
                edge_r_count[:, depth],
                q_alpha[:, depth],
                value_alpha[:, depth],
                legal_mask[:, depth],
                policy_samples_by_depth[depth],
                kappa_n,
            )
            policy = policy.at[:, depth].set(depth_policy)
            c_v = c_v.at[:, depth].set(depth_c_v)
            n_down = n_down.at[:, depth].set(depth_n_down)
            parent_c_v = jnp.take_along_axis(depth_c_v, slots[..., None], axis=1)
            beta = jnp.where(active[..., None], parent_c_v, beta)
            beta_players = jnp.where(active, parent_players, beta_players)
            return edge_b, edge_completed, edge_r_count, c_v, n_down, policy, beta, beta_players

        return jax.lax.cond(jnp.any(active), active_body, lambda x: x, carry)

    edge_b, edge_completed, edge_r_count, c_v, n_down, policy, _, _ = jax.lax.fori_loop(
        0,
        max_depth,
        body,
        (
            edge_b,
            edge_completed,
            edge_r_count,
            c_v,
            n_down,
            policy,
            leaf_alpha,
            leaf_players,
        ),
    )
    return BackupArrays(
        edge_b=edge_b,
        edge_completed=edge_completed,
        edge_r_count=edge_r_count,
        c_v=c_v,
        n_down=n_down,
        policy=policy,
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

    Duplicate active `(node, edge)` updates at the same reverse depth are
    coalesced before edge tables are written, so counts and backed-up beta
    snapshots are deterministic.
    """

    if edge_b.ndim == 5:
        return _apply_depth_bucketed_backup_from_samples(
            edge_b,
            edge_completed,
            edge_r_count,
            q_alpha,
            value_alpha,
            legal_mask,
            node_players,
            path_nodes,
            path_edges,
            path_mask,
            leaf_alpha,
            leaf_players,
            policy_samples_by_depth,
            kappa_n,
        )

    path_nodes, path_edges, path_mask, leaf_alpha, leaf_players = _flatten_path_batch(
        path_nodes,
        path_edges,
        path_mask,
        leaf_alpha,
        leaf_players,
    )

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

        edge_b, edge_completed, edge_r_count = _coalesced_edge_update(
            edge_b,
            edge_completed,
            edge_r_count,
            parent_nodes,
            edge_indices,
            active,
            aligned_beta,
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
