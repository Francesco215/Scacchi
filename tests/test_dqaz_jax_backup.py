from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from scacchi.dqaz_jax_backup import (
    apply_batched_backup,
    apply_batched_backup_from_samples,
    posterior_best_policy_from_samples,
    should_use_jax_backup,
)
from scacchi.dirichlet_tree.native import NO_OUTCOME, OUTCOME_DRAW, OUTCOME_WIN


def _posterior_best_policy_from_samples_np(
    wdl_samples: np.ndarray,
    legal_mask: np.ndarray,
) -> np.ndarray:
    sample_count, num_nodes, max_actions, _ = wdl_samples.shape
    counts = np.zeros((num_nodes, max_actions), dtype=np.float32)
    utility = wdl_samples[..., 2] - wdl_samples[..., 0]
    for sample in range(sample_count):
        for node in range(num_nodes):
            legal = np.flatnonzero(legal_mask[node])
            if legal.size == 0:
                continue
            best = legal[int(np.argmax(utility[sample, node, legal]))]
            counts[node, best] += 1.0
    return counts / float(sample_count)


def _flip_wdl_np(alpha: np.ndarray) -> np.ndarray:
    return alpha[..., [2, 1, 0]]


def _align_wdl_np(
    alpha: np.ndarray,
    from_player: np.ndarray,
    to_player: np.ndarray,
) -> np.ndarray:
    return np.where((from_player == to_player)[..., None], alpha, _flip_wdl_np(alpha))


def _recompute_node_cache_np(
    edge_b: np.ndarray,
    edge_completed: np.ndarray,
    edge_r_count: np.ndarray,
    q_alpha: np.ndarray,
    value_alpha: np.ndarray,
    legal_mask: np.ndarray,
    wdl_samples: np.ndarray,
    kappa_n: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy = _posterior_best_policy_from_samples_np(wdl_samples, legal_mask)
    edge_posterior = np.where(edge_completed[..., None], edge_b, q_alpha)
    policy = np.where(legal_mask, policy, 0.0)
    evidence = np.sum(policy[..., None] * edge_posterior, axis=1)
    n_down = np.sum(np.where(legal_mask, edge_r_count, 0), axis=1).astype(np.int32)
    gamma = np.where(n_down > 0, n_down / (kappa_n + n_down), 0.0).astype(np.float32)
    c_v = (1.0 - gamma[:, None]) * value_alpha + gamma[:, None] * evidence
    return policy, c_v.astype(np.float32), n_down


def _apply_batched_backup_np(
    *,
    edge_b: np.ndarray,
    edge_completed: np.ndarray,
    edge_r_count: np.ndarray,
    q_alpha: np.ndarray,
    value_alpha: np.ndarray,
    legal_mask: np.ndarray,
    node_players: np.ndarray,
    path_nodes: np.ndarray,
    path_edges: np.ndarray,
    path_mask: np.ndarray,
    leaf_alpha: np.ndarray,
    leaf_players: np.ndarray,
    policy_samples_by_depth: np.ndarray,
    kappa_n: float,
) -> dict[str, np.ndarray]:
    edge_b = edge_b.copy()
    edge_completed = edge_completed.copy()
    edge_r_count = edge_r_count.copy()
    policy, c_v, n_down = _recompute_node_cache_np(
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        policy_samples_by_depth[0],
        kappa_n,
    )
    beta = leaf_alpha.copy()
    beta_players = leaf_players.copy()

    for depth in range(path_nodes.shape[1] - 1, -1, -1):
        for row in range(path_nodes.shape[0]):
            if not path_mask[row, depth]:
                continue
            parent = int(path_nodes[row, depth])
            edge = int(path_edges[row, depth])
            aligned = _align_wdl_np(
                beta[row],
                np.asarray(beta_players[row]),
                np.asarray(node_players[parent]),
            )
            edge_b[parent, edge] = aligned
            edge_completed[parent, edge] = True
            edge_r_count[parent, edge] += 1

        policy, c_v, n_down = _recompute_node_cache_np(
            edge_b,
            edge_completed,
            edge_r_count,
            q_alpha,
            value_alpha,
            legal_mask,
            policy_samples_by_depth[depth],
            kappa_n,
        )
        for row in range(path_nodes.shape[0]):
            if not path_mask[row, depth]:
                continue
            parent = int(path_nodes[row, depth])
            beta[row] = c_v[parent]
            beta_players[row] = node_players[parent]

    return {
        "edge_b": edge_b,
        "edge_completed": edge_completed,
        "edge_r_count": edge_r_count,
        "c_v": c_v,
        "n_down": n_down,
        "policy": policy,
    }


def _random_wdl_samples(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    raw = rng.gamma(shape=2.0, scale=1.0, size=shape).astype(np.float32)
    return raw / raw.sum(axis=-1, keepdims=True)


def _base_tree_arrays():
    num_nodes = 5
    max_actions = 3
    edge_b = np.zeros((num_nodes, max_actions, 3), dtype=np.float32)
    edge_completed = np.zeros((num_nodes, max_actions), dtype=bool)
    edge_r_count = np.zeros((num_nodes, max_actions), dtype=np.int32)
    q_alpha = np.asarray(
        [
            [[1.0, 1.0, 2.0], [2.0, 1.0, 1.0], [1.0, 3.0, 1.0]],
            [[1.0, 2.0, 3.0], [4.0, 2.0, 1.0], [2.0, 2.0, 2.0]],
            [[3.0, 1.0, 1.0], [1.0, 1.0, 3.0], [1.0, 4.0, 1.0]],
            [[2.0, 1.0, 2.0], [1.0, 3.0, 2.0], [3.0, 2.0, 1.0]],
            [[1.0, 2.0, 1.0], [2.0, 2.0, 3.0], [3.0, 1.0, 2.0]],
        ],
        dtype=np.float32,
    )
    value_alpha = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.asarray(
        [
            [True, True, False],
            [True, True, True],
            [True, False, True],
            [True, False, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    node_players = np.asarray([0, 1, 0, 1, 0], dtype=np.int32)
    return edge_b, edge_completed, edge_r_count, q_alpha, value_alpha, legal_mask, node_players


def _run_jax_backup(**kwargs):
    out = jax.jit(apply_batched_backup_from_samples)(
        jnp.asarray(kwargs["edge_b"]),
        jnp.asarray(kwargs["edge_completed"]),
        jnp.asarray(kwargs["edge_r_count"]),
        jnp.asarray(kwargs["q_alpha"]),
        jnp.asarray(kwargs["value_alpha"]),
        jnp.asarray(kwargs["legal_mask"]),
        jnp.asarray(kwargs["node_players"]),
        jnp.asarray(kwargs["path_nodes"]),
        jnp.asarray(kwargs["path_edges"]),
        jnp.asarray(kwargs["path_mask"]),
        jnp.asarray(kwargs["leaf_alpha"]),
        jnp.asarray(kwargs["leaf_players"]),
        jnp.asarray(kwargs["policy_samples_by_depth"]),
        kwargs["kappa_n"],
    )
    return {
        "edge_b": np.asarray(out.edge_b),
        "edge_completed": np.asarray(out.edge_completed),
        "edge_r_count": np.asarray(out.edge_r_count),
        "c_v": np.asarray(out.c_v),
        "n_down": np.asarray(out.n_down),
        "policy": np.asarray(out.policy),
    }


def _assert_backup_matches_reference(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]):
    np.testing.assert_array_equal(actual["edge_completed"], expected["edge_completed"])
    np.testing.assert_array_equal(actual["edge_r_count"], expected["edge_r_count"])
    np.testing.assert_array_equal(actual["n_down"], expected["n_down"])
    np.testing.assert_allclose(actual["edge_b"], expected["edge_b"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual["c_v"], expected["c_v"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual["policy"], expected["policy"], rtol=1e-6, atol=1e-6)


def test_posterior_best_policy_from_samples_matches_numpy_reference():
    samples = np.asarray(
        [
            [
                [[0.1, 0.1, 0.8], [0.7, 0.1, 0.2], [0.2, 0.7, 0.1]],
                [[0.3, 0.2, 0.5], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1]],
            ],
            [
                [[0.6, 0.2, 0.2], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]],
                [[0.4, 0.2, 0.4], [0.2, 0.4, 0.4], [0.1, 0.1, 0.8]],
            ],
            [
                [[0.2, 0.1, 0.7], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]],
                [[0.5, 0.1, 0.4], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6]],
            ],
        ],
        dtype=np.float32,
    )
    legal = np.asarray([[True, True, False], [True, False, True]], dtype=bool)

    actual = np.asarray(posterior_best_policy_from_samples(jnp.asarray(samples), jnp.asarray(legal)))
    expected = _posterior_best_policy_from_samples_np(samples, legal)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=-1), np.ones((2,), dtype=np.float32))
    assert actual[0, 2] == 0.0
    assert actual[1, 1] == 0.0


def test_jax_backup_guard_rejects_any_categorical_outcome():
    node_cat_outcome = np.full((4,), int(NO_OUTCOME), dtype=np.int8)
    edge_cat_outcome = np.full((4, 3), int(NO_OUTCOME), dtype=np.int8)

    assert should_use_jax_backup(
        node_cat_outcome=node_cat_outcome,
        edge_cat_outcome=edge_cat_outcome,
    )

    edge_cat_outcome[2, 1] = int(OUTCOME_WIN)
    assert not should_use_jax_backup(
        node_cat_outcome=node_cat_outcome,
        edge_cat_outcome=edge_cat_outcome,
    )

    assert not should_use_jax_backup(
        leaf_cat_outcome=jnp.asarray([int(NO_OUTCOME), int(OUTCOME_DRAW)], dtype=jnp.int8),
    )


def test_reverse_depth_jax_backup_matches_numpy_with_shared_parent_and_padded_paths():
    rng = np.random.default_rng(7)
    (
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        node_players,
    ) = _base_tree_arrays()
    path_nodes = np.asarray(
        [
            [0, 1, 3],
            [0, 0, 0],
            [2, 4, 0],
        ],
        dtype=np.int32,
    )
    path_edges = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 0],
            [0, 2, 0],
        ],
        dtype=np.int32,
    )
    path_mask = np.asarray(
        [
            [True, True, True],
            [True, False, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    leaf_alpha = np.asarray(
        [
            [1.0, 1.0, 4.0],
            [5.0, 1.0, 1.0],
            [2.0, 3.0, 1.0],
        ],
        dtype=np.float32,
    )
    leaf_players = np.asarray([0, 1, 1], dtype=np.int32)
    policy_samples_by_depth = _random_wdl_samples(
        rng,
        (path_nodes.shape[1], 5, edge_b.shape[0], edge_b.shape[1], 3),
    )
    kwargs = dict(
        edge_b=edge_b,
        edge_completed=edge_completed,
        edge_r_count=edge_r_count,
        q_alpha=q_alpha,
        value_alpha=value_alpha,
        legal_mask=legal_mask,
        node_players=node_players,
        path_nodes=path_nodes,
        path_edges=path_edges,
        path_mask=path_mask,
        leaf_alpha=leaf_alpha,
        leaf_players=leaf_players,
        policy_samples_by_depth=policy_samples_by_depth,
        kappa_n=4.0,
    )

    actual = _run_jax_backup(**kwargs)
    expected = _apply_batched_backup_np(**kwargs)

    _assert_backup_matches_reference(actual, expected)
    np.testing.assert_array_equal(actual["edge_completed"][0, :2], np.array([True, True]))
    assert actual["edge_r_count"][0, 0] == 1
    assert actual["edge_r_count"][0, 1] == 1


def test_snapshot_replacement_across_two_submits_does_not_accumulate_edge_alpha():
    rng = np.random.default_rng(11)
    (
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        node_players,
    ) = _base_tree_arrays()
    path_nodes = np.asarray([[0]], dtype=np.int32)
    path_edges = np.asarray([[0]], dtype=np.int32)
    path_mask = np.asarray([[True]], dtype=bool)
    leaf_players = np.asarray([0], dtype=np.int32)
    first_samples = _random_wdl_samples(rng, (1, 4, edge_b.shape[0], edge_b.shape[1], 3))
    second_samples = _random_wdl_samples(rng, (1, 4, edge_b.shape[0], edge_b.shape[1], 3))
    first_kwargs = dict(
        edge_b=edge_b,
        edge_completed=edge_completed,
        edge_r_count=edge_r_count,
        q_alpha=q_alpha,
        value_alpha=value_alpha,
        legal_mask=legal_mask,
        node_players=node_players,
        path_nodes=path_nodes,
        path_edges=path_edges,
        path_mask=path_mask,
        leaf_alpha=np.asarray([[1.0, 1.0, 3.0]], dtype=np.float32),
        leaf_players=leaf_players,
        policy_samples_by_depth=first_samples,
        kappa_n=4.0,
    )
    first = _run_jax_backup(**first_kwargs)
    second_kwargs = dict(
        first_kwargs,
        edge_b=first["edge_b"],
        edge_completed=first["edge_completed"],
        edge_r_count=first["edge_r_count"],
        leaf_alpha=np.asarray([[5.0, 1.0, 1.0]], dtype=np.float32),
        policy_samples_by_depth=second_samples,
    )

    actual = _run_jax_backup(**second_kwargs)
    expected = _apply_batched_backup_np(**second_kwargs)

    _assert_backup_matches_reference(actual, expected)
    np.testing.assert_allclose(actual["edge_b"][0, 0], np.array([5.0, 1.0, 1.0], dtype=np.float32))
    assert actual["edge_r_count"][0, 0] == 2
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        actual["edge_b"][0, 0],
        np.array([6.0, 2.0, 4.0], dtype=np.float32),
    )


def test_single_action_cache_update_matches_closed_form_value_blend():
    (
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        node_players,
    ) = _base_tree_arrays()
    legal_mask = np.zeros_like(legal_mask)
    legal_mask[:, 0] = True
    path_nodes = np.asarray([[0]], dtype=np.int32)
    path_edges = np.asarray([[0]], dtype=np.int32)
    path_mask = np.asarray([[True]], dtype=bool)
    leaf_alpha = np.asarray([[2.0, 1.0, 5.0]], dtype=np.float32)
    leaf_players = np.asarray([1], dtype=np.int32)
    samples = np.ones((1, 2, edge_b.shape[0], edge_b.shape[1], 3), dtype=np.float32) / 3.0

    actual = _run_jax_backup(
        edge_b=edge_b,
        edge_completed=edge_completed,
        edge_r_count=edge_r_count,
        q_alpha=q_alpha,
        value_alpha=value_alpha,
        legal_mask=legal_mask,
        node_players=node_players,
        path_nodes=path_nodes,
        path_edges=path_edges,
        path_mask=path_mask,
        leaf_alpha=leaf_alpha,
        leaf_players=leaf_players,
        policy_samples_by_depth=samples,
        kappa_n=4.0,
    )

    aligned = np.array([5.0, 1.0, 2.0], dtype=np.float32)
    gamma = np.float32(1.0 / 5.0)
    expected_c_v = (1.0 - gamma) * value_alpha[0] + gamma * aligned
    np.testing.assert_allclose(actual["edge_b"][0, 0], aligned)
    np.testing.assert_allclose(actual["c_v"][0], expected_c_v, rtol=1e-6, atol=1e-6)
    assert actual["n_down"][0] == 1


def test_gamma_sampled_backup_single_action_matches_closed_form_value_blend():
    (
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        node_players,
    ) = _base_tree_arrays()
    legal_mask = np.zeros_like(legal_mask)
    legal_mask[:, 0] = True
    out = jax.jit(apply_batched_backup, static_argnames=("sample_count",))(
        jax.random.PRNGKey(3),
        jnp.asarray(edge_b),
        jnp.asarray(edge_completed),
        jnp.asarray(edge_r_count),
        jnp.asarray(q_alpha),
        jnp.asarray(value_alpha),
        jnp.asarray(legal_mask),
        jnp.asarray(node_players),
        jnp.asarray([[0]], dtype=jnp.int32),
        jnp.asarray([[0]], dtype=jnp.int32),
        jnp.asarray([[True]], dtype=jnp.bool_),
        jnp.asarray([[2.0, 1.0, 5.0]], dtype=jnp.float32),
        jnp.asarray([1], dtype=jnp.int32),
        4.0,
        sample_count=8,
    )

    aligned = np.array([5.0, 1.0, 2.0], dtype=np.float32)
    gamma = np.float32(1.0 / 5.0)
    expected_c_v = (1.0 - gamma) * value_alpha[0] + gamma * aligned
    np.testing.assert_allclose(np.asarray(out.edge_b)[0, 0], aligned)
    np.testing.assert_allclose(np.asarray(out.c_v)[0], expected_c_v, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(out.policy)[0],
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert int(np.asarray(out.n_down)[0]) == 1
