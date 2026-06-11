import numpy as np
import pytest

import dqaz


def _config(simulations_per_root=1):
    return dqaz.SearchConfig(
        action_size=8,
        observation_shape=(1,),
        simulations_per_root=simulations_per_root,
        posterior_best_samples=8,
        kappa_n=4.0,
        seed=3,
        debug=True,
    )


def test_pending_limit_controls_multi_request_batching_without_solve_flag():
    engine = dqaz.SearchEngine(
        dqaz.SearchConfig(
            action_size=8,
            observation_shape=(1,),
            simulations_per_root=2,
            posterior_best_samples=8,
            kappa_n=4.0,
            seed=3,
            debug=True,
            max_pending_requests_per_root=2,
        )
    )
    tree_ids = engine.add_roots(
        [10],
        np.array([[1.0]], dtype=np.float32),
        np.array([0, 2], dtype=np.int64),
        np.array([2, 6], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    )

    batch = engine.request_transitions(max_batch_size=2)

    assert tree_ids.tolist() == [1]
    assert batch.size == 2
    assert sorted(batch.actions.tolist()) == [2, 6]
    assert batch.active_mask.tolist() == [True, True]


def test_jax_prepare_exports_depth_bucketed_shared_prefix_slots():
    engine = dqaz.SearchEngine(
        dqaz.SearchConfig(
            action_size=8,
            observation_shape=(1,),
            simulations_per_root=2,
            posterior_best_samples=8,
            kappa_n=4.0,
            seed=3,
            debug=True,
            max_pending_requests_per_root=2,
        )
    )
    engine.add_roots(
        [10],
        np.array([[1.0]], dtype=np.float32),
        np.array([0, 2], dtype=np.int64),
        np.array([2, 6], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    )
    batch = engine.request_transitions(max_batch_size=2)
    prepared = engine.submit_transitions_jax_prepare(
        batch.token,
        [100, 101],
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((3,), dtype=np.int64),
        np.zeros((0,), dtype=np.int32),
        np.ones((2,), dtype=np.int32),
        np.ones((2,), dtype=bool),
        np.array([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0]], dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
    )

    assert prepared.used_jax
    assert prepared.path_count == 2
    assert prepared.node_count == 1
    assert prepared.path_slots.shape == (1, 8, 2)
    assert prepared.edge_b.shape == (1, 8, 2, 8, 3)
    np.testing.assert_array_equal(np.asarray(prepared.path_slots)[0, 0], np.array([0, 0]))
    np.testing.assert_array_equal(np.asarray(prepared.node_ids)[0, 0], np.array([0, -1]))
    np.testing.assert_array_equal(np.sort(np.asarray(prepared.path_edges)[0, 0]), np.array([0, 1]))
    assert np.asarray(prepared.path_mask)[0, 0].all()


def test_jax_prepare_padding_does_not_inflate_backup_width():
    engine = dqaz.SearchEngine(
        dqaz.SearchConfig(
            action_size=8,
            observation_shape=(1,),
            simulations_per_root=2,
            posterior_best_samples=8,
            kappa_n=4.0,
            seed=3,
            debug=True,
            max_pending_requests_per_root=2,
        )
    )
    engine.add_roots(
        [10],
        np.array([[1.0]], dtype=np.float32),
        np.array([0, 2], dtype=np.int64),
        np.array([2, 6], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    )
    batch = engine.request_transitions(max_batch_size=2, pad_to=8)
    padded = batch.padded_size

    prepared = engine.submit_transitions_jax_prepare(
        batch.token,
        [100, 101],
        np.zeros((padded, 1), dtype=np.float32),
        np.zeros((padded + 1,), dtype=np.int64),
        np.zeros((0,), dtype=np.int32),
        np.ones((padded,), dtype=np.int32),
        np.ones((padded,), dtype=bool),
        np.ones((padded, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.ones((padded, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
    )

    assert batch.size == 2
    assert batch.padded_size == 8
    assert prepared.path_count == 2
    assert prepared.path_slots.shape == (1, 8, 2)
    assert prepared.edge_b.shape == (1, 8, 2, 8, 3)


def _add_two_sparse_roots(engine):
    return engine.add_roots(
        [10, 20],
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([2, 6], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    )


def _submit_terminal_children(engine, batch):
    padded = batch.padded_size
    terminated = np.zeros((padded,), dtype=bool)
    terminated[: batch.size] = True
    terminal_alpha = np.zeros((padded, 3), dtype=np.float32)
    terminal_alpha[: batch.size] = np.array([1.0, 1.0, 3.0], dtype=np.float32)
    engine.submit_transitions(
        batch.token,
        [100 + int(action) for action in batch.actions.tolist()],
        np.zeros((padded, 1), dtype=np.float32),
        np.zeros((padded + 1,), dtype=np.int64),
        np.zeros((0,), dtype=np.int32),
        np.ones((padded,), dtype=np.int32),
        terminated,
        terminal_alpha,
        np.zeros((0,), dtype=np.float32),
        np.ones((padded, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
    )


def test_sparse_fused_transition_finish_and_export():
    engine = dqaz.SearchEngine(_config())
    tree_ids = _add_two_sparse_roots(engine)

    batch = engine.request_transitions(max_batch_size=4, pad_to=4)

    assert batch.size == 2
    assert batch.padded_size == 4
    assert batch.actions.tolist()[:2] == [2, 6]
    assert batch.active_mask.tolist() == [True, True, False, False]
    assert list(batch.parent_states)[:2] == [10, 20]

    _submit_terminal_children(engine, batch)

    assert engine.is_done(tree_ids)
    results = engine.finish(tree_ids, commit="posterior_argmax")
    assert results.actions.tolist() == [2, 6]
    assert results.action_offsets.tolist() == [0, 1, 2]
    assert results.legal_actions.tolist() == [2, 6]
    assert results.pi_search.shape == (2,)
    assert results.root_alpha.shape == (2, 3)
    np.testing.assert_allclose(
        results.root_alpha,
        np.array([[3.0, 1.0, 1.0], [3.0, 1.0, 1.0]], dtype=np.float32),
    )

    targets = engine.export_targets(tree_ids)
    assert targets.observations.shape == (2, 1)
    assert targets.action_offsets.tolist() == [0, 1, 2]
    assert targets.legal_actions.tolist() == [2, 6]
    assert targets.policy_target.shape == (2,)
    assert targets.q_target_alpha.shape == (2, 3)
    assert targets.q_loss_weight.shape == (2,)
    assert targets.v_target_alpha.shape == (2, 3)
    assert targets.row_mask.tolist() == [True, True]


def test_dense_q_rows_are_rejected_for_sparse_roots():
    engine = dqaz.SearchEngine(_config())

    with pytest.raises(ValueError, match="q_alpha length"):
        engine.add_roots(
            [10],
            np.array([[1.0]], dtype=np.float32),
            np.array([0, 1], dtype=np.int64),
            np.array([2], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0.0], dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
            np.ones((8, 3), dtype=np.float32),
        )


def test_advance_roots_requires_existing_child():
    engine = dqaz.SearchEngine(_config(simulations_per_root=2))
    tree_ids = _add_two_sparse_roots(engine)

    with pytest.raises(ValueError, match="without an existing child"):
        engine.advance_roots(tree_ids[:1], np.array([2], dtype=np.int32))

    batch = engine.request_transitions(max_batch_size=2)
    _submit_terminal_children(engine, batch)
    engine.advance_roots(tree_ids[:1], np.array([2], dtype=np.int32))
    assert engine.is_done(tree_ids[:1])


def test_nonterminal_child_can_continue_search_and_export():
    engine = dqaz.SearchEngine(_config(simulations_per_root=2))
    tree_ids = engine.add_roots(
        [10],
        np.array([[1.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([2], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )

    first = engine.request_transitions(max_batch_size=1)
    first_action = int(first.actions.tolist()[0])
    categorical_edge = [2, 6].index(first_action)
    engine.submit_transitions(
        first.token,
        [100],
        np.array([[5.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([3], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([False], dtype=bool),
        np.ones((1, 3), dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )

    second = engine.request_transitions(max_batch_size=1)
    assert second.actions.tolist() == [3]
    assert list(second.parent_states) == [100]
    _submit_terminal_children(engine, second)

    assert engine.is_done(tree_ids)
    targets = engine.export_targets(tree_ids)
    assert targets.observations.tolist() == [[1.0], [5.0]]
    assert targets.legal_actions.tolist() == [2, 3]
    assert targets.action_offsets.tolist() == [0, 1, 2]


def test_jax_prepare_and_apply_nonterminal_backup():
    engine = dqaz.SearchEngine(_config(simulations_per_root=1))
    tree_ids = engine.add_roots(
        [10],
        np.array([[1.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([2], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )

    batch = engine.request_transitions(max_batch_size=1)
    prepared = engine.submit_transitions_jax_prepare(
        batch.token,
        [100],
        np.array([[5.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([3], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([False], dtype=bool),
        np.ones((1, 3), dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([[1.0, 1.0, 3.0]], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )

    assert prepared.used_jax
    assert prepared.node_count == 1
    assert prepared.path_count == 1
    assert prepared.edge_b.shape == (1, 8, 1, 8, 3)
    assert prepared.path_slots.shape == (1, 8, 1)
    assert prepared.path_slots[0, 0, 0] == 0
    assert prepared.path_edges[0, 0, 0] == 0
    assert prepared.path_mask[0, 0, 0]
    assert not np.asarray(prepared.path_mask)[0, 1:, :].any()
    edge_b = np.asarray(prepared.edge_b, dtype=np.float32)
    edge_completed = np.asarray(prepared.edge_completed, dtype=bool)
    edge_r_count = np.asarray(prepared.edge_r_count, dtype=np.int32)
    c_v = np.asarray(prepared.value_alpha, dtype=np.float32)
    n_down = np.zeros(edge_b.shape[:-2], dtype=np.int32)
    policy = np.zeros(edge_b.shape[:-1], dtype=np.float32)

    edge_b[0, 0, 0, 0] = np.array([3.0, 1.0, 1.0], dtype=np.float32)
    edge_completed[0, 0, 0, 0] = True
    edge_r_count[0, 0, 0, 0] = 1
    policy[0, 0, 0, 0] = 1.0
    gamma = np.float32(1.0 / 5.0)
    c_v[0, 0, 0] = (1.0 - gamma) * np.ones((3,), dtype=np.float32) + gamma * edge_b[0, 0, 0, 0]
    n_down[0, 0, 0] = 1

    engine.apply_jax_backup(
        prepared.tree_ids,
        prepared.node_ids,
        edge_b,
        edge_completed,
        edge_r_count,
        c_v,
        n_down,
        policy,
        prepared.node_count,
    )

    assert engine.is_done(tree_ids)
    results = engine.finish(tree_ids, commit="mean_utility_argmax")
    np.testing.assert_allclose(results.root_alpha, np.array([[3.0, 1.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(results.beta_v, c_v[0, :1, 0])
    np.testing.assert_allclose(results.pi_search, np.array([1.0], dtype=np.float32))


def test_jax_applied_policy_is_reused_by_finish():
    engine = dqaz.SearchEngine(_config(simulations_per_root=1))
    tree_ids = engine.add_roots(
        [10],
        np.array([[1.0]], dtype=np.float32),
        np.array([0, 2], dtype=np.int64),
        np.array([2, 6], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    )

    batch = engine.request_transitions(max_batch_size=1)
    prepared = engine.submit_transitions_jax_prepare(
        batch.token,
        [100],
        np.array([[5.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([3], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([False], dtype=bool),
        np.ones((1, 3), dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )

    edge_b = np.asarray(prepared.edge_b, dtype=np.float32)
    edge_completed = np.asarray(prepared.edge_completed, dtype=bool)
    edge_r_count = np.asarray(prepared.edge_r_count, dtype=np.int32)
    c_v = np.asarray(prepared.value_alpha, dtype=np.float32)
    n_down = np.zeros(edge_b.shape[:-2], dtype=np.int32)
    policy = np.zeros(edge_b.shape[:-1], dtype=np.float32)
    selected_edge = int(np.asarray(prepared.path_edges)[0, 0, 0])

    edge_b[0, 0, 0, selected_edge] = np.array([1.0, 1.0, 3.0], dtype=np.float32)
    edge_completed[0, 0, 0, selected_edge] = True
    edge_r_count[0, 0, 0, selected_edge] = 1
    c_v[0, 0, 0] = np.array([1.0, 1.0, 3.0], dtype=np.float32)
    n_down[0, 0, 0] = 1
    policy[0, 0, 0, 1] = 1.0

    engine.apply_jax_backup(
        prepared.tree_ids,
        prepared.node_ids,
        edge_b,
        edge_completed,
        edge_r_count,
        c_v,
        n_down,
        policy,
        prepared.node_count,
    )

    results = engine.finish(tree_ids, commit="posterior_argmax")
    assert results.actions.tolist() == [6]
    np.testing.assert_allclose(results.pi_search, np.array([0.0, 1.0], dtype=np.float32))


def test_jax_backup_preserves_existing_categorical_edge_metadata():
    engine = dqaz.SearchEngine(_config(simulations_per_root=2))
    tree_ids = engine.add_roots(
        [10],
        np.array([[1.0]], dtype=np.float32),
        np.array([0, 2], dtype=np.int64),
        np.array([2, 6], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    )

    first = engine.request_transitions(max_batch_size=1)
    first_action = int(first.actions.tolist()[0])
    categorical_edge = [2, 6].index(first_action)
    engine.submit_transitions(
        first.token,
        [100],
        np.array([[5.0]], dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        np.zeros((0,), dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([True], dtype=bool),
        np.array([[1.0, 1.0, 3.0]], dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
    )

    second = engine.request_transitions(max_batch_size=1)
    assert second.actions.tolist() == [action for action in [2, 6] if action != first_action]
    prepared = engine.submit_transitions_jax_prepare(
        second.token,
        [101],
        np.array([[7.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([4], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([False], dtype=bool),
        np.ones((1, 3), dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([[1.0, 1.0, 3.0]], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )

    edge_b = np.asarray(prepared.edge_b, dtype=np.float32)
    edge_completed = np.asarray(prepared.edge_completed, dtype=bool)
    edge_r_count = np.asarray(prepared.edge_r_count, dtype=np.int32)
    c_v = np.asarray(prepared.value_alpha, dtype=np.float32)
    n_down = np.asarray(prepared.edge_r_count, dtype=np.int32).sum(axis=-1)
    policy = np.zeros(edge_b.shape[:-1], dtype=np.float32)
    selected_edge = int(np.asarray(prepared.path_edges)[0, 0, 0])

    edge_b[0, 0, 0, selected_edge] = np.array([1.0, 1.0, 3.0], dtype=np.float32)
    edge_completed[0, 0, 0, selected_edge] = True
    edge_r_count[0, 0, 0, selected_edge] = 1
    c_v[0, 0, 0] = np.array([1.0, 1.0, 3.0], dtype=np.float32)
    n_down[0, 0, 0] = 2
    policy[0, 0, 0, selected_edge] = 1.0

    engine.apply_jax_backup(
        prepared.tree_ids,
        prepared.node_ids,
        edge_b,
        edge_completed,
        edge_r_count,
        c_v,
        n_down,
        policy,
        prepared.node_count,
    )

    assert engine.is_done(tree_ids)
    results = engine.finish(tree_ids, commit="posterior_argmax")
    assert results.q_target_kind.tolist()[categorical_edge] == 2
    assert results.q_target_outcome.tolist()[categorical_edge] == 0


def test_jax_prepare_uses_jax_for_terminal_rows_then_marks_categorical():
    engine = dqaz.SearchEngine(_config())
    tree_ids = _add_two_sparse_roots(engine)
    batch = engine.request_transitions(max_batch_size=2)
    prepared = engine.submit_transitions_jax_prepare(
        batch.token,
        [100, 101],
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((3,), dtype=np.int64),
        np.zeros((0,), dtype=np.int32),
        np.ones((2,), dtype=np.int32),
        np.ones((2,), dtype=bool),
        np.array([[1.0, 1.0, 3.0], [1.0, 1.0, 3.0]], dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
    )

    assert prepared.used_jax
    assert prepared.node_count == 2
    assert prepared.edge_b.shape == (2, 8, 1, 8, 3)
    assert prepared.path_slots.shape == (2, 8, 1)
    edge_b = np.asarray(prepared.edge_b, dtype=np.float32)
    edge_completed = np.asarray(prepared.edge_completed, dtype=bool)
    edge_r_count = np.asarray(prepared.edge_r_count, dtype=np.int32)
    c_v = np.asarray(prepared.value_alpha, dtype=np.float32)
    n_down = np.zeros(edge_b.shape[:-2], dtype=np.int32)
    policy = np.zeros(edge_b.shape[:-1], dtype=np.float32)
    aligned_terminal = np.array([3.0, 1.0, 1.0], dtype=np.float32)
    gamma = np.float32(1.0 / 5.0)
    path_mask = np.asarray(prepared.path_mask)
    for root, depth, trajectory in np.argwhere(path_mask):
        slot = int(np.asarray(prepared.path_slots)[root, depth, trajectory])
        edge_index = int(np.asarray(prepared.path_edges)[root, depth, trajectory])
        edge_b[root, depth, slot, edge_index] = aligned_terminal
        edge_completed[root, depth, slot, edge_index] = True
        edge_r_count[root, depth, slot, edge_index] = 1
        c_v[root, depth, slot] = (
            (1.0 - gamma) * np.ones((3,), dtype=np.float32) + gamma * aligned_terminal
        )
        n_down[root, depth, slot] = 1
        policy[root, depth, slot, edge_index] = 1.0

    engine.apply_jax_backup(
        prepared.tree_ids,
        prepared.node_ids,
        edge_b,
        edge_completed,
        edge_r_count,
        c_v,
        n_down,
        policy,
        prepared.node_count,
    )

    assert engine.is_done(tree_ids)
    results = engine.finish(tree_ids, commit="posterior_argmax")
    assert results.actions.tolist() == [2, 6]
    np.testing.assert_allclose(
        results.root_alpha,
        np.array([[3.0, 1.0, 1.0], [3.0, 1.0, 1.0]], dtype=np.float32),
    )


def test_pending_request_submitted_after_prune_is_ignored():
    engine = dqaz.SearchEngine(_config(simulations_per_root=2))
    tree_ids = engine.add_roots(
        [10],
        np.array([[1.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([2], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )

    first = engine.request_transitions(max_batch_size=1)
    engine.submit_transitions(
        first.token,
        [100],
        np.array([[5.0]], dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array([3], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([False], dtype=bool),
        np.ones((1, 3), dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )

    stale = engine.request_transitions(max_batch_size=1)
    assert stale.actions.tolist() == [3]
    engine.advance_roots(tree_ids, np.array([2], dtype=np.int32))

    _submit_terminal_children(engine, stale)

    fresh = engine.request_transitions(max_batch_size=1)
    assert fresh.actions.tolist() == [3]
