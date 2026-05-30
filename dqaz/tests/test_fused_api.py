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
    assert first.actions.tolist() == [2]
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
