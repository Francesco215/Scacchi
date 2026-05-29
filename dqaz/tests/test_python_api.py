import numpy as np
import pytest

import dqaz


def mock_model(observations, legal_masks, *, favor_first=False):
    batch = observations.shape[0]
    actions = legal_masks.shape[1]
    policy_logits = np.zeros((batch, actions), dtype=np.float32)
    value_alpha = np.ones((batch, 3), dtype=np.float32)
    q_alpha = np.ones((batch, actions, 3), dtype=np.float32)
    if favor_first:
        q_alpha[:, :, 0] = 4.0
        q_alpha[:, 0, :] = np.array([1.0, 1.0, 6.0], dtype=np.float32)
    return policy_logits, value_alpha, q_alpha


def run_search(engine, tree_ids, *, favor_first=False, max_steps=200):
    for _ in range(max_steps):
        if engine.is_done(tree_ids):
            return
        batch = engine.request_evaluations(max_batch_size=16)
        if batch.size:
            engine.submit_evaluations(
                batch.token,
                *mock_model(batch.observations, batch.legal_masks, favor_first=favor_first),
            )
    raise AssertionError("search did not finish")


def test_deterministic_mock_search_finish_and_export():
    config = dqaz.SearchConfig(
        action_size=3,
        observation_shape=(4,),
        simulations_per_root=5,
        posterior_best_samples=32,
        kappa_n=8.0,
        seed=7,
        debug=True,
        game="toy_deterministic",
    )
    engine = dqaz.SearchEngine(config)
    tree_ids = engine.add_roots(np.array([[0, 0, 0], [0, 1, 0]], dtype=np.uint8))

    run_search(engine, tree_ids)

    results = engine.finish(tree_ids, commit="posterior_argmax")
    assert results.actions.shape == (2,)
    assert results.pi_search.shape == (2, 3)
    np.testing.assert_allclose(results.pi_search.sum(axis=1), np.ones(2))
    assert results.root_alpha.shape == (2, 3, 3)
    assert np.all(results.root_alpha > 0)
    assert np.all(results.legal_masks)

    targets = engine.export_targets(tree_ids)
    assert targets.observations.shape[1:] == (4,)
    assert targets.legal_masks.shape[1:] == (3,)
    assert targets.policy_target.shape[1:] == (3,)
    assert targets.q_target_alpha.shape[1:] == (3, 3)
    assert targets.v_target_alpha.shape[1:] == (3,)
    assert np.all(targets.row_mask)
    assert np.all(targets.q_target_alpha > 0)


def test_stale_request_after_prune_is_ignored():
    config = dqaz.SearchConfig(
        action_size=1,
        observation_shape=(4,),
        simulations_per_root=2,
        posterior_best_samples=16,
        seed=3,
        debug=True,
        game="toy_deterministic",
    )
    engine = dqaz.SearchEngine(config)
    tree_ids = engine.add_roots(np.array([[0, 0, 0]], dtype=np.uint8))

    batch = engine.request_evaluations(max_batch_size=8)
    assert batch.size == 1
    engine.advance_roots(tree_ids, np.array([0], dtype=np.int32))

    engine.submit_evaluations(
        batch.token,
        *mock_model(batch.observations, batch.legal_masks),
    )

    run_search(engine, tree_ids)
    results = engine.finish(tree_ids, commit="posterior_argmax")
    assert results.actions.tolist() == [0]


def test_reused_subtree_can_continue_after_advance():
    config = dqaz.SearchConfig(
        action_size=1,
        observation_shape=(4,),
        simulations_per_root=3,
        posterior_best_samples=16,
        seed=11,
        debug=True,
        game="toy_deterministic",
    )
    engine = dqaz.SearchEngine(config)
    tree_ids = engine.add_roots(np.array([[0, 0, 0]], dtype=np.uint8))

    run_search(engine, tree_ids, favor_first=True)
    first = engine.finish(tree_ids, commit="posterior_argmax")
    assert first.actions.tolist() == [0]

    engine.advance_roots(tree_ids, first.actions)
    run_search(engine, tree_ids, favor_first=True)
    second = engine.finish(tree_ids, commit="posterior_argmax")
    assert second.actions.tolist() == [0]


def test_errors_are_python_exceptions():
    with pytest.raises(ValueError):
        dqaz.SearchConfig(action_size=0, observation_shape=(4,))

