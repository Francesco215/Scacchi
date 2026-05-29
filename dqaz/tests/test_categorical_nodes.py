import numpy as np
import pytest

import dqaz

from test_python_api import mock_model, run_search


def test_categorical_game_search_and_export():
    config = dqaz.SearchConfig(
        action_size=1,
        observation_shape=(4,),
        simulations_per_root=3,
        posterior_best_samples=32,
        kappa_n=4.0,
        seed=19,
        debug=True,
        game="toy_categorical",
    )
    engine = dqaz.SearchEngine(config)
    tree_ids = engine.add_roots(np.array([[0, 0, 0]], dtype=np.uint8))

    run_search(engine, tree_ids)

    results = engine.finish(tree_ids, commit="posterior_argmax")
    assert results.actions.tolist() == [0]
    assert results.root_alpha.shape == (1, 1, 3)
    assert np.all(results.root_alpha > 0)

    targets = engine.export_targets(tree_ids)
    assert targets.observations.shape[0] >= 1
    assert targets.policy_target.shape == (targets.observations.shape[0], 1)
    assert np.all(targets.row_mask)


def test_categorical_root_can_be_advanced_by_observed_outcome():
    config = dqaz.SearchConfig(
        action_size=1,
        observation_shape=(4,),
        simulations_per_root=2,
        posterior_best_samples=16,
        seed=23,
        debug=True,
        game="toy_categorical",
    )
    engine = dqaz.SearchEngine(config)
    tree_ids = engine.add_roots(np.array([[1, 1, 0]], dtype=np.uint8))

    engine.advance_categorical_roots(tree_ids, np.array([1], dtype=np.uint32))
    assert engine.is_done(tree_ids)
    with pytest.raises(ValueError):
        engine.finish(tree_ids)


def test_empty_eval_batch_has_stable_shapes():
    config = dqaz.SearchConfig(
        action_size=1,
        observation_shape=(2, 2),
        simulations_per_root=1,
        posterior_best_samples=8,
        game="toy_categorical",
    )
    engine = dqaz.SearchEngine(config)
    batch = engine.request_evaluations(max_batch_size=4)
    assert batch.size == 0
    assert batch.token == 0
    assert batch.observations.shape == (0, 2, 2)
    assert batch.legal_masks.shape == (0, 1)

