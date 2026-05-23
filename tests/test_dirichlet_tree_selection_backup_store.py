import numpy as np
import pytest
import jax
import jax.numpy as jnp

from scacchi.dirichlet_tree.backup import backup_path, terminal_one_hot, update_edge_base_from_child
from scacchi.dirichlet_tree.arena_search import BatchedPosteriorArenaSearch, PosteriorArena
from scacchi.dirichlet_tree.selection import thompson_select_jax, thompson_select_np
from scacchi.dirichlet_tree.store import InMemoryNodeStore, RedisNodeStore
from scacchi.dirichlet_tree.types import NodeBlob, PathStep, StateKey, outcome_mean


def test_thompson_selection_masks_invalid_actions_and_returns_batch_shape():
    alpha = jnp.array(
        [
            [[1.0, 1.0, 10.0], [100.0, 1.0, 1.0]],
            [[100.0, 1.0, 1.0], [1.0, 1.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    legal_actions = jnp.array([[4, 9], [3, 8]], dtype=jnp.int32)
    mask = jnp.array([[True, False], [False, True]])

    action = thompson_select_jax(jax.random.PRNGKey(0), alpha, legal_actions, mask)

    assert action.shape == (2,)
    assert action.tolist() == [4, 8]

    np_action, np_pos = thompson_select_np(
        np.random.default_rng(0),
        np.asarray(alpha),
        np.asarray(legal_actions),
        np.asarray(mask),
    )
    assert np_action.shape == (2,)
    assert np_pos.shape == (2,)
    assert np_action.tolist() == [4, 8]


def test_thompson_selection_prefers_higher_wdl_utility_statistically():
    alpha = jnp.tile(
        jnp.array([[[1.0, 1.0, 20.0], [20.0, 1.0, 1.0]]], dtype=jnp.float32),
        (512, 1, 1),
    )
    legal_actions = jnp.tile(jnp.array([[0, 1]], dtype=jnp.int32), (512, 1))
    mask = jnp.ones((512, 2), dtype=bool)

    actions = np.asarray(thompson_select_jax(jax.random.PRNGKey(1), alpha, legal_actions, mask))

    assert np.mean(actions == 0) > 0.95

    np_actions, _ = thompson_select_np(
        np.random.default_rng(1),
        np.asarray(alpha),
        np.asarray(legal_actions),
        np.asarray(mask),
    )

    assert np.mean(np_actions == 0) > 0.95


def test_backup_updates_direct_evidence_and_normalized_ancestor_summary():
    store = InMemoryNodeStore()
    root = NodeBlob.expanded_node(
        key=StateKey((1, 0, 0, 0)),
        current_player=0,
        legal_action_mask=np.array([True, False]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((2,), dtype=np.float32),
        q_alpha=np.ones((2, 3), dtype=np.float32),
    )
    child = NodeBlob.expanded_node(
        key=StateKey((2, 0, 0, 0)),
        current_player=1,
        legal_action_mask=np.array([False, True]),
        value_alpha=np.array([1.0, 1.0, 4.0], dtype=np.float32),
        policy_logits=np.zeros((2,), dtype=np.float32),
        q_alpha=np.ones((2, 3), dtype=np.float32),
    )
    leaf = NodeBlob.expanded_node(
        key=StateKey((3, 0, 0, 0)),
        current_player=0,
        legal_action_mask=np.array([True, False]),
        value_alpha=np.array([1.0, 1.0, 3.0], dtype=np.float32),
        policy_logits=np.zeros((2,), dtype=np.float32),
        q_alpha=np.ones((2, 3), dtype=np.float32),
    )
    root.child_keys[0] = child.key.to_array()
    child.child_keys[0] = leaf.key.to_array()
    store.put_many([root, child, leaf])
    update_edge_base_from_child(store, parent_key=root.key, action=0, child=child)

    backup_path(
        store,
        path=[PathStep(root.key, 0), PathStep(child.key, 1)],
        leaf_node=leaf,
        leaf_value=outcome_mean(leaf.value_alpha),
        leaf_weight=2.0,
        c_state=0.5,
    )

    assert np.allclose(child.edge_evidence_E[0], 2.0 * np.array([0.6, 0.2, 0.2]))
    child_summary = outcome_mean(child.state_summary_alpha)
    assert np.allclose(root.edge_evidence_E[0], 0.5 * child_summary[::-1])
    assert np.allclose(root.edge_post_alpha[0], root.edge_base_alpha[0] + root.edge_evidence_E[0])


def test_ancestor_backup_uses_child_state_summary_not_value_alpha():
    store = InMemoryNodeStore()
    root = NodeBlob.expanded_node(
        key=StateKey((11, 0, 0, 0)),
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    child = NodeBlob.expanded_node(
        key=StateKey((12, 0, 0, 0)),
        current_player=1,
        legal_action_mask=np.array([True]),
        value_alpha=np.array([100.0, 1.0, 1.0], dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    grandchild = NodeBlob.expanded_node(
        key=StateKey((13, 0, 0, 0)),
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    leaf = NodeBlob.expanded_node(
        key=StateKey((14, 0, 0, 0)),
        current_player=1,
        legal_action_mask=np.array([True]),
        value_alpha=np.array([1.0, 1.0, 5.0], dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    child.state_summary_alpha = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    root.child_keys[0] = child.key.to_array()
    child.child_keys[0] = grandchild.key.to_array()
    grandchild.child_keys[0] = leaf.key.to_array()
    store.put_many([root, child, grandchild, leaf])

    backup_path(
        store,
        path=[PathStep(root.key, 0), PathStep(child.key, 0), PathStep(grandchild.key, 0)],
        leaf_node=leaf,
        leaf_value=outcome_mean(leaf.value_alpha),
        leaf_weight=1.0,
        c_state=0.25,
    )

    expected_root_evidence = 0.25 * outcome_mean(child.state_summary_alpha)[::-1]
    assert np.allclose(root.edge_evidence_E[0], expected_root_evidence)
    assert not np.allclose(root.edge_evidence_E[0], 0.25 * outcome_mean(child.value_alpha)[::-1])


def test_arena_child_value_alpha_replaces_parent_edge_base_with_perspective_flip():
    arena = PosteriorArena(max_nodes=4, max_edges=4, num_actions=1, num_outcomes=3)
    root_id = arena.add_expanded_node(
        key=np.array([1, 0, 0, 0], dtype=np.uint32),
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    child_id = arena.add_expanded_node(
        key=np.array([2, 0, 0, 0], dtype=np.uint32),
        current_player=1,
        legal_action_mask=np.array([True]),
        value_alpha=np.array([1.0, 2.0, 7.0], dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    edge_id = int(arena.node_first_edge[root_id])
    search = BatchedPosteriorArenaSearch(env=object())
    search.arena = arena

    search._update_edge_base_from_children(np.array([edge_id], dtype=np.int32), np.array([child_id], dtype=np.int32))

    assert np.allclose(arena.edge_base_alpha[edge_id], np.array([7.0, 2.0, 1.0], dtype=np.float32))
    assert np.allclose(arena.edge_post_alpha[edge_id], arena.edge_base_alpha[edge_id] + arena.edge_E[edge_id])


def test_arena_terminal_backup_uses_terminal_node_perspective():
    arena = PosteriorArena(max_nodes=4, max_edges=4, num_actions=1, num_outcomes=3)
    root_id = arena.add_expanded_node(
        key=np.array([1, 0, 0, 0], dtype=np.uint32),
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    terminal_id = arena.add_terminal_node(
        key=np.array([2, 0, 0, 0], dtype=np.uint32),
        current_player=1,
        terminal_outcome=2,
    )
    edge_id = int(arena.node_first_edge[root_id])
    arena.edge_child_node[edge_id] = terminal_id
    search = BatchedPosteriorArenaSearch(env=object())
    search.arena = arena

    search._backup_path(
        np.array([root_id], dtype=np.int32),
        np.array([edge_id], dtype=np.int32),
        1,
        leaf_node_id=terminal_id,
        leaf_value=terminal_one_hot(2),
        leaf_weight=8.0,
        c_state=0.1,
    )

    assert np.allclose(arena.edge_E[edge_id], np.array([8.0, 0.0, 0.0], dtype=np.float32))


def test_inflight_and_duplicate_scheduling_do_not_change_posterior():
    store = InMemoryNodeStore()
    root = NodeBlob.expanded_node(
        key=StateKey((21, 0, 0, 0)),
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32).reshape((1, 3)),
    )
    store.put_many([root])
    before_base = root.edge_base_alpha.copy()
    before_evidence = root.edge_evidence_E.copy()
    before_post = root.edge_post_alpha.copy()

    child_key = StateKey((22, 0, 0, 0))
    claim = store.claim_many_inflight([child_key, child_key])

    assert claim.claimed == (child_key,)
    assert np.allclose(root.edge_base_alpha, before_base)
    assert np.allclose(root.edge_evidence_E, before_evidence)
    assert np.allclose(root.edge_post_alpha, before_post)


def test_transposition_two_parents_reference_same_canonical_child_node():
    arena = PosteriorArena(max_nodes=8, max_edges=8, num_actions=1, num_outcomes=3)
    parent_a = arena.add_expanded_node(
        key=np.array([1, 0, 0, 0], dtype=np.uint32),
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    parent_b = arena.add_expanded_node(
        key=np.array([2, 0, 0, 0], dtype=np.uint32),
        current_player=1,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    child = arena.add_expanded_node(
        key=np.array([3, 0, 0, 0], dtype=np.uint32),
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.array([1.0, 3.0, 9.0], dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    edge_a = int(arena.node_first_edge[parent_a])
    edge_b = int(arena.node_first_edge[parent_b])
    arena.edge_child_node[[edge_a, edge_b]] = child
    arena.edge_child_key[[edge_a, edge_b]] = arena.node_key[child]
    search = BatchedPosteriorArenaSearch(env=object())
    search.arena = arena

    search._update_edge_base_from_children(
        np.array([edge_a, edge_b], dtype=np.int32),
        np.array([child, child], dtype=np.int32),
    )

    assert int(arena.edge_child_node[edge_a]) == int(arena.edge_child_node[edge_b]) == child
    assert np.array_equal(arena.edge_child_key[edge_a], arena.edge_child_key[edge_b])
    assert np.allclose(arena.edge_base_alpha[edge_a], np.array([1.0, 3.0, 9.0], dtype=np.float32))
    assert np.allclose(arena.edge_base_alpha[edge_b], np.array([9.0, 3.0, 1.0], dtype=np.float32))


def test_arena_batch_expansion_preserves_variable_legal_action_order():
    arena = PosteriorArena(max_nodes=4, max_edges=8, num_actions=4, num_outcomes=3)
    q_alpha = np.arange(2 * 4 * 3, dtype=np.float32).reshape((2, 4, 3)) + 1.0
    logits = np.arange(2 * 4, dtype=np.float32).reshape((2, 4))

    node_ids = arena.add_expanded_nodes_batch(
        keys=np.array([[1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.uint32),
        current_players=np.array([0, 1], dtype=np.int32),
        legal_action_mask=np.array(
            [[True, False, True, False], [False, True, False, True]],
            dtype=bool,
        ),
        value_alpha=np.ones((2, 3), dtype=np.float32),
        policy_logits=logits,
        q_alpha=q_alpha,
        assume_unique_new=True,
    )

    assert node_ids.tolist() == [0, 1]
    assert arena.edge_action[:4].tolist() == [0, 2, 1, 3]
    assert np.allclose(arena.edge_base_alpha[0], q_alpha[0, 0])
    assert np.allclose(arena.edge_base_alpha[1], q_alpha[0, 2])
    assert np.allclose(arena.edge_base_alpha[2], q_alpha[1, 1])
    assert np.allclose(arena.edge_base_alpha[3], q_alpha[1, 3])
    assert arena.edge_logit[:4].tolist() == [0.0, 2.0, 5.0, 7.0]


def test_grouped_arena_expansion_preserves_original_row_order_across_legal_counts():
    arena = PosteriorArena(max_nodes=8, max_edges=16, num_actions=4, num_outcomes=3)
    q_alpha = np.arange(3 * 4 * 3, dtype=np.float32).reshape((3, 4, 3)) + 1.0
    logits = np.arange(3 * 4, dtype=np.float32).reshape((3, 4))

    node_ids = arena.add_expanded_nodes_batch(
        keys=np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]], dtype=np.uint32),
        current_players=np.array([0, 1, 0], dtype=np.int32),
        legal_action_mask=np.array(
            [
                [True, False, False, False],
                [False, True, True, False],
                [False, False, False, True],
            ],
            dtype=bool,
        ),
        value_alpha=np.ones((3, 3), dtype=np.float32),
        policy_logits=logits,
        q_alpha=q_alpha,
        assume_unique_new=True,
    )

    assert node_ids.shape == (3,)
    assert arena.node_key[node_ids].tolist() == [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]
    assert arena.edge_action[arena.node_first_edge[node_ids[0]] : arena.node_first_edge[node_ids[0]] + 1].tolist() == [0]
    assert arena.edge_action[arena.node_first_edge[node_ids[1]] : arena.node_first_edge[node_ids[1]] + 2].tolist() == [1, 2]
    assert arena.edge_action[arena.node_first_edge[node_ids[2]] : arena.node_first_edge[node_ids[2]] + 1].tolist() == [3]


def test_refresh_summaries_groups_parents_with_different_legal_counts():
    from scacchi.dirichlet_tree.arena_search import BatchedPosteriorArenaSearch

    arena = PosteriorArena(max_nodes=8, max_edges=16, num_actions=4, num_outcomes=3)
    node_ids = arena.add_expanded_nodes_batch(
        keys=np.array([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]], dtype=np.uint32),
        current_players=np.array([0, 0, 0], dtype=np.int32),
        legal_action_mask=np.array(
            [
                [True, False, False, False],
                [False, True, True, False],
                [False, False, False, True],
            ],
            dtype=bool,
        ),
        value_alpha=np.ones((3, 3), dtype=np.float32),
        policy_logits=np.zeros((3, 4), dtype=np.float32),
        q_alpha=np.ones((3, 4, 3), dtype=np.float32),
        assume_unique_new=True,
    )
    arena.edge_E[arena.node_first_edge[node_ids[0]]] = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    arena.edge_E[arena.node_first_edge[node_ids[1]]] = np.array([3.0, 0.0, 0.0], dtype=np.float32)

    search = BatchedPosteriorArenaSearch(env=object())
    search.arena = arena
    search._refresh_edges_and_summaries(
        arena.node_first_edge[node_ids],
        node_ids,
    )

    assert np.all(arena.node_summary_alpha[node_ids] > 0.0)
    assert arena.node_summary_alpha[node_ids[0], 2] > arena.node_summary_alpha[node_ids[0], 0]
    assert arena.node_summary_alpha[node_ids[1], 0] > 0.0


def test_in_memory_claim_missing_once_and_reports_existing_status():
    store = InMemoryNodeStore()
    key = StateKey((10, 0, 0, 0))

    first = store.claim_many_inflight([key, key])
    second = store.claim_many_inflight([key])
    node = NodeBlob.expanded_node(
        key=key,
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    store.put_many([node])
    third = store.claim_many_inflight([key])

    assert first.claimed == (key,)
    assert second.inflight == (key,)
    assert third.expanded == (key,)


def test_redis_store_claim_and_round_trip_with_fakeredis():
    fakeredis = pytest.importorskip("fakeredis")
    store = RedisNodeStore(fakeredis.FakeRedis(), namespace="dqaz:test")
    key = StateKey((1, 2, 3, 4))

    assert store.claim_many_inflight([key]).claimed == (key,)
    assert store.claim_many_inflight([key]).inflight == (key,)
    node = NodeBlob.expanded_node(
        key=key,
        current_player=0,
        legal_action_mask=np.array([True]),
        value_alpha=np.ones((3,), dtype=np.float32),
        policy_logits=np.zeros((1,), dtype=np.float32),
        q_alpha=np.ones((1, 3), dtype=np.float32),
    )
    store.put_many([node])

    assert store.claim_many_inflight([key]).expanded == (key,)
    assert store.get_many([key])[key].expanded
