import numpy as np
import pytest
import jax
import jax.numpy as jnp

from scacchi.dirichlet_tree.backup import backup_path, update_edge_base_from_child
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
