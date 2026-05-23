from __future__ import annotations

import numpy as np

from .selection import posterior_best_policy_target_np
from .store import NodeStore
from .types import KEY_WORDS, NodeBlob, PathStep, StateKey, align_wdl


def update_parent_child_edge(
    store: NodeStore,
    *,
    parent_key: StateKey,
    action: int,
    child_key: StateKey,
) -> None:
    parent = _require_node(store, parent_key)
    ix = _action_index(parent, action)
    words = child_key.to_array()
    if not np.array_equal(parent.child_keys[ix], words):
        parent.child_keys[ix] = words
        parent.mark_dirty()
        store.mark_dirty(parent.key)


def update_edge_base_from_child(
    store: NodeStore,
    *,
    parent_key: StateKey,
    action: int,
    child: NodeBlob,
) -> None:
    if child.terminal:
        return
    parent = _require_node(store, parent_key)
    ix = _action_index(parent, action)
    aligned = align_wdl(child.value_alpha, child.current_player, parent.current_player)
    if not np.allclose(parent.edge_base_alpha[ix], aligned):
        parent.edge_base_alpha[ix] = aligned
        parent.mark_dirty()
        store.mark_dirty(parent.key)


def backup_path(
    store: NodeStore,
    *,
    path: list[PathStep],
    leaf_node: NodeBlob,
    leaf_value: np.ndarray,
    leaf_weight: float,
    c_state: float,
    rng: np.random.Generator | None = None,
    backup_mc_samples: int = 16,
    normalize_ancestor_summary: bool = True,
) -> None:
    del normalize_ancestor_summary
    if not path:
        return

    final_step = path[-1]
    final_parent = _require_node(store, final_step.key)
    final_ix = _action_index(final_parent, final_step.action)
    final_parent.edge_evidence_E[final_ix] += np.asarray(leaf_weight, dtype=np.float32) * align_wdl(
        leaf_value,
        leaf_node.current_player,
        final_parent.current_player,
    )
    final_parent.visits[final_ix] += np.uint32(1)
    final_parent.mark_dirty()
    store.mark_dirty(final_parent.key)

    for step in reversed(path[:-1]):
        parent = _require_node(store, step.key)
        ix = _action_index(parent, step.action)
        child_key = StateKey(tuple(int(x) for x in parent.child_keys[ix].reshape((KEY_WORDS,))))
        child = _require_node(store, child_key)
        summary = state_search_posterior(
            child,
            rng=rng,
            num_samples=backup_mc_samples,
        )
        parent.edge_evidence_E[ix] += np.asarray(c_state, dtype=np.float32) * align_wdl(
            summary,
            child.current_player,
            parent.current_player,
        )
        parent.visits[ix] += np.uint32(1)
        parent.mark_dirty()
        store.mark_dirty(parent.key)


def terminal_one_hot(outcome_index: int, num_outcomes: int = 3) -> np.ndarray:
    out = np.zeros((num_outcomes,), dtype=np.float32)
    out[int(outcome_index)] = 1.0
    return out


def state_search_posterior(
    node: NodeBlob,
    *,
    rng: np.random.Generator | None = None,
    num_samples: int = 16,
) -> np.ndarray:
    if node.terminal or node.legal_actions.shape[0] == 0:
        return np.maximum(np.asarray(node.value_alpha, dtype=np.float32), np.float32(1e-6))
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if rng is None:
        rng = np.random.default_rng(0)
    alpha = node.edge_post_alpha
    legal = np.ones((alpha.shape[0],), dtype=bool)
    pi_search = posterior_best_policy_target_np(rng, alpha, legal, int(num_samples))
    beta_v = np.sum(pi_search[:, None] * alpha, axis=0)
    return np.maximum(np.asarray(beta_v, dtype=np.float32), np.float32(1e-6))


def _require_node(store: NodeStore, key: StateKey) -> NodeBlob:
    node = store.get_many([key])[key]
    if node is None:
        raise KeyError(f"missing node for key {key.redis_hex}")
    return node


def _action_index(node: NodeBlob, action: int) -> int:
    try:
        return node.action_to_index[int(action)]
    except KeyError as exc:
        raise KeyError(f"action {action} is not legal at node {node.key.redis_hex}") from exc
