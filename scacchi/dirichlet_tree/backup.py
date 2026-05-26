from __future__ import annotations

import numpy as np

from .selection import posterior_best_policy_target_np
from .store import NodeStore
from .types import (
    EVAL_EXPANDED,
    KEY_WORDS,
    VALUE_CACHE_CLEAN,
    VALUE_CACHE_DIRTY,
    NodeBlob,
    PathStep,
    StateKey,
    align_wdl,
)


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
    if bool(parent.edge_has_post[ix]):
        return
    aligned = align_wdl(child.value_alpha, child.current_player, parent.current_player)
    if not np.allclose(parent.edge_base_alpha[ix], aligned):
        parent.edge_base_alpha[ix] = aligned
        store.mark_dirty(parent.key)


def backup_path(
    store: NodeStore,
    *,
    path: list[PathStep],
    leaf_node: NodeBlob,
    leaf_value: np.ndarray,
    leaf_weight: float | None = None,
    rng: np.random.Generator | None = None,
    policy_mc_samples: int = 16,
    state_posterior_kappa_n: float = 9.0,
    normalize_ancestor_summary: bool = True,
) -> None:
    del normalize_ancestor_summary
    if not path:
        return

    beta_leaf = np.asarray(leaf_value, dtype=np.float32)
    if leaf_weight is not None:
        beta_leaf = np.asarray(leaf_weight, dtype=np.float32) * beta_leaf

    final_step = path[-1]
    final_parent = _require_node(store, final_step.key)
    final_ix = _action_index(final_parent, final_step.action)
    final_parent.publish_edge_post(
        final_ix,
        align_wdl(beta_leaf, leaf_node.current_player, final_parent.current_player),
        eval_count=1 + int(final_parent.edge_eval_count_R[final_ix]),
    )
    final_parent.edge_child_cache_version[final_ix] = -1
    store.mark_dirty(final_parent.key)
    for step in path:
        node = _require_node(store, step.key)
        node.mark_dirty()
        store.mark_dirty(node.key)
    for step in reversed(path):
        repair_path_to_root(
            store,
            step.key,
            rng=rng,
            num_samples=policy_mc_samples,
            state_posterior_kappa_n=state_posterior_kappa_n,
        )
    repair_dirty_frontier(
        store,
        rng=rng,
        num_samples=policy_mc_samples,
        state_posterior_kappa_n=state_posterior_kappa_n,
    )


def refresh_edge_from_child(
    store: NodeStore,
    *,
    parent_key: StateKey,
    action: int,
    child: NodeBlob | None = None,
) -> bool:
    parent = _require_node(store, parent_key)
    ix = _action_index(parent, action)
    if child is None:
        child_key = StateKey(tuple(int(x) for x in parent.child_keys[ix].reshape((KEY_WORDS,))))
        if child_key.is_zero():
            return True
        child = _require_node(store, child_key)
    if child.terminal:
        return True
    if child.status != EVAL_EXPANDED:
        return True
    if not child.has_child_evidence:
        return True
    if child.value_cache_status != VALUE_CACHE_CLEAN:
        parent.mark_dirty()
        store.mark_dirty(parent.key)
        return False
    if int(parent.edge_child_cache_version[ix]) == int(child.value_cache_version):
        return True
    parent.publish_edge_post(
        ix,
        align_wdl(child.value_cache_C, child.current_player, parent.current_player),
        eval_count=1 + int(child.downstream_eval_count),
    )
    parent.edge_child_cache_version[ix] = np.int64(child.value_cache_version)
    store.mark_dirty(parent.key)
    return True


def try_repair_node(
    store: NodeStore,
    key: StateKey,
    *,
    rng: np.random.Generator | None = None,
    num_samples: int = 16,
    state_posterior_kappa_n: float = 9.0,
) -> bool:
    node = _require_node(store, key)
    if node.terminal or node.status != EVAL_EXPANDED:
        return False
    if node.value_cache_status == VALUE_CACHE_CLEAN:
        return True
    if node.value_cache_status != VALUE_CACHE_DIRTY:
        return False

    epoch = int(node.edge_epoch)
    for action in node.legal_actions:
        if not refresh_edge_from_child(store, parent_key=node.key, action=int(action)):
            node.value_cache_status = VALUE_CACHE_DIRTY
            store.mark_dirty(node.key)
            return False
    if int(node.edge_epoch) != epoch:
        node.value_cache_status = VALUE_CACHE_DIRTY
        store.mark_dirty(node.key)
        return False
    if not node.has_child_evidence:
        node.publish_clean_cache(
            node.value_alpha,
            downstream_eval_count=0,
            pi_search=np.zeros((node.legal_actions.shape[0],), dtype=np.float16),
        )
        store.mark_dirty(node.key)
        _mark_parent_dirty_if_present(store, node)
        return True

    cache, n_down, pi = compute_state_search_posterior(
        node,
        rng=rng,
        num_samples=num_samples,
        state_posterior_kappa_n=state_posterior_kappa_n,
    )
    if int(node.edge_epoch) != epoch:
        node.value_cache_status = VALUE_CACHE_DIRTY
        store.mark_dirty(node.key)
        return False

    node.publish_clean_cache(cache, n_down, pi)
    store.mark_dirty(node.key)
    _mark_parent_dirty_if_present(store, node)
    return True


def repair_path_to_root(
    store: NodeStore,
    start_key: StateKey,
    *,
    rng: np.random.Generator | None = None,
    num_samples: int = 16,
    state_posterior_kappa_n: float = 9.0,
) -> None:
    key = start_key
    seen: set[StateKey] = set()
    while not key.is_zero() and key not in seen:
        seen.add(key)
        node = _require_node(store, key)
        if node.value_cache_status != VALUE_CACHE_CLEAN:
            ok = try_repair_node(
                store,
                key,
                rng=rng,
                num_samples=num_samples,
                state_posterior_kappa_n=state_posterior_kappa_n,
            )
            if not ok:
                return
        if node.parent_key is None or node.parent_key.is_zero():
            return
        key = node.parent_key


def repair_dirty_frontier(
    store: NodeStore,
    *,
    rng: np.random.Generator | None = None,
    num_samples: int = 16,
    state_posterior_kappa_n: float = 9.0,
    max_passes: int = 8,
) -> None:
    if not hasattr(store, "nodes"):
        return
    for _ in range(max(1, int(max_passes))):
        nodes = list(getattr(store, "nodes").values())
        dirty = [
            node
            for node in nodes
            if node.status == EVAL_EXPANDED and node.value_cache_status != VALUE_CACHE_CLEAN
        ]
        if not dirty:
            return
        changed = False
        for node in sorted(dirty, key=lambda n: int(n.depth), reverse=True):
            before = int(node.value_cache_version)
            try_repair_node(
                store,
                node.key,
                rng=rng,
                num_samples=num_samples,
                state_posterior_kappa_n=state_posterior_kappa_n,
            )
            changed = changed or int(node.value_cache_version) != before
        if not changed:
            return


def compute_state_search_posterior(
    node: NodeBlob,
    *,
    rng: np.random.Generator | None = None,
    num_samples: int = 16,
    state_posterior_kappa_n: float = 9.0,
) -> tuple[np.ndarray, int, np.ndarray]:
    if node.terminal or node.legal_actions.shape[0] == 0:
        pi = np.zeros((node.legal_actions.shape[0],), dtype=np.float16)
        return _positive(np.asarray(node.value_alpha, dtype=np.float32)), 0, pi
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if rng is None:
        rng = np.random.default_rng(0)
    alpha = node.edge_post_alpha
    legal = np.ones((alpha.shape[0],), dtype=bool)
    pi_search = posterior_best_policy_target_np(rng, alpha, legal, int(num_samples))
    e_v = np.sum(pi_search[:, None] * alpha, axis=0)
    n_down = int(np.sum(np.asarray(node.edge_eval_count_R, dtype=np.uint64)))
    denom = float(state_posterior_kappa_n) + float(n_down)
    gamma = 0.0 if denom <= 0.0 else float(n_down) / denom
    cache = (1.0 - gamma) * np.asarray(node.value_alpha, dtype=np.float32) + gamma * e_v
    return _positive(cache), n_down, pi_search


def terminal_dirichlet(
    outcome_index: int,
    num_outcomes: int = 3,
    *,
    kappa_terminal: float = 8.0,
    epsilon_terminal: float = 1e-6,
) -> np.ndarray:
    out = np.full((num_outcomes,), float(epsilon_terminal), dtype=np.float32)
    out[int(outcome_index)] += np.float32(kappa_terminal)
    return out


def state_search_posterior(
    node: NodeBlob,
    *,
    rng: np.random.Generator | None = None,
    num_samples: int = 16,
) -> np.ndarray:
    cache, _, _ = compute_state_search_posterior(node, rng=rng, num_samples=num_samples)
    return cache


def _mark_parent_dirty_if_present(store: NodeStore, node: NodeBlob) -> None:
    if node.parent_key is None or node.parent_key.is_zero():
        return
    parent = store.get_many([node.parent_key]).get(node.parent_key)
    if parent is None:
        return
    parent.mark_dirty()
    store.mark_dirty(parent.key)


def _require_node(store: NodeStore, key: StateKey) -> NodeBlob:
    node = store.get_many([key])[key]
    if node is None:
        raise KeyError(f"missing node for key {key.hex}")
    return node


def _action_index(node: NodeBlob, action: int) -> int:
    try:
        return node.action_to_index[int(action)]
    except KeyError as exc:
        raise KeyError(f"action {action} is not legal at node {node.key.hex}") from exc


def _positive(alpha: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(alpha, dtype=np.float32), np.float32(1e-6))
