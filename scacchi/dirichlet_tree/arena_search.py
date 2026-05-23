from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from .selection import greedy_q_action, posterior_best_policy_target_np, thompson_select_jax
from .state_hash import canonical_state_key
from .types import LeafEvaluator, SearchConfig, SearchResult, outcome_mean, terminal_outcome_from_reward


UNKNOWN = -1
STATUS_INFLIGHT = np.uint8(0)
STATUS_EXPANDED = np.uint8(1)
STATUS_TERMINAL = np.uint8(2)

_STEP_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, Any]] = {}
_STEP_INFO_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, Any]] = {}
_KEY_CACHE: dict[type, Any] = {}


@dataclass(slots=True)
class _PendingBatch:
    observations: Any
    key_words: np.ndarray
    players: np.ndarray
    legal: np.ndarray
    root_ids: np.ndarray
    path_nodes: np.ndarray
    path_edges: np.ndarray
    path_len: np.ndarray

    @property
    def size(self) -> int:
        return int(self.key_words.shape[0])


class PosteriorArena:
    def __init__(
        self,
        *,
        max_nodes: int,
        max_edges: int,
        num_actions: int,
        num_outcomes: int,
    ) -> None:
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self.num_actions = int(num_actions)
        self.num_outcomes = int(num_outcomes)
        self.num_nodes = 0
        self.num_edges = 0

        self.node_status = np.zeros((self.max_nodes,), dtype=np.uint8)
        self.node_key = np.zeros((self.max_nodes, 4), dtype=np.uint32)
        self.node_current_player = np.zeros((self.max_nodes,), dtype=np.int8)
        self.node_first_edge = np.zeros((self.max_nodes,), dtype=np.int32)
        self.node_num_edges = np.zeros((self.max_nodes,), dtype=np.int16)
        self.node_value_alpha = np.ones((self.max_nodes, self.num_outcomes), dtype=np.float32)
        self.node_summary_alpha = np.ones((self.max_nodes, self.num_outcomes), dtype=np.float32)
        self.node_terminal_outcome = np.full((self.max_nodes,), -1, dtype=np.int8)

        self.edge_parent_node = np.full((self.max_edges,), UNKNOWN, dtype=np.int32)
        self.edge_action = np.zeros((self.max_edges,), dtype=np.int32)
        self.edge_child_node = np.full((self.max_edges,), UNKNOWN, dtype=np.int32)
        self.edge_child_key = np.zeros((self.max_edges, 4), dtype=np.uint32)
        self.edge_base_alpha = np.ones((self.max_edges, self.num_outcomes), dtype=np.float32)
        self.edge_E = np.zeros((self.max_edges, self.num_outcomes), dtype=np.float32)
        self.edge_post_alpha = np.ones((self.max_edges, self.num_outcomes), dtype=np.float32)
        self.edge_logit = np.zeros((self.max_edges,), dtype=np.float32)
        self.edge_visits = np.zeros((self.max_edges,), dtype=np.uint32)

        self.key_to_node: dict[bytes, int] = {}
        self._key_index_complete = True
        self.sorted_key_view: np.ndarray | None = None

    def add_expanded_node(
        self,
        *,
        key: np.ndarray,
        current_player: int,
        legal_action_mask: np.ndarray,
        value_alpha: np.ndarray,
        policy_logits: np.ndarray,
        q_alpha: np.ndarray,
    ) -> int:
        self.ensure_key_index()
        key_id = _key_id(key)
        existing = self.key_to_node.get(key_id)
        if existing is not None and self.node_status[existing] != STATUS_INFLIGHT:
            return existing

        legal_actions = np.flatnonzero(np.asarray(legal_action_mask, dtype=bool)).astype(np.int32)
        node_id = existing if existing is not None else self._alloc_node(key_id)
        first_edge = self.num_edges
        edge_count = int(legal_actions.shape[0])
        if first_edge + edge_count > self.max_edges:
            raise MemoryError("posterior arena edge capacity exceeded")
        self.sorted_key_view = None
        self.node_status[node_id] = STATUS_EXPANDED
        self.node_key[node_id] = np.asarray(key, dtype=np.uint32)
        self.node_current_player[node_id] = np.int8(current_player)
        self.node_first_edge[node_id] = np.int32(first_edge)
        self.node_num_edges[node_id] = np.int16(edge_count)
        self.node_value_alpha[node_id] = _positive(value_alpha)
        self.node_terminal_outcome[node_id] = np.int8(-1)

        end_edge = first_edge + edge_count
        self.edge_parent_node[first_edge:end_edge] = node_id
        self.edge_action[first_edge:end_edge] = legal_actions
        self.edge_child_node[first_edge:end_edge] = UNKNOWN
        self.edge_child_key[first_edge:end_edge] = 0
        sparse_q = _positive(np.asarray(q_alpha, dtype=np.float32)[legal_actions])
        self.edge_base_alpha[first_edge:end_edge] = sparse_q
        self.edge_E[first_edge:end_edge] = 0.0
        self.edge_post_alpha[first_edge:end_edge] = sparse_q
        self.edge_logit[first_edge:end_edge] = np.asarray(policy_logits, dtype=np.float32)[legal_actions]
        self.edge_visits[first_edge:end_edge] = 0
        self.num_edges = end_edge
        self.recompute_summary(node_id)
        return node_id

    def add_expanded_nodes_batch(
        self,
        *,
        keys: np.ndarray,
        current_players: np.ndarray,
        legal_action_mask: np.ndarray,
        value_alpha: np.ndarray,
        policy_logits: np.ndarray,
        q_alpha: np.ndarray,
        assume_unique_new: bool = False,
    ) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.uint32).reshape((-1, 4))
        count = int(keys.shape[0])
        if count == 0:
            return np.zeros((0,), dtype=np.int32)

        legal = np.asarray(legal_action_mask, dtype=bool)
        if (
            legal.ndim != 2
            or not np.all(legal == legal[0])
            or (
                not assume_unique_new
                and (
                    _first_unique_indices(keys).shape[0] != count
                    or not self._keys_are_new(keys)
                )
            )
        ):
            return np.asarray(
                [
                    self.add_expanded_node(
                        key=keys[ix],
                        current_player=int(current_players[ix]),
                        legal_action_mask=legal[ix],
                        value_alpha=value_alpha[ix],
                        policy_logits=policy_logits[ix],
                        q_alpha=q_alpha[ix],
                    )
                    for ix in range(count)
                ],
                dtype=np.int32,
            )

        legal_actions = np.flatnonzero(legal[0]).astype(np.int32)
        edges_per_node = int(legal_actions.shape[0])
        node_start = self.num_nodes
        edge_start = self.num_edges
        edge_count = count * edges_per_node
        if node_start + count > self.max_nodes:
            raise MemoryError("posterior arena node capacity exceeded")
        if edge_start + edge_count > self.max_edges:
            raise MemoryError("posterior arena edge capacity exceeded")

        node_ids = np.arange(node_start, node_start + count, dtype=np.int32)
        self.num_nodes += count
        self.sorted_key_view = None
        self.node_status[node_ids] = STATUS_EXPANDED
        self.node_key[node_ids] = keys
        self.node_current_player[node_ids] = np.asarray(current_players, dtype=np.int8)
        self.node_first_edge[node_ids] = edge_start + np.arange(count, dtype=np.int32) * edges_per_node
        self.node_num_edges[node_ids] = np.int16(edges_per_node)
        self.node_value_alpha[node_ids] = _positive(value_alpha)
        self.node_terminal_outcome[node_ids] = np.int8(-1)
        if self._key_index_complete and count < 4096:
            self.key_to_node.update((keys[ix].tobytes(), int(node_ids[ix])) for ix in range(count))
        else:
            self._key_index_complete = False

        if edges_per_node == 0:
            self.node_summary_alpha[node_ids] = self.node_value_alpha[node_ids]
            return node_ids

        edge_end = edge_start + edge_count
        edge_ids = np.arange(edge_start, edge_end, dtype=np.int32)
        self.edge_parent_node[edge_start:edge_end] = np.repeat(node_ids, edges_per_node)
        self.edge_action[edge_start:edge_end] = np.tile(legal_actions, count)
        self.edge_child_node[edge_start:edge_end] = UNKNOWN
        self.edge_child_key[edge_start:edge_end] = 0
        sparse_q = _positive(np.asarray(q_alpha, dtype=np.float32)[:, legal_actions, :])
        self.edge_base_alpha[edge_start:edge_end] = sparse_q.reshape((edge_count, self.num_outcomes))
        self.edge_E[edge_start:edge_end] = 0.0
        self.edge_post_alpha[edge_start:edge_end] = self.edge_base_alpha[edge_start:edge_end]
        self.edge_logit[edge_start:edge_end] = np.asarray(policy_logits, dtype=np.float32)[
            :, legal_actions
        ].reshape((edge_count,))
        self.edge_visits[edge_start:edge_end] = 0
        self.num_edges = edge_end
        self._recompute_uniform_summaries(node_ids, edges_per_node)
        return node_ids

    def add_terminal_node(
        self,
        *,
        key: np.ndarray,
        current_player: int,
        terminal_outcome: int,
    ) -> int:
        self.ensure_key_index()
        key_id = _key_id(key)
        existing = self.key_to_node.get(key_id)
        if existing is not None and self.node_status[existing] != STATUS_INFLIGHT:
            return existing
        node_id = existing if existing is not None else self._alloc_node(key_id)
        self.sorted_key_view = None
        self.node_status[node_id] = STATUS_TERMINAL
        self.node_key[node_id] = np.asarray(key, dtype=np.uint32)
        self.node_current_player[node_id] = np.int8(current_player)
        self.node_first_edge[node_id] = np.int32(self.num_edges)
        self.node_num_edges[node_id] = np.int16(0)
        self.node_value_alpha[node_id] = 1e-6
        self.node_value_alpha[node_id, int(terminal_outcome)] = 1.0
        self.node_summary_alpha[node_id] = self.node_value_alpha[node_id]
        self.node_terminal_outcome[node_id] = np.int8(terminal_outcome)
        return node_id

    def recompute_summary(self, node_id: int) -> None:
        start = int(self.node_first_edge[node_id])
        count = int(self.node_num_edges[node_id])
        if count <= 0:
            self.node_summary_alpha[node_id] = self.node_value_alpha[node_id]
            return
        edge_slice = slice(start, start + count)
        post = _positive(self.edge_base_alpha[edge_slice] + self.edge_E[edge_slice])
        self.edge_post_alpha[edge_slice] = post
        q_mean = (post[:, -1] - post[:, 0]) / np.sum(post, axis=-1)
        q_mean = q_mean - np.max(q_mean)
        pi = np.exp(q_mean)
        pi = pi / np.sum(pi)
        self.node_summary_alpha[node_id] = _positive(np.sum(pi[:, None] * post, axis=0))

    def _recompute_uniform_summaries(self, node_ids: np.ndarray, edges_per_node: int) -> None:
        if edges_per_node <= 0:
            self.node_summary_alpha[node_ids] = self.node_value_alpha[node_ids]
            return
        first_edges = self.node_first_edge[node_ids].astype(np.int32)
        edge_ids = first_edges[:, None] + np.arange(edges_per_node, dtype=np.int32)[None, :]
        post = _positive(self.edge_base_alpha[edge_ids] + self.edge_E[edge_ids])
        self.edge_post_alpha[edge_ids.reshape((-1,))] = post.reshape((-1, self.num_outcomes))
        q_mean = (post[..., -1] - post[..., 0]) / np.sum(post, axis=-1)
        q_mean = q_mean - np.max(q_mean, axis=-1, keepdims=True)
        pi = np.exp(q_mean)
        pi = pi / np.sum(pi, axis=-1, keepdims=True)
        self.node_summary_alpha[node_ids] = _positive(np.sum(pi[..., None] * post, axis=1))

    def _alloc_node(self, key_id: bytes) -> int:
        if self.num_nodes >= self.max_nodes:
            raise MemoryError("posterior arena node capacity exceeded")
        node_id = self.num_nodes
        self.num_nodes += 1
        self.sorted_key_view = None
        if self._key_index_complete:
            self.key_to_node[key_id] = node_id
        return node_id

    def _keys_are_new(self, keys: np.ndarray) -> bool:
        self.ensure_key_index()
        for ix in range(keys.shape[0]):
            existing = self.key_to_node.get(keys[ix].tobytes())
            if existing is not None and self.node_status[existing] != STATUS_INFLIGHT:
                return False
        return True

    def ensure_key_index(self) -> None:
        if self._key_index_complete:
            return
        self.key_to_node = {
            self.node_key[ix].tobytes(): int(ix)
            for ix in range(self.num_nodes)
        }
        self._key_index_complete = True
        self.sorted_key_view = None


class BatchedPosteriorArenaSearch:
    def __init__(self, *, env: Any, rng_key: jax.Array | None = None) -> None:
        self.env = env
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        seed = int(
            jax.device_get(
                jax.random.randint(rng_key, (), minval=0, maxval=np.iinfo(np.int32).max)
            )
        )
        self.rng = np.random.default_rng(seed)
        self.jax_key = rng_key
        self.arena: PosteriorArena | None = None
        self.root_node_ids: np.ndarray | None = None
        self.root_keys: np.ndarray | None = None

    def search_batch(
        self,
        root_states: list[Any],
        leaf_evaluator: LeafEvaluator,
        config: SearchConfig,
        *,
        max_nodes: int | None = None,
        max_edges: int | None = None,
    ) -> SearchResult:
        if not root_states:
            raise ValueError("root_states must not be empty")
        root_state_batch = _stack_states(root_states)
        return self.search_state_batch(
            root_state_batch,
            leaf_evaluator,
            config,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )

    def search_state_batch(
        self,
        root_state_batch: Any,
        leaf_evaluator: LeafEvaluator,
        config: SearchConfig,
        *,
        max_nodes: int | None = None,
        max_edges: int | None = None,
    ) -> SearchResult:
        original_num_roots = int(root_state_batch.current_player.shape[0])
        root_keys_all = np.asarray(jax.device_get(_batched_key_fn(root_state_batch)), dtype=np.uint32)
        unique_root_indices, root_inverse, sorted_root_key_view = _unique_key_indices_inverse_sorted(root_keys_all)
        if unique_root_indices.shape[0] != original_num_roots:
            root_state_batch = _select_state_rows(root_state_batch, unique_root_indices)
            root_keys = root_keys_all[unique_root_indices]
        else:
            root_keys = root_keys_all

        root_observations = root_state_batch.observation
        root_logits, root_value_alpha, root_q_alpha = jax.device_get(
            leaf_evaluator(root_observations)
        )
        num_roots = int(root_logits.shape[0])
        num_actions = int(root_logits.shape[-1])
        num_outcomes = int(root_value_alpha.shape[-1])
        root_players, root_legal, root_term, root_rewards = jax.device_get(
            (
                root_state_batch.current_player,
                root_state_batch.legal_action_mask,
                root_state_batch.terminated,
                root_state_batch.rewards,
            )
        )
        root_players = np.asarray(root_players, dtype=np.int32)
        root_legal = np.asarray(root_legal, dtype=bool)
        root_term = np.asarray(root_term, dtype=bool)
        root_rewards = np.asarray(root_rewards, dtype=np.float32)

        if max_nodes is None:
            max_nodes = _default_max_nodes(num_roots, config)
        if max_edges is None:
            max_edges = max_nodes * max(1, num_actions)
        arena = PosteriorArena(
            max_nodes=max_nodes,
            max_edges=max_edges,
            num_actions=num_actions,
            num_outcomes=num_outcomes,
        )
        if not np.any(root_term):
            root_node_ids = arena.add_expanded_nodes_batch(
                keys=root_keys,
                current_players=root_players,
                legal_action_mask=root_legal,
                value_alpha=root_value_alpha,
                policy_logits=root_logits,
                q_alpha=root_q_alpha,
                assume_unique_new=True,
            )
        else:
            root_node_ids = np.empty((num_roots,), dtype=np.int32)
            for ix in range(num_roots):
                if root_term[ix]:
                    outcome = terminal_outcome_from_reward(float(root_rewards[ix, root_players[ix]]), num_outcomes)
                    root_node_ids[ix] = arena.add_terminal_node(
                        key=root_keys[ix],
                        current_player=int(root_players[ix]),
                        terminal_outcome=outcome,
                    )
                else:
                    root_node_ids[ix] = arena.add_expanded_node(
                        key=root_keys[ix],
                        current_player=int(root_players[ix]),
                        legal_action_mask=root_legal[ix],
                        value_alpha=root_value_alpha[ix],
                        policy_logits=root_logits[ix],
                        q_alpha=root_q_alpha[ix],
                    )

        arena.sorted_key_view = sorted_root_key_view
        self.arena = arena
        self.root_node_ids = root_node_ids
        self.root_keys = root_keys
        self._run_wavefront(root_state_batch, root_node_ids, leaf_evaluator, config)
        result = self.finish_search(config)
        if unique_root_indices.shape[0] != original_num_roots:
            return _broadcast_search_result(result, root_inverse)
        return result

    def finish_search(self, config: SearchConfig) -> SearchResult:
        if self.arena is None or self.root_node_ids is None:
            raise ValueError("search has not been initialized")
        arena = self.arena
        dense_result = self._finish_search_dense(config)
        if dense_result is not None:
            return dense_result

        actions = np.zeros((self.root_node_ids.shape[0],), dtype=np.int32)
        policies = np.zeros((self.root_node_ids.shape[0], arena.num_actions), dtype=np.float32)
        beta_q = np.zeros((self.root_node_ids.shape[0], arena.num_actions, arena.num_outcomes), dtype=np.float32)
        beta_v = np.zeros((self.root_node_ids.shape[0], arena.num_outcomes), dtype=np.float32)
        q_mass = np.zeros((self.root_node_ids.shape[0], arena.num_actions), dtype=np.float32)
        alpha_root = np.zeros_like(beta_q)
        for out_ix, node_id in enumerate(self.root_node_ids):
            start = int(arena.node_first_edge[node_id])
            count = int(arena.node_num_edges[node_id])
            legal = np.zeros((arena.num_actions,), dtype=bool)
            for edge_id in range(start, start + count):
                action = int(arena.edge_action[edge_id])
                legal[action] = True
                alpha_root[out_ix, action] = arena.edge_post_alpha[edge_id]
                beta_q[out_ix, action] = arena.edge_post_alpha[edge_id]
                q_mass[out_ix, action] = np.sum(arena.edge_E[edge_id])
            policies[out_ix] = posterior_best_policy_target_np(
                self.rng,
                alpha_root[out_ix],
                legal,
                config.policy_mc_samples,
            )
            value_proxy = np.sum(policies[out_ix, :, None] * alpha_root[out_ix], axis=0)
            beta_v[out_ix] = (
                arena.node_value_alpha[node_id]
                + np.asarray(config.c_value_search, dtype=np.float32) * value_proxy
            )
            actions[out_ix] = _commit_action(self.rng, config, policies[out_ix], alpha_root[out_ix], legal)

        return SearchResult(
            action=jnp.asarray(actions, dtype=jnp.int32),
            action_weights=jnp.asarray(policies, dtype=jnp.float32),
            beta_Q_target=jnp.asarray(beta_q, dtype=jnp.float32),
            beta_V_target=jnp.asarray(beta_v, dtype=jnp.float32),
            q_evidence_mass=jnp.asarray(q_mass, dtype=jnp.float32),
            alpha_root=jnp.asarray(alpha_root, dtype=jnp.float32),
        )

    def _finish_search_dense(self, config: SearchConfig) -> SearchResult | None:
        if self.arena is None or self.root_node_ids is None:
            return None
        arena = self.arena
        root_ids = self.root_node_ids.astype(np.int32)
        if root_ids.size == 0:
            return None
        edge_counts = arena.node_num_edges[root_ids].astype(np.int32)
        if not np.all(edge_counts == edge_counts[0]):
            return None
        edge_count = int(edge_counts[0])
        if edge_count <= 0:
            return None

        first_edges = arena.node_first_edge[root_ids].astype(np.int32)
        edge_ids = first_edges[:, None] + np.arange(edge_count, dtype=np.int32)[None, :]
        action_ids = arena.edge_action[edge_ids].astype(np.int32)
        row_ids = np.arange(root_ids.shape[0], dtype=np.int32)[:, None]

        alpha_root = np.zeros(
            (root_ids.shape[0], arena.num_actions, arena.num_outcomes),
            dtype=np.float32,
        )
        alpha_sparse = arena.edge_post_alpha[edge_ids]
        alpha_root[row_ids, action_ids] = alpha_sparse
        beta_q = alpha_root.copy()

        q_mass = np.zeros((root_ids.shape[0], arena.num_actions), dtype=np.float32)
        q_mass[row_ids, action_ids] = np.sum(arena.edge_E[edge_ids], axis=-1)
        legal = np.zeros((root_ids.shape[0], arena.num_actions), dtype=bool)
        legal[row_ids, action_ids] = True

        policies = _posterior_best_policy_target_batch_np(
            self.rng,
            alpha_root,
            legal,
            config.policy_mc_samples,
        )
        value_proxy = np.sum(policies[:, :, None] * alpha_root, axis=1)
        beta_v = (
            arena.node_value_alpha[root_ids]
            + np.asarray(config.c_value_search, dtype=np.float32) * value_proxy
        )
        actions = _commit_actions_batch(self.rng, config, policies, alpha_root, legal)
        return SearchResult(
            action=jnp.asarray(actions, dtype=jnp.int32),
            action_weights=jnp.asarray(policies, dtype=jnp.float32),
            beta_Q_target=jnp.asarray(beta_q, dtype=jnp.float32),
            beta_V_target=jnp.asarray(beta_v, dtype=jnp.float32),
            q_evidence_mass=jnp.asarray(q_mass, dtype=jnp.float32),
            alpha_root=jnp.asarray(alpha_root, dtype=jnp.float32),
        )

    def _run_wavefront(
        self,
        root_state_batch: Any,
        root_node_ids: np.ndarray,
        leaf_evaluator: LeafEvaluator,
        config: SearchConfig,
    ) -> None:
        assert self.arena is not None
        arena = self.arena
        num_roots = int(root_node_ids.shape[0])
        done = np.zeros((num_roots,), dtype=np.int32)
        max_attempts = max(1, num_roots * config.num_simulations * (config.max_depth + 4) * 4)
        attempts = 0
        while np.any(done < config.num_simulations):
            attempts += 1
            if attempts > max_attempts:
                unfinished = np.flatnonzero(done < config.num_simulations).tolist()
                raise RuntimeError(f"arena wavefront posterior search stalled for roots {unfinished}")

            unfinished = np.flatnonzero(done < config.num_simulations).astype(np.int32)
            if unfinished.size == 0:
                break
            lane_root_ids = np.repeat(unfinished, max(1, int(config.num_lanes_per_root))).astype(np.int32)
            current_node_ids = root_node_ids[lane_root_ids].copy()
            state_batch = _select_state_rows(root_state_batch, lane_root_ids)
            path_nodes = np.full((lane_root_ids.shape[0], config.max_depth), UNKNOWN, dtype=np.int32)
            path_edges = np.full((lane_root_ids.shape[0], config.max_depth), UNKNOWN, dtype=np.int32)
            path_len = np.zeros((lane_root_ids.shape[0],), dtype=np.int16)

            pending = self._traverse_lanes(
                state_batch,
                lane_root_ids,
                current_node_ids,
                path_nodes,
                path_edges,
                path_len,
                done,
                config,
            )
            self._evaluate_pending(pending, leaf_evaluator, done, config)

    def _traverse_lanes(
        self,
        state_batch: Any,
        lane_root_ids: np.ndarray,
        current_node_ids: np.ndarray,
        path_nodes: np.ndarray,
        path_edges: np.ndarray,
        path_len: np.ndarray,
        done: np.ndarray,
        config: SearchConfig,
    ) -> list[_PendingBatch]:
        assert self.arena is not None
        arena = self.arena
        pending_batches: list[_PendingBatch] = []
        active_rows = np.arange(lane_root_ids.shape[0], dtype=np.int32)
        active_state_pos = np.arange(lane_root_ids.shape[0], dtype=np.int32)

        for _ in range(config.max_depth):
            if active_rows.size == 0:
                break
            still_needed = done[lane_root_ids[active_rows]] < config.num_simulations
            active_rows = active_rows[still_needed]
            active_state_pos = active_state_pos[still_needed]
            if active_rows.size == 0:
                break

            node_ids = current_node_ids[active_rows]
            status = arena.node_status[node_ids]
            terminal_rows = active_rows[status == STATUS_TERMINAL]
            for row in terminal_rows:
                if path_len[row] <= 0 or done[lane_root_ids[row]] >= config.num_simulations:
                    continue
                node_id = int(current_node_ids[row])
                outcome = int(arena.node_terminal_outcome[node_id])
                value = np.zeros((arena.num_outcomes,), dtype=np.float32)
                value[outcome] = 1.0
                self._backup_path(
                    path_nodes[row],
                    path_edges[row],
                    int(path_len[row]),
                    leaf_node_id=node_id,
                    leaf_value=value,
                    leaf_weight=config.c_terminal,
                    c_state=config.c_state,
                )
                done[lane_root_ids[row]] += 1

            selectable_mask = (status == STATUS_EXPANDED) & (arena.node_num_edges[node_ids] > 0)
            selectable_rows = active_rows[selectable_mask]
            selectable_state_pos = active_state_pos[selectable_mask]
            if selectable_rows.size == 0:
                break

            select_nodes = current_node_ids[selectable_rows]
            first_edges = arena.node_first_edge[select_nodes].astype(np.int32)
            edge_counts = arena.node_num_edges[select_nodes].astype(np.int32)
            if np.all(edge_counts == 1):
                selected_edge_ids = first_edges
                actions = arena.edge_action[selected_edge_ids].astype(np.int32)
            else:
                max_edges = int(np.max(edge_counts))
                offsets = np.arange(max_edges, dtype=np.int32)
                edge_ids = first_edges[:, None] + offsets[None, :]
                mask = offsets[None, :] < edge_counts[:, None]
                safe_edge_ids = np.where(mask, edge_ids, 0)
                alpha = np.where(
                    mask[..., None],
                    arena.edge_post_alpha[safe_edge_ids],
                    np.ones((selectable_rows.shape[0], max_edges, arena.num_outcomes), dtype=np.float32),
                )
                actions_padded = np.where(mask, arena.edge_action[safe_edge_ids], 0).astype(np.int32)
                self.jax_key, select_key = jax.random.split(self.jax_key)
                actions = np.asarray(
                    jax.device_get(
                        thompson_select_jax(
                            select_key,
                            jnp.asarray(alpha),
                            jnp.asarray(actions_padded),
                            jnp.asarray(mask),
                        )
                    ),
                    dtype=np.int32,
                )
                selected_pos = _selected_positions(actions_padded, actions, mask)
                selected_edge_ids = edge_ids[np.arange(edge_ids.shape[0]), selected_pos].astype(np.int32)

            depth_ix = path_len[selectable_rows].astype(np.int32)
            path_nodes[selectable_rows, depth_ix] = select_nodes
            path_edges[selectable_rows, depth_ix] = selected_edge_ids
            path_len[selectable_rows] += 1

            real_select_count = int(selectable_rows.shape[0])
            padded_state_pos, padded_actions = _pad_step_inputs(
                selectable_state_pos,
                actions,
                int(lane_root_ids.shape[0]),
            )
            selected_state_batch = _select_state_rows(state_batch, padded_state_pos)
            next_state_batch, key_words, terminated, players, rewards, legal = _batched_step_info(self.env)(
                selected_state_batch,
                jnp.asarray(padded_actions),
            )
            key_words, terminated, players, rewards, legal = jax.device_get(
                (key_words, terminated, players, rewards, legal)
            )
            key_words = np.asarray(key_words[:real_select_count], dtype=np.uint32)
            terminated = np.asarray(terminated[:real_select_count], dtype=bool)
            players = np.asarray(players[:real_select_count], dtype=np.int32)
            rewards = np.asarray(rewards[:real_select_count], dtype=np.float32)
            legal = np.asarray(legal[:real_select_count], dtype=bool)

            if self._can_fast_path_fresh_leaves(selected_edge_ids, key_words, terminated):
                arena.edge_child_key[selected_edge_ids] = key_words
                arena.edge_child_node[selected_edge_ids] = UNKNOWN
                pending_batches.append(
                    _make_pending_batch(
                        next_state_batch=next_state_batch,
                        key_words=key_words,
                        players=players,
                        legal=legal,
                        lane_root_ids=lane_root_ids,
                        path_nodes=path_nodes,
                        path_edges=path_edges,
                        path_len=path_len,
                        missing_rows=selectable_rows.astype(np.int32, copy=False),
                        missing_state_indices=np.arange(selectable_rows.shape[0], dtype=np.int32),
                        recycle_duplicates=config.duplicate_leaf_mode == "recycle_lane",
                    )
                )
                break

            arena.ensure_key_index()
            next_rows: list[int] = []
            next_state_indices: list[int] = []
            missing_rows: list[int] = []
            missing_state_indices: list[int] = []
            for ix, row in enumerate(selectable_rows):
                edge_id = int(selected_edge_ids[ix])
                child_node_id = arena.key_to_node.get(key_words[ix].tobytes(), UNKNOWN)
                arena.edge_child_key[edge_id] = key_words[ix]
                arena.edge_child_node[edge_id] = child_node_id

                if terminated[ix]:
                    if child_node_id == UNKNOWN:
                        outcome = terminal_outcome_from_reward(float(rewards[ix, players[ix]]), arena.num_outcomes)
                        child_node_id = arena.add_terminal_node(
                            key=key_words[ix],
                            current_player=int(players[ix]),
                            terminal_outcome=outcome,
                        )
                        arena.edge_child_node[edge_id] = child_node_id
                    outcome = int(arena.node_terminal_outcome[child_node_id])
                    value = np.zeros((arena.num_outcomes,), dtype=np.float32)
                    value[outcome] = 1.0
                    if done[lane_root_ids[row]] < config.num_simulations:
                        self._backup_path(
                            path_nodes[row],
                            path_edges[row],
                            int(path_len[row]),
                            leaf_node_id=child_node_id,
                            leaf_value=value,
                            leaf_weight=config.c_terminal,
                            c_state=config.c_state,
                        )
                        done[lane_root_ids[row]] += 1
                    continue

                if child_node_id == UNKNOWN:
                    missing_rows.append(int(row))
                    missing_state_indices.append(ix)
                    continue

                child_status = arena.node_status[child_node_id]
                if child_status == STATUS_INFLIGHT:
                    continue
                self._update_edge_base_from_child(edge_id, child_node_id)
                if child_status == STATUS_TERMINAL:
                    outcome = int(arena.node_terminal_outcome[child_node_id])
                    value = np.zeros((arena.num_outcomes,), dtype=np.float32)
                    value[outcome] = 1.0
                    if done[lane_root_ids[row]] < config.num_simulations:
                        self._backup_path(
                            path_nodes[row],
                            path_edges[row],
                            int(path_len[row]),
                            leaf_node_id=child_node_id,
                            leaf_value=value,
                            leaf_weight=config.c_terminal,
                            c_state=config.c_state,
                        )
                        done[lane_root_ids[row]] += 1
                else:
                    next_rows.append(int(row))
                    next_state_indices.append(ix)
                    current_node_ids[row] = child_node_id

            if missing_rows:
                pending_batches.append(
                    _make_pending_batch(
                        next_state_batch=next_state_batch,
                        key_words=key_words,
                        players=players,
                        legal=legal,
                        lane_root_ids=lane_root_ids,
                        path_nodes=path_nodes,
                        path_edges=path_edges,
                        path_len=path_len,
                        missing_rows=np.asarray(missing_rows, dtype=np.int32),
                        missing_state_indices=np.asarray(missing_state_indices, dtype=np.int32),
                        recycle_duplicates=config.duplicate_leaf_mode == "recycle_lane",
                    )
                )

            if not next_rows:
                break
            active_rows = np.asarray(next_rows, dtype=np.int32)
            padded_next_state_indices = _pad_indices(
                np.asarray(next_state_indices, dtype=np.int32),
                int(lane_root_ids.shape[0]),
            )
            state_batch = _select_state_rows(next_state_batch, padded_next_state_indices)
            active_state_pos = np.arange(active_rows.shape[0], dtype=np.int32)

        return pending_batches

    def _can_fast_path_fresh_leaves(
        self,
        selected_edge_ids: np.ndarray,
        key_words: np.ndarray,
        terminated: np.ndarray,
    ) -> bool:
        if self.arena is None or self.root_node_ids is None:
            return False
        if self.arena.num_nodes != int(self.root_node_ids.shape[0]):
            return False
        if np.any(terminated):
            return False
        if not np.all(self.arena.edge_child_node[selected_edge_ids] == UNKNOWN):
            return False
        return not _any_key_overlap(
            key_words,
            self.arena.node_key[: self.arena.num_nodes],
            sorted_existing_view=self.arena.sorted_key_view,
        )

    def _evaluate_pending(
        self,
        pending: list[_PendingBatch],
        leaf_evaluator: LeafEvaluator,
        done: np.ndarray,
        config: SearchConfig,
    ) -> None:
        if not pending:
            return
        assert self.arena is not None
        arena = self.arena
        for pending_batch in pending:
            for start in range(0, pending_batch.size, config.eval_batch_size):
                end = min(start + config.eval_batch_size, pending_batch.size)
                observations = pending_batch.observations[start:end]
                logits, value_alpha, q_alpha = jax.device_get(leaf_evaluator(observations))
                request_slice = slice(start, end)
                child_node_ids = arena.add_expanded_nodes_batch(
                    keys=pending_batch.key_words[request_slice],
                    current_players=pending_batch.players[request_slice],
                    legal_action_mask=pending_batch.legal[request_slice],
                    value_alpha=value_alpha,
                    policy_logits=logits,
                    q_alpha=q_alpha,
                    assume_unique_new=True,
                )
                self._backup_pending_rows(
                    child_node_ids=child_node_ids,
                    root_ids=pending_batch.root_ids[request_slice],
                    path_nodes=pending_batch.path_nodes[request_slice],
                    path_edges=pending_batch.path_edges[request_slice],
                    path_len=pending_batch.path_len[request_slice],
                    done=done,
                    leaf_weight=config.c_leaf,
                    c_state=config.c_state,
                    num_simulations=config.num_simulations,
                )

    def _backup_pending_rows(
        self,
        *,
        child_node_ids: np.ndarray,
        root_ids: np.ndarray,
        path_nodes: np.ndarray,
        path_edges: np.ndarray,
        path_len: np.ndarray,
        done: np.ndarray,
        leaf_weight: float,
        c_state: float,
        num_simulations: int,
    ) -> None:
        assert self.arena is not None
        arena = self.arena
        child_node_ids = np.asarray(child_node_ids, dtype=np.int32)
        root_ids = np.asarray(root_ids, dtype=np.int32)
        path_len = np.asarray(path_len, dtype=np.int16)
        active = _backup_active_mask(root_ids, path_len, done, int(num_simulations))
        if not np.any(active):
            return

        active_ix = np.flatnonzero(active).astype(np.int32)
        active_child_ids = child_node_ids[active_ix]
        active_root_ids = root_ids[active_ix]
        active_path_len = path_len[active_ix].astype(np.int32)
        final_edges = path_edges[active_ix, active_path_len - 1].astype(np.int32)
        final_parents = path_nodes[active_ix, active_path_len - 1].astype(np.int32)

        arena.edge_child_node[final_edges] = active_child_ids
        self._update_edge_base_from_children(final_edges, active_child_ids)
        leaf_value = outcome_mean(arena.node_value_alpha[active_child_ids])
        parent_players = arena.node_current_player[final_parents]
        child_players = arena.node_current_player[active_child_ids]
        aligned_leaf = _align_rows(leaf_value, parent_players != child_players)
        np.add.at(arena.edge_E, final_edges, np.asarray(leaf_weight, dtype=np.float32) * aligned_leaf)
        np.add.at(arena.edge_visits, final_edges, np.uint32(1))
        self._refresh_edges_and_summaries(final_edges, final_parents)
        np.add.at(done, active_root_ids, 1)

        max_len = int(np.max(active_path_len))
        for depth in range(max_len - 2, -1, -1):
            depth_mask = active_path_len > depth + 1
            if not np.any(depth_mask):
                continue
            row_ix = active_ix[depth_mask]
            edge_ids = path_edges[row_ix, depth].astype(np.int32)
            parent_ids = path_nodes[row_ix, depth].astype(np.int32)
            child_ids = arena.edge_child_node[edge_ids].astype(np.int32)
            valid = child_ids != UNKNOWN
            if not np.any(valid):
                continue
            edge_ids = edge_ids[valid]
            parent_ids = parent_ids[valid]
            child_ids = child_ids[valid]
            summary = outcome_mean(arena.node_summary_alpha[child_ids])
            parent_players = arena.node_current_player[parent_ids]
            child_players = arena.node_current_player[child_ids]
            aligned_summary = _align_rows(summary, parent_players != child_players)
            np.add.at(arena.edge_E, edge_ids, np.asarray(c_state, dtype=np.float32) * aligned_summary)
            np.add.at(arena.edge_visits, edge_ids, np.uint32(1))
            self._refresh_edges_and_summaries(edge_ids, parent_ids)

    def _update_edge_base_from_child(self, edge_id: int, child_node_id: int) -> None:
        assert self.arena is not None
        arena = self.arena
        parent_node_id = int(arena.edge_parent_node[edge_id])
        parent_player = int(arena.node_current_player[parent_node_id])
        child_player = int(arena.node_current_player[child_node_id])
        value = arena.node_value_alpha[child_node_id]
        arena.edge_base_alpha[edge_id] = value[::-1] if parent_player != child_player else value
        arena.edge_post_alpha[edge_id] = _positive(arena.edge_base_alpha[edge_id] + arena.edge_E[edge_id])
        arena.recompute_summary(parent_node_id)

    def _update_edge_base_from_children(self, edge_ids: np.ndarray, child_node_ids: np.ndarray) -> None:
        assert self.arena is not None
        arena = self.arena
        edge_ids = np.asarray(edge_ids, dtype=np.int32)
        child_node_ids = np.asarray(child_node_ids, dtype=np.int32)
        parent_node_ids = arena.edge_parent_node[edge_ids].astype(np.int32)
        parent_players = arena.node_current_player[parent_node_ids]
        child_players = arena.node_current_player[child_node_ids]
        value = arena.node_value_alpha[child_node_ids]
        arena.edge_base_alpha[edge_ids] = _align_rows(value, parent_players != child_players)
        arena.edge_post_alpha[edge_ids] = _positive(arena.edge_base_alpha[edge_ids] + arena.edge_E[edge_ids])

    def _refresh_edges_and_summaries(self, edge_ids: np.ndarray, parent_ids: np.ndarray) -> None:
        assert self.arena is not None
        arena = self.arena
        edge_ids = np.unique(np.asarray(edge_ids, dtype=np.int32))
        if edge_ids.size:
            arena.edge_post_alpha[edge_ids] = _positive(arena.edge_base_alpha[edge_ids] + arena.edge_E[edge_ids])
        parent_ids = np.unique(np.asarray(parent_ids, dtype=np.int32))
        if parent_ids.size == 0:
            return
        edge_counts = arena.node_num_edges[parent_ids].astype(np.int32)
        if np.all(edge_counts == edge_counts[0]):
            arena._recompute_uniform_summaries(parent_ids, int(edge_counts[0]))
            return
        for parent_id in parent_ids:
            arena.recompute_summary(int(parent_id))

    def _backup_path(
        self,
        path_nodes: np.ndarray | None,
        path_edges: np.ndarray | None,
        path_len: int,
        *,
        leaf_node_id: int,
        leaf_value: np.ndarray,
        leaf_weight: float,
        c_state: float,
    ) -> None:
        if path_nodes is None or path_edges is None or path_len <= 0:
            return
        assert self.arena is not None
        arena = self.arena
        final_edge_id = int(path_edges[path_len - 1])
        final_parent_id = int(path_nodes[path_len - 1])
        parent_player = int(arena.node_current_player[final_parent_id])
        leaf_player = int(arena.node_current_player[leaf_node_id])
        aligned = leaf_value[::-1] if parent_player != leaf_player else leaf_value
        arena.edge_E[final_edge_id] += np.asarray(leaf_weight, dtype=np.float32) * aligned
        arena.edge_visits[final_edge_id] += np.uint32(1)
        arena.edge_post_alpha[final_edge_id] = _positive(
            arena.edge_base_alpha[final_edge_id] + arena.edge_E[final_edge_id]
        )
        arena.recompute_summary(final_parent_id)

        for depth in range(path_len - 2, -1, -1):
            edge_id = int(path_edges[depth])
            parent_id = int(path_nodes[depth])
            child_id = int(arena.edge_child_node[edge_id])
            if child_id == UNKNOWN:
                continue
            summary = outcome_mean(arena.node_summary_alpha[child_id])
            parent_player = int(arena.node_current_player[parent_id])
            child_player = int(arena.node_current_player[child_id])
            aligned_summary = summary[::-1] if parent_player != child_player else summary
            arena.edge_E[edge_id] += np.asarray(c_state, dtype=np.float32) * aligned_summary
            arena.edge_visits[edge_id] += np.uint32(1)
            arena.edge_post_alpha[edge_id] = _positive(
                arena.edge_base_alpha[edge_id] + arena.edge_E[edge_id]
            )
            arena.recompute_summary(parent_id)

def run_arena_posterior_tree_search(
    *,
    env: Any,
    root_states: list[Any],
    leaf_evaluator: LeafEvaluator,
    rng_key: jax.Array,
    config: Any,
) -> SearchResult:
    from .search import search_config_from_any

    search_config = search_config_from_any(config, num_roots=len(root_states))
    search = BatchedPosteriorArenaSearch(env=env, rng_key=rng_key)
    return search.search_batch(
        root_states,
        leaf_evaluator,
        search_config,
        max_nodes=getattr(config, "wavefront_max_nodes", None),
        max_edges=getattr(config, "wavefront_max_edges", None),
    )


def _default_max_nodes(num_roots: int, config: SearchConfig) -> int:
    lanes = max(1, int(config.num_lanes_per_root))
    return max(16, int(num_roots) * (int(config.num_simulations) + 2) * lanes * 2)


def _make_pending_batch(
    *,
    next_state_batch: Any,
    key_words: np.ndarray,
    players: np.ndarray,
    legal: np.ndarray,
    lane_root_ids: np.ndarray,
    path_nodes: np.ndarray,
    path_edges: np.ndarray,
    path_len: np.ndarray,
    missing_rows: np.ndarray,
    missing_state_indices: np.ndarray,
    recycle_duplicates: bool,
) -> _PendingBatch:
    if recycle_duplicates and missing_state_indices.shape[0] > 1:
        keep = _first_unique_indices(key_words[missing_state_indices])
        missing_rows = missing_rows[keep]
        missing_state_indices = missing_state_indices[keep]
    max_path_len = int(np.max(path_len[missing_rows])) if missing_rows.size else 0
    return _PendingBatch(
        observations=next_state_batch.observation[jnp.asarray(missing_state_indices, dtype=jnp.int32)],
        key_words=key_words[missing_state_indices].copy(),
        players=players[missing_state_indices].astype(np.int32, copy=True),
        legal=legal[missing_state_indices].copy(),
        root_ids=lane_root_ids[missing_rows].astype(np.int32, copy=True),
        path_nodes=path_nodes[missing_rows, :max_path_len].copy(),
        path_edges=path_edges[missing_rows, :max_path_len].copy(),
        path_len=path_len[missing_rows].astype(np.int16, copy=True),
    )


def _first_unique_indices(key_words: np.ndarray) -> np.ndarray:
    return _unique_key_indices_inverse(key_words)[0]


def _key_void_view(key_words: np.ndarray) -> np.ndarray:
    keys = np.ascontiguousarray(key_words, dtype=np.uint32).reshape((-1, 4))
    return keys.view(np.dtype((np.void, keys.dtype.itemsize * 4))).reshape((-1,))


def _any_key_overlap(
    candidate_keys: np.ndarray,
    existing_keys: np.ndarray,
    *,
    sorted_existing_view: np.ndarray | None = None,
) -> bool:
    if candidate_keys.size == 0 or existing_keys.size == 0:
        return False
    candidate_view = _key_void_view(candidate_keys)
    existing_view = (
        np.asarray(sorted_existing_view)
        if sorted_existing_view is not None
        else np.sort(_key_void_view(existing_keys))
    )
    positions = np.searchsorted(existing_view, candidate_view)
    in_bounds = positions < existing_view.shape[0]
    if not np.any(in_bounds):
        return False
    return bool(np.any(existing_view[positions[in_bounds]] == candidate_view[in_bounds]))


def _unique_key_indices_inverse(key_words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first, inverse, _ = _unique_key_indices_inverse_sorted(key_words)
    return first, inverse


def _unique_key_indices_inverse_sorted(key_words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_words = np.ascontiguousarray(key_words, dtype=np.uint32).reshape((-1, 4))
    count = int(key_words.shape[0])
    if count <= 1:
        indices = np.arange(count, dtype=np.int32)
        return indices, indices, _key_void_view(key_words)
    key_view = _key_void_view(key_words)
    unique_sorted, first, inverse = np.unique(key_view, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.shape[0], dtype=order.dtype)
    return (
        first[order].astype(np.int32, copy=False),
        remap[inverse].astype(np.int32, copy=False),
        unique_sorted,
    )


def _broadcast_search_result(result: SearchResult, inverse: np.ndarray) -> SearchResult:
    inverse_jax = jnp.asarray(inverse, dtype=jnp.int32)
    return SearchResult(
        action=result.action[inverse_jax],
        action_weights=result.action_weights[inverse_jax],
        beta_Q_target=result.beta_Q_target[inverse_jax],
        beta_V_target=result.beta_V_target[inverse_jax],
        q_evidence_mass=result.q_evidence_mass[inverse_jax],
        alpha_root=result.alpha_root[inverse_jax],
    )


def _selected_positions(actions_padded: np.ndarray, actions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    matches = (actions_padded == actions[:, None]) & mask
    return np.argmax(matches, axis=1).astype(np.int32)


def _pad_indices(indices: np.ndarray, size: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int32)
    if indices.shape[0] >= size:
        return indices
    if indices.shape[0] == 0:
        return np.zeros((size,), dtype=np.int32)
    padded = np.empty((size,), dtype=np.int32)
    padded[: indices.shape[0]] = indices
    padded[indices.shape[0] :] = indices[0]
    return padded


def _pad_step_inputs(
    state_indices: np.ndarray,
    actions: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_indices = np.asarray(state_indices, dtype=np.int32)
    actions = np.asarray(actions, dtype=np.int32)
    if state_indices.shape[0] >= size:
        return state_indices, actions
    padded_state_indices = _pad_indices(state_indices, size)
    padded_actions = np.empty((size,), dtype=np.int32)
    padded_actions[: actions.shape[0]] = actions
    padded_actions[actions.shape[0] :] = actions[0] if actions.shape[0] else 0
    return padded_state_indices, padded_actions


def _backup_active_mask(
    root_ids: np.ndarray,
    path_len: np.ndarray,
    done: np.ndarray,
    num_simulations: int,
) -> np.ndarray:
    candidate = (path_len > 0) & (done[root_ids] < num_simulations)
    candidate_ix = np.flatnonzero(candidate)
    if candidate_ix.size <= 1:
        return candidate
    candidate_roots = root_ids[candidate_ix]
    if np.all(candidate_roots[1:] != candidate_roots[:-1]):
        return candidate

    active = np.zeros_like(candidate, dtype=bool)
    used: dict[int, int] = {}
    for ix in candidate_ix:
        root_id = int(root_ids[ix])
        already_used = used.get(root_id, 0)
        if int(done[root_id]) + already_used >= num_simulations:
            continue
        active[ix] = True
        used[root_id] = already_used + 1
    return active


def _align_rows(values: np.ndarray, flip_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    flip_mask = np.asarray(flip_mask, dtype=bool)
    if values.shape[0] == 0 or not np.any(flip_mask):
        return values.copy()
    aligned = values.copy()
    aligned[flip_mask] = aligned[flip_mask, ::-1]
    return aligned


def _commit_action(
    rng: np.random.Generator,
    config: SearchConfig,
    policy: np.ndarray,
    alpha: np.ndarray,
    legal: np.ndarray,
) -> int:
    if not np.any(legal):
        return 0
    if config.final_action_mode == "posterior_sample":
        probs = np.where(legal, policy, 0.0)
        total = float(np.sum(probs))
        if total <= 0.0:
            probs = legal.astype(np.float64)
            total = float(np.sum(probs))
        return int(rng.choice(alpha.shape[0], p=probs / total))
    if config.final_action_mode == "posterior_argmax":
        return int(np.argmax(np.where(legal, policy, -np.inf)))
    return greedy_q_action(alpha, legal)


def _commit_actions_batch(
    rng: np.random.Generator,
    config: SearchConfig,
    policies: np.ndarray,
    alpha: np.ndarray,
    legal: np.ndarray,
) -> np.ndarray:
    if config.final_action_mode == "posterior_sample":
        actions = np.zeros((policies.shape[0],), dtype=np.int32)
        for ix in range(policies.shape[0]):
            actions[ix] = _commit_action(rng, config, policies[ix], alpha[ix], legal[ix])
        return actions
    if config.final_action_mode == "posterior_argmax":
        return np.argmax(np.where(legal, policies, -np.inf), axis=-1).astype(np.int32)
    scores = outcome_mean(alpha)[..., -1] - outcome_mean(alpha)[..., 0]
    return np.argmax(np.where(legal, scores, -np.inf), axis=-1).astype(np.int32)


def _posterior_best_policy_target_batch_np(
    rng: np.random.Generator,
    alpha: np.ndarray,
    legal_action_mask: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    alpha = _positive(alpha)
    legal = np.asarray(legal_action_mask, dtype=bool)
    target = np.zeros(alpha.shape[:2], dtype=np.float32)
    has_legal = np.any(legal, axis=-1)
    if not np.any(has_legal):
        return target

    samples = int(num_samples)
    batch_size, num_actions, num_outcomes = alpha.shape
    max_gamma_values = 8_000_000
    chunk_size = max(1, min(batch_size, max_gamma_values // max(1, samples * num_actions * num_outcomes)))
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        chunk_alpha = alpha[start:end]
        chunk_legal = legal[start:end]
        gamma_shape = np.broadcast_to(
            chunk_alpha[None, :, :, :],
            (samples, end - start, num_actions, num_outcomes),
        )
        gamma = rng.gamma(gamma_shape, 1.0).astype(np.float32, copy=False)
        denom = np.maximum(np.sum(gamma, axis=-1, keepdims=True), np.float32(1e-12))
        phi = gamma / denom
        utility = phi[..., -1] - phi[..., 0]
        utility = np.where(chunk_legal[None, :, :], utility, -np.inf)
        best = np.argmax(utility, axis=-1).astype(np.int32)
        chunk_target = np.zeros((end - start, num_actions), dtype=np.float32)
        rows = np.tile(np.arange(end - start, dtype=np.int32), samples)
        np.add.at(chunk_target, (rows, best.reshape((-1,))), 1.0)
        chunk_target /= float(samples)
        chunk_target[~np.any(chunk_legal, axis=-1)] = 0.0
        target[start:end] = chunk_target
    return target


def _batched_step(env: Any):
    cache_key = id(env)
    cached = _STEP_CACHE.get(cache_key)
    if cached is not None:
        env_ref, fn = cached
        if env_ref is None or env_ref() is env:
            return fn
    fn = jax.jit(jax.vmap(env.step))
    try:
        env_ref = weakref.ref(env)
    except TypeError:
        env_ref = None
    _STEP_CACHE[cache_key] = (env_ref, fn)
    return fn


def _batched_step_info(env: Any):
    cache_key = id(env)
    cached = _STEP_INFO_CACHE.get(cache_key)
    if cached is not None:
        env_ref, fn = cached
        if env_ref is None or env_ref() is env:
            return fn

    def step_info(state: Any, action: jax.Array):
        next_state = env.step(state, action)
        return (
            next_state,
            canonical_state_key(next_state),
            next_state.terminated,
            next_state.current_player,
            next_state.rewards,
            next_state.legal_action_mask,
        )

    fn = jax.jit(jax.vmap(step_info))
    try:
        env_ref = weakref.ref(env)
    except TypeError:
        env_ref = None
    _STEP_INFO_CACHE[cache_key] = (env_ref, fn)
    return fn


def _batched_key_fn(state: Any) -> jax.Array:
    key = type(state)
    fn = _KEY_CACHE.get(key)
    if fn is None:
        fn = jax.jit(jax.vmap(canonical_state_key))
        _KEY_CACHE[key] = fn
    return fn(state)


def _stack_states(states: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _select_state_rows(state: Any, rows: np.ndarray) -> Any:
    rows = np.asarray(rows, dtype=np.int32)
    batch_size = _state_batch_size(state)
    if rows.shape == (batch_size,) and (rows.size == 0 or np.array_equal(rows, np.arange(batch_size, dtype=np.int32))):
        return state
    rows_jax = jnp.asarray(rows, dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda x: x[rows_jax], state)


def _state_batch_size(state: Any) -> int:
    leaves = jax.tree_util.tree_leaves(state)
    if not leaves:
        return 0
    return int(leaves[0].shape[0])


def _key_id(key: np.ndarray) -> bytes:
    return np.ascontiguousarray(key, dtype=np.uint32).reshape((4,)).tobytes()


def _positive(alpha: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(alpha, dtype=np.float32), np.float32(1e-6))
