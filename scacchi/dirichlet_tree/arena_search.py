from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import time
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from .native import (
    NO_DISTANCE,
    NO_OUTCOME,
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    categorical_proxy_np,
    native_policy_target_np,
)
from .selection import (
    thompson_select_jax,
    thompson_select_np,
)
from .state_hash import canonical_state_key
from .types import (
    LeafEvaluator,
    SearchConfig,
    SearchDiagnostics,
    SearchResult,
    TreeTrainingData,
    outcome_mean,
    outcome_utility,
    terminal_outcome_from_reward,
)

if TYPE_CHECKING:
    from ..train import SearchConfig as RuntimeSearchConfig


UNKNOWN = -1
STATUS_UNEXPANDED = np.uint8(0)
STATUS_INFLIGHT = np.uint8(1)
STATUS_EXPANDING = np.uint8(2)
STATUS_EXPANDED = np.uint8(3)
STATUS_TERMINAL = np.uint8(4)

VALUE_CACHE_DIRTY = np.uint8(0)
VALUE_CACHE_CLEAN = np.uint8(1)
VALUE_CACHE_UPDATING = np.uint8(2)

_STEP_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, Any]] = {}
_STEP_INFO_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, Any]] = {}
_KEY_CACHE: dict[type, Any] = {}


@jax.jit
def _take_rows_jit(array: jax.Array, rows: jax.Array) -> jax.Array:
    return array[rows]

TIMING_BUCKETS = (
    "root_hashing",
    "root_eval",
    "node_packing",
    "thompson_selection",
    "pgx_step_hash",
    "device_get",
    "child_classification",
    "leaf_observation_gather",
    "nn_eval",
    "expansion",
    "backup",
    "posterior_target_generation",
    "dirty_flush_store",
)


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
    eval_indices: np.ndarray | None = None

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
        observation_shape: tuple[int, ...] | None = None,
        observation_dtype: np.dtype | type | None = None,
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
        self.node_parent_node = np.full((self.max_nodes,), UNKNOWN, dtype=np.int32)
        self.node_parent_action = np.full((self.max_nodes,), UNKNOWN, dtype=np.int32)
        self.node_depth = np.zeros((self.max_nodes,), dtype=np.int16)
        self.node_value_alpha = np.ones((self.max_nodes, self.num_outcomes), dtype=np.float32)
        self.node_summary_alpha = np.ones((self.max_nodes, self.num_outcomes), dtype=np.float32)
        self.node_value_cache_C = np.ones((self.max_nodes, self.num_outcomes), dtype=np.float32)
        self.node_downstream_eval_count = np.zeros((self.max_nodes,), dtype=np.uint32)
        self.node_value_cache_status = np.full(
            (self.max_nodes,),
            VALUE_CACHE_CLEAN,
            dtype=np.uint8,
        )
        self.node_value_cache_version = np.zeros((self.max_nodes,), dtype=np.uint32)
        self.node_edge_epoch = np.zeros((self.max_nodes,), dtype=np.uint32)
        self.node_terminal_outcome = np.full((self.max_nodes,), -1, dtype=np.int8)
        self.node_cat_outcome = np.full((self.max_nodes,), int(NO_OUTCOME), dtype=np.int8)
        self.node_cat_distance = np.full((self.max_nodes,), int(NO_DISTANCE), dtype=np.int32)
        self.node_cat_action = np.full((self.max_nodes,), UNKNOWN, dtype=np.int32)
        self.node_observation = None
        if observation_shape is not None:
            dtype = np.dtype(np.float32 if observation_dtype is None else observation_dtype)
            self.node_observation = np.zeros(
                (self.max_nodes, *tuple(int(x) for x in observation_shape)),
                dtype=dtype,
            )

        self.edge_parent_node = np.full((self.max_edges,), UNKNOWN, dtype=np.int32)
        self.edge_action = np.zeros((self.max_edges,), dtype=np.int32)
        self.edge_child_node = np.full((self.max_edges,), UNKNOWN, dtype=np.int32)
        self.edge_child_key = np.zeros((self.max_edges, 4), dtype=np.uint32)
        self.edge_base_alpha = np.ones((self.max_edges, self.num_outcomes), dtype=np.float32)
        self.edge_B = np.ones((self.max_edges, self.num_outcomes), dtype=np.float32)
        self.edge_has_post = np.zeros((self.max_edges,), dtype=bool)
        self.edge_eval_count_R = np.zeros((self.max_edges,), dtype=np.uint32)
        self.edge_version = np.zeros((self.max_edges,), dtype=np.uint32)
        self.edge_child_cache_version = np.full((self.max_edges,), -1, dtype=np.int64)
        self.edge_E = np.zeros((self.max_edges, self.num_outcomes), dtype=np.float32)
        self.edge_post_alpha = np.ones((self.max_edges, self.num_outcomes), dtype=np.float32)
        self.edge_logit = np.zeros((self.max_edges,), dtype=np.float32)
        self.edge_visits = np.zeros((self.max_edges,), dtype=np.uint32)
        self.edge_cat_outcome = np.full((self.max_edges,), int(NO_OUTCOME), dtype=np.int8)
        self.edge_cat_distance = np.full((self.max_edges,), int(NO_DISTANCE), dtype=np.int32)

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
        observation: np.ndarray | None = None,
        parent_node_id: int = UNKNOWN,
        parent_action: int = UNKNOWN,
        depth: int = 0,
    ) -> int:
        self.ensure_key_index()
        key_id = _key_id(key)
        existing = self.key_to_node.get(key_id)
        if existing is not None and self.node_status[existing] != STATUS_INFLIGHT:
            self._set_node_observation(existing, observation)
            return existing

        legal_actions = np.flatnonzero(np.asarray(legal_action_mask, dtype=bool)).astype(np.int32)
        node_id = existing if existing is not None else self._alloc_node(key_id)
        first_edge = self.num_edges
        edge_count = int(legal_actions.shape[0])
        if first_edge + edge_count > self.max_edges:
            raise MemoryError("posterior arena edge capacity exceeded")
        self.sorted_key_view = None
        self.node_status[node_id] = STATUS_EXPANDING
        self.node_key[node_id] = np.asarray(key, dtype=np.uint32)
        self.node_current_player[node_id] = np.int8(current_player)
        self.node_first_edge[node_id] = np.int32(first_edge)
        self.node_num_edges[node_id] = np.int16(edge_count)
        self.node_parent_node[node_id] = np.int32(parent_node_id)
        self.node_parent_action[node_id] = np.int32(parent_action)
        self.node_depth[node_id] = np.int16(depth)
        self.node_value_alpha[node_id] = _positive(value_alpha)
        self.node_value_cache_C[node_id] = self.node_value_alpha[node_id]
        self.node_downstream_eval_count[node_id] = np.uint32(0)
        self.node_value_cache_status[node_id] = VALUE_CACHE_CLEAN
        self.node_value_cache_version[node_id] += np.uint32(1)
        self.node_edge_epoch[node_id] = np.uint32(0)
        self.node_terminal_outcome[node_id] = np.int8(-1)
        self.node_cat_outcome[node_id] = np.int8(NO_OUTCOME)
        self.node_cat_distance[node_id] = np.int32(NO_DISTANCE)
        self.node_cat_action[node_id] = np.int32(UNKNOWN)
        self._set_node_observation(node_id, observation)

        end_edge = first_edge + edge_count
        self.edge_parent_node[first_edge:end_edge] = node_id
        self.edge_action[first_edge:end_edge] = legal_actions
        self.edge_child_node[first_edge:end_edge] = UNKNOWN
        self.edge_child_key[first_edge:end_edge] = 0
        sparse_q = _positive(np.asarray(q_alpha, dtype=np.float32)[legal_actions])
        self.edge_base_alpha[first_edge:end_edge] = sparse_q
        self.edge_B[first_edge:end_edge] = sparse_q
        self.edge_has_post[first_edge:end_edge] = False
        self.edge_eval_count_R[first_edge:end_edge] = 0
        self.edge_version[first_edge:end_edge] = 0
        self.edge_child_cache_version[first_edge:end_edge] = -1
        self.edge_E[first_edge:end_edge] = 0.0
        self.edge_post_alpha[first_edge:end_edge] = sparse_q
        self.edge_logit[first_edge:end_edge] = np.asarray(policy_logits, dtype=np.float32)[legal_actions]
        self.edge_visits[first_edge:end_edge] = 0
        self.edge_cat_outcome[first_edge:end_edge] = np.int8(NO_OUTCOME)
        self.edge_cat_distance[first_edge:end_edge] = np.int32(NO_DISTANCE)
        self.num_edges = end_edge
        self.node_summary_alpha[node_id] = self.node_value_cache_C[node_id]
        self.node_status[node_id] = STATUS_EXPANDED
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
        observations: np.ndarray | None = None,
        assume_unique_new: bool = False,
        allow_grouped: bool = True,
        parent_node_ids: np.ndarray | None = None,
        parent_actions: np.ndarray | None = None,
        depths: np.ndarray | None = None,
    ) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.uint32).reshape((-1, 4))
        count = int(keys.shape[0])
        if count == 0:
            return np.zeros((0,), dtype=np.int32)

        legal = np.asarray(legal_action_mask, dtype=bool)
        if not allow_grouped:
            return np.asarray(
                [
                    self.add_expanded_node(
                        key=keys[ix],
                        current_player=int(current_players[ix]),
                        legal_action_mask=legal[ix],
                        value_alpha=value_alpha[ix],
                        policy_logits=policy_logits[ix],
                        q_alpha=q_alpha[ix],
                        observation=None if observations is None else observations[ix],
                        parent_node_id=(
                            UNKNOWN if parent_node_ids is None else int(parent_node_ids[ix])
                        ),
                        parent_action=UNKNOWN if parent_actions is None else int(parent_actions[ix]),
                        depth=0 if depths is None else int(depths[ix]),
                    )
                    for ix in range(count)
                ],
                dtype=np.int32,
            )

        uniform_legal_count = False
        legal_counts = np.zeros((count,), dtype=np.int32)
        if legal.ndim == 2:
            legal_counts = np.sum(legal, axis=1, dtype=np.int32)
            uniform_legal_count = bool(np.all(legal_counts == legal_counts[0]))
        if (
            legal.ndim == 2
            and not uniform_legal_count
            and assume_unique_new
        ):
            node_ids = np.empty((count,), dtype=np.int32)
            for legal_count in np.unique(legal_counts):
                group = np.flatnonzero(legal_counts == legal_count).astype(np.int32)
                node_ids[group] = self.add_expanded_nodes_batch(
                    keys=keys[group],
                    current_players=np.asarray(current_players)[group],
                    legal_action_mask=legal[group],
                    value_alpha=np.asarray(value_alpha)[group],
                    policy_logits=np.asarray(policy_logits)[group],
                    q_alpha=np.asarray(q_alpha)[group],
                    observations=None if observations is None else np.asarray(observations)[group],
                    assume_unique_new=True,
                    allow_grouped=True,
                    parent_node_ids=(
                        None if parent_node_ids is None else np.asarray(parent_node_ids)[group]
                    ),
                    parent_actions=(
                        None if parent_actions is None else np.asarray(parent_actions)[group]
                    ),
                    depths=None if depths is None else np.asarray(depths)[group],
                )
            return node_ids
        if (
            legal.ndim != 2
            or not uniform_legal_count
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
                        observation=None if observations is None else observations[ix],
                        parent_node_id=(
                            UNKNOWN if parent_node_ids is None else int(parent_node_ids[ix])
                        ),
                        parent_action=UNKNOWN if parent_actions is None else int(parent_actions[ix]),
                        depth=0 if depths is None else int(depths[ix]),
                    )
                    for ix in range(count)
                ],
                dtype=np.int32,
            )

        edges_per_node = int(legal_counts[0])
        legal_actions_by_node = np.nonzero(legal)[1].astype(np.int32).reshape((count, edges_per_node))
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
        self.node_status[node_ids] = STATUS_EXPANDING
        self.node_key[node_ids] = keys
        self.node_current_player[node_ids] = np.asarray(current_players, dtype=np.int8)
        self.node_first_edge[node_ids] = edge_start + np.arange(count, dtype=np.int32) * edges_per_node
        self.node_num_edges[node_ids] = np.int16(edges_per_node)
        self.node_parent_node[node_ids] = (
            UNKNOWN
            if parent_node_ids is None
            else np.asarray(parent_node_ids, dtype=np.int32)
        )
        self.node_parent_action[node_ids] = (
            UNKNOWN
            if parent_actions is None
            else np.asarray(parent_actions, dtype=np.int32)
        )
        self.node_depth[node_ids] = 0 if depths is None else np.asarray(depths, dtype=np.int16)
        self.node_value_alpha[node_ids] = _positive(value_alpha)
        self.node_value_cache_C[node_ids] = self.node_value_alpha[node_ids]
        self.node_downstream_eval_count[node_ids] = np.uint32(0)
        self.node_value_cache_status[node_ids] = VALUE_CACHE_CLEAN
        self.node_value_cache_version[node_ids] += np.uint32(1)
        self.node_edge_epoch[node_ids] = np.uint32(0)
        self.node_terminal_outcome[node_ids] = np.int8(-1)
        self.node_cat_outcome[node_ids] = np.int8(NO_OUTCOME)
        self.node_cat_distance[node_ids] = np.int32(NO_DISTANCE)
        self.node_cat_action[node_ids] = np.int32(UNKNOWN)
        self._set_node_observations(node_ids, observations)
        if self._key_index_complete and count < 4096:
            self.key_to_node.update((keys[ix].tobytes(), int(node_ids[ix])) for ix in range(count))
        else:
            self._key_index_complete = False

        if edges_per_node == 0:
            self.node_summary_alpha[node_ids] = self.node_value_cache_C[node_ids]
            self.node_status[node_ids] = STATUS_EXPANDED
            return node_ids

        edge_end = edge_start + edge_count
        edge_ids = np.arange(edge_start, edge_end, dtype=np.int32)
        self.edge_parent_node[edge_start:edge_end] = np.repeat(node_ids, edges_per_node)
        self.edge_action[edge_start:edge_end] = legal_actions_by_node.reshape((edge_count,))
        self.edge_child_node[edge_start:edge_end] = UNKNOWN
        self.edge_child_key[edge_start:edge_end] = 0
        sparse_q = _positive(
            np.take_along_axis(
                np.asarray(q_alpha, dtype=np.float32),
                legal_actions_by_node[..., None],
                axis=1,
            )
        )
        self.edge_base_alpha[edge_start:edge_end] = sparse_q.reshape((edge_count, self.num_outcomes))
        self.edge_B[edge_start:edge_end] = self.edge_base_alpha[edge_start:edge_end]
        self.edge_has_post[edge_start:edge_end] = False
        self.edge_eval_count_R[edge_start:edge_end] = 0
        self.edge_version[edge_start:edge_end] = 0
        self.edge_child_cache_version[edge_start:edge_end] = -1
        self.edge_E[edge_start:edge_end] = 0.0
        self.edge_post_alpha[edge_start:edge_end] = self.edge_base_alpha[edge_start:edge_end]
        sparse_logits = np.take_along_axis(
            np.asarray(policy_logits, dtype=np.float32),
            legal_actions_by_node,
            axis=1,
        )
        self.edge_logit[edge_start:edge_end] = sparse_logits.reshape((edge_count,))
        self.edge_visits[edge_start:edge_end] = 0
        self.edge_cat_outcome[edge_start:edge_end] = np.int8(NO_OUTCOME)
        self.edge_cat_distance[edge_start:edge_end] = np.int32(NO_DISTANCE)
        self.num_edges = edge_end
        self.node_summary_alpha[node_ids] = self.node_value_cache_C[node_ids]
        self.node_status[node_ids] = STATUS_EXPANDED
        return node_ids

    def add_terminal_node(
        self,
        *,
        key: np.ndarray,
        current_player: int,
        terminal_outcome: int,
        observation: np.ndarray | None = None,
        parent_node_id: int = UNKNOWN,
        parent_action: int = UNKNOWN,
        depth: int = 0,
    ) -> int:
        self.ensure_key_index()
        key_id = _key_id(key)
        existing = self.key_to_node.get(key_id)
        if existing is not None and self.node_status[existing] != STATUS_INFLIGHT:
            self._set_node_observation(existing, observation)
            return existing
        node_id = existing if existing is not None else self._alloc_node(key_id)
        self.sorted_key_view = None
        self.node_status[node_id] = STATUS_TERMINAL
        self.node_key[node_id] = np.asarray(key, dtype=np.uint32)
        self.node_current_player[node_id] = np.int8(current_player)
        self.node_first_edge[node_id] = np.int32(self.num_edges)
        self.node_num_edges[node_id] = np.int16(0)
        self.node_parent_node[node_id] = np.int32(parent_node_id)
        self.node_parent_action[node_id] = np.int32(parent_action)
        self.node_depth[node_id] = np.int16(depth)
        self.node_value_alpha[node_id] = 1e-6
        self.node_value_alpha[node_id, int(terminal_outcome)] = 1.0
        self.node_value_cache_C[node_id] = self.node_value_alpha[node_id]
        self.node_downstream_eval_count[node_id] = np.uint32(0)
        self.node_value_cache_status[node_id] = VALUE_CACHE_CLEAN
        self.node_value_cache_version[node_id] += np.uint32(1)
        self.node_edge_epoch[node_id] = np.uint32(0)
        self.node_summary_alpha[node_id] = self.node_value_cache_C[node_id]
        self.node_terminal_outcome[node_id] = np.int8(terminal_outcome)
        self.node_cat_outcome[node_id] = np.int8(terminal_outcome)
        self.node_cat_distance[node_id] = np.int32(0)
        self.node_cat_action[node_id] = np.int32(UNKNOWN)
        self._set_node_observation(node_id, observation)
        return node_id

    def add_inflight_node(
        self,
        *,
        key: np.ndarray,
        current_player: int,
        parent_node_id: int,
        parent_action: int,
        depth: int,
        observation: np.ndarray | None = None,
    ) -> int:
        self.ensure_key_index()
        key_id = _key_id(key)
        existing = self.key_to_node.get(key_id)
        if existing is not None:
            self._set_node_observation(existing, observation)
            return existing
        node_id = self._alloc_node(key_id)
        self.sorted_key_view = None
        self.node_status[node_id] = STATUS_INFLIGHT
        self.node_key[node_id] = np.asarray(key, dtype=np.uint32)
        self.node_current_player[node_id] = np.int8(current_player)
        self.node_first_edge[node_id] = np.int32(self.num_edges)
        self.node_num_edges[node_id] = np.int16(0)
        self.node_parent_node[node_id] = np.int32(parent_node_id)
        self.node_parent_action[node_id] = np.int32(parent_action)
        self.node_depth[node_id] = np.int16(depth)
        self.node_value_alpha[node_id] = 1.0
        self.node_value_cache_C[node_id] = self.node_value_alpha[node_id]
        self.node_downstream_eval_count[node_id] = np.uint32(0)
        self.node_value_cache_status[node_id] = VALUE_CACHE_CLEAN
        self.node_value_cache_version[node_id] = np.uint32(0)
        self.node_edge_epoch[node_id] = np.uint32(0)
        self.node_summary_alpha[node_id] = self.node_value_cache_C[node_id]
        self.node_terminal_outcome[node_id] = np.int8(-1)
        self.node_cat_outcome[node_id] = np.int8(NO_OUTCOME)
        self.node_cat_distance[node_id] = np.int32(NO_DISTANCE)
        self.node_cat_action[node_id] = np.int32(UNKNOWN)
        self._set_node_observation(node_id, observation)
        return node_id

    def _set_node_observation(self, node_id: int, observation: np.ndarray | None) -> None:
        if self.node_observation is None or observation is None:
            return
        self.node_observation[int(node_id)] = np.asarray(
            observation,
            dtype=self.node_observation.dtype,
        )

    def _set_node_observations(
        self,
        node_ids: np.ndarray,
        observations: np.ndarray | None,
    ) -> None:
        if self.node_observation is None or observations is None:
            return
        self.node_observation[np.asarray(node_ids, dtype=np.int32)] = np.asarray(
            observations,
            dtype=self.node_observation.dtype,
        )

    def recompute_summary(self, node_id: int) -> None:
        start = int(self.node_first_edge[node_id])
        count = int(self.node_num_edges[node_id])
        if count <= 0:
            self.node_summary_alpha[node_id] = self.node_value_alpha[node_id]
            return
        edge_slice = slice(start, start + count)
        post = self.edge_post_alpha[edge_slice]
        q_mean = (post[:, -1] - post[:, 0]) / np.sum(post, axis=-1)
        if np.all(q_mean == q_mean[0]):
            self.node_summary_alpha[node_id] = _positive(np.mean(post, axis=0))
            return
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
        post = self.edge_post_alpha[edge_ids]
        q_mean = (post[..., -1] - post[..., 0]) / np.sum(post, axis=-1)
        if np.all(q_mean == q_mean[:, :1]):
            self.node_summary_alpha[node_ids] = _positive(np.mean(post, axis=1))
            return
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
            if existing is not None:
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
        self.tree_sample_capacity: int | None = None
        self._completed_path_depths: np.ndarray | None = None
        self._completed_path_counts: np.ndarray | None = None
        self.timing_enabled = False
        self.timing_sync = False
        self.timing = {bucket: 0.0 for bucket in TIMING_BUCKETS}

    def enable_timing(self, *, sync: bool = True) -> None:
        self.timing_enabled = True
        self.timing_sync = bool(sync)
        self.timing = {bucket: 0.0 for bucket in TIMING_BUCKETS}

    @contextmanager
    def _timed(self, bucket: str):
        if not self.timing_enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timing[bucket] = self.timing.get(bucket, 0.0) + (time.perf_counter() - start)

    def _block_if_timing(self, value: Any) -> None:
        if self.timing_enabled and self.timing_sync:
            _block_until_ready(value)

    def _record_completed_path_depth(self, root_id: int, depth: int) -> None:
        if self._completed_path_depths is None or self._completed_path_counts is None:
            return
        root_ix = int(root_id)
        if root_ix < 0 or root_ix >= self._completed_path_counts.shape[0]:
            return
        write_ix = int(self._completed_path_counts[root_ix])
        if write_ix < self._completed_path_depths.shape[1]:
            self._completed_path_depths[root_ix, write_ix] = np.int16(max(0, int(depth)))
        self._completed_path_counts[root_ix] += 1

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
        with self._timed("root_hashing"):
            root_key_result = _batched_key_fn(root_state_batch)
            self._block_if_timing(root_key_result)
        with self._timed("device_get"):
            root_keys_all = np.asarray(jax.device_get(root_key_result), dtype=np.uint32)
        unique_root_indices, root_inverse, sorted_root_key_view = _unique_key_indices_inverse_sorted(root_keys_all)
        if unique_root_indices.shape[0] != original_num_roots:
            root_state_batch = _select_state_rows(root_state_batch, unique_root_indices)
            root_keys = root_keys_all[unique_root_indices]
        else:
            root_keys = root_keys_all

        root_observations = root_state_batch.observation
        with self._timed("root_eval"):
            root_eval_result = leaf_evaluator(root_observations)
            self._block_if_timing(root_eval_result)
        with self._timed("device_get"):
            root_logits, root_value_alpha, root_q_alpha = jax.device_get(root_eval_result)
        num_roots = int(root_logits.shape[0])
        num_actions = int(root_logits.shape[-1])
        num_outcomes = int(root_value_alpha.shape[-1])
        self._completed_path_depths = np.full(
            (num_roots, max(1, int(config.num_simulations))),
            -1,
            dtype=np.int16,
        )
        self._completed_path_counts = np.zeros((num_roots,), dtype=np.int32)
        with self._timed("device_get"):
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
        store_tree_nodes = bool(config.train_tree_nodes)
        root_observations_np = (
            np.asarray(jax.device_get(root_observations))
            if store_tree_nodes
            else None
        )

        if max_nodes is None:
            capacity_roots = original_num_roots if store_tree_nodes else num_roots
            max_nodes = _default_max_nodes(capacity_roots, config)
        if max_edges is None:
            max_edges = max_nodes * max(1, num_actions)
        self.tree_sample_capacity = (
            _tree_sample_capacity(original_num_roots, max_nodes, config)
            if store_tree_nodes
            else None
        )
        arena = PosteriorArena(
            max_nodes=max_nodes,
            max_edges=max_edges,
            num_actions=num_actions,
            num_outcomes=num_outcomes,
            observation_shape=None if root_observations_np is None else root_observations_np.shape[1:],
            observation_dtype=None if root_observations_np is None else root_observations_np.dtype,
        )
        if not np.any(root_term):
            with self._timed("node_packing"):
                root_node_ids = arena.add_expanded_nodes_batch(
                    keys=root_keys,
                    current_players=root_players,
                    legal_action_mask=root_legal,
                    value_alpha=root_value_alpha,
                    policy_logits=root_logits,
                    q_alpha=root_q_alpha,
                    observations=root_observations_np,
                    assume_unique_new=True,
                    allow_grouped=config.grouped_expansion,
                )
        else:
            root_node_ids = np.empty((num_roots,), dtype=np.int32)
            with self._timed("node_packing"):
                for ix in range(num_roots):
                    if root_term[ix]:
                        outcome = terminal_outcome_from_reward(float(root_rewards[ix, root_players[ix]]), num_outcomes)
                        root_node_ids[ix] = arena.add_terminal_node(
                            key=root_keys[ix],
                            current_player=int(root_players[ix]),
                            terminal_outcome=outcome,
                            observation=None if root_observations_np is None else root_observations_np[ix],
                        )
                    else:
                        root_node_ids[ix] = arena.add_expanded_node(
                            key=root_keys[ix],
                            current_player=int(root_players[ix]),
                            legal_action_mask=root_legal[ix],
                            value_alpha=root_value_alpha[ix],
                            policy_logits=root_logits[ix],
                            q_alpha=root_q_alpha[ix],
                            observation=None if root_observations_np is None else root_observations_np[ix],
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
        with self._timed("posterior_target_generation"):
            self._repair_dirty_frontier(config)
            dense_result = self._finish_search_dense(config)
            if dense_result is not None:
                dense_result = self._attach_tree_data(dense_result, config)
                self._block_if_timing(dense_result)
                return dense_result

            actions = np.zeros((self.root_node_ids.shape[0],), dtype=np.int32)
            policies = np.zeros((self.root_node_ids.shape[0], arena.num_actions), dtype=np.float32)
            beta_q = np.zeros((self.root_node_ids.shape[0], arena.num_actions, arena.num_outcomes), dtype=np.float32)
            beta_v = np.zeros((self.root_node_ids.shape[0], arena.num_outcomes), dtype=np.float32)
            q_weight = np.zeros((self.root_node_ids.shape[0], arena.num_actions), dtype=np.float32)
            alpha_root = np.zeros_like(beta_q)
            q_kind = np.zeros((self.root_node_ids.shape[0], arena.num_actions), dtype=np.int8)
            q_target_weight = np.zeros((self.root_node_ids.shape[0], arena.num_actions), dtype=np.float32)
            q_outcome = np.full((self.root_node_ids.shape[0], arena.num_actions), int(NO_OUTCOME), dtype=np.int8)
            q_distance = np.full((self.root_node_ids.shape[0], arena.num_actions), int(NO_DISTANCE), dtype=np.int32)
            v_kind = np.full((self.root_node_ids.shape[0],), int(TARGET_DIRICHLET), dtype=np.int8)
            v_target_weight = np.ones((self.root_node_ids.shape[0],), dtype=np.float32)
            v_outcome = np.full((self.root_node_ids.shape[0],), int(NO_OUTCOME), dtype=np.int8)
            v_distance = np.full((self.root_node_ids.shape[0],), int(NO_DISTANCE), dtype=np.int32)
            for out_ix, node_id in enumerate(self.root_node_ids):
                start = int(arena.node_first_edge[node_id])
                count = int(arena.node_num_edges[node_id])
                legal = np.zeros((arena.num_actions,), dtype=bool)
                for edge_id in range(start, start + count):
                    action = int(arena.edge_action[edge_id])
                    legal[action] = True
                    alpha_root[out_ix, action] = arena.edge_post_alpha[edge_id]
                    beta_q[out_ix, action] = arena.edge_post_alpha[edge_id]
                    q_kind[out_ix, action] = (
                        int(TARGET_CATEGORICAL)
                        if int(arena.edge_cat_outcome[edge_id]) != int(NO_OUTCOME)
                        else int(TARGET_DIRICHLET)
                    )
                    q_target_weight[out_ix, action] = 1.0
                    q_outcome[out_ix, action] = arena.edge_cat_outcome[edge_id]
                    q_distance[out_ix, action] = arena.edge_cat_distance[edge_id]
                if int(arena.node_cat_outcome[node_id]) != int(NO_OUTCOME):
                    action = int(arena.node_cat_action[node_id])
                    if action < 0 or action >= arena.num_actions:
                        action = _first_legal_action(legal)
                    if 0 <= action < arena.num_actions and bool(legal[action]):
                        policies[out_ix, action] = 1.0
                    beta_v[out_ix] = categorical_proxy_np(
                        int(arena.node_cat_outcome[node_id]),
                        arena.num_outcomes,
                        epsilon=1e-6,
                    )
                    v_kind[out_ix] = int(TARGET_CATEGORICAL)
                    v_outcome[out_ix] = arena.node_cat_outcome[node_id]
                    v_distance[out_ix] = arena.node_cat_distance[node_id]
                    actions[out_ix] = action
                    q_weight[out_ix] = policies[out_ix]
                    continue
                policies[out_ix] = native_policy_target_np(
                    self.rng,
                    alpha_root[out_ix],
                    legal,
                    q_kind[out_ix],
                    q_outcome[out_ix],
                    config.policy_mc_samples,
                )
                q_weight[out_ix] = policies[out_ix]
                beta_v[out_ix] = arena.node_value_cache_C[node_id]
                actions[out_ix] = _commit_action(self.rng, config, policies[out_ix], alpha_root[out_ix], legal)

            result = SearchResult(
                action=jnp.asarray(actions, dtype=jnp.int32),
                action_weights=jnp.asarray(policies, dtype=jnp.float32),
                beta_Q_target=jnp.asarray(beta_q, dtype=jnp.float32),
                beta_V_target=jnp.asarray(beta_v, dtype=jnp.float32),
                q_loss_weight=jnp.asarray(q_weight, dtype=jnp.float32),
                alpha_root=jnp.asarray(alpha_root, dtype=jnp.float32),
                search_loss_mask=jnp.asarray(np.sum(policies, axis=-1) > 0.0),
                diagnostics=self._build_search_diagnostics(config, policies, alpha_root),
                q_target_kind=jnp.asarray(q_kind, dtype=jnp.int8),
                q_target_weight=jnp.asarray(q_target_weight, dtype=jnp.float32),
                q_target_outcome=jnp.asarray(q_outcome, dtype=jnp.int8),
                q_target_distance=jnp.asarray(q_distance, dtype=jnp.int32),
                v_target_kind=jnp.asarray(v_kind, dtype=jnp.int8),
                v_target_weight=jnp.asarray(v_target_weight, dtype=jnp.float32),
                v_target_outcome=jnp.asarray(v_outcome, dtype=jnp.int8),
                v_target_distance=jnp.asarray(v_distance, dtype=jnp.int32),
            )
            result = self._attach_tree_data(result, config)
            self._block_if_timing(result)
            return result

    def _attach_tree_data(self, result: SearchResult, config: SearchConfig) -> SearchResult:
        if not config.train_tree_nodes:
            return result
        return result._replace(tree_data=self._build_tree_training_data(config))

    def _finish_search_dense(self, config: SearchConfig) -> SearchResult | None:
        if self.arena is None or self.root_node_ids is None:
            return None
        arena = self.arena
        root_ids = self.root_node_ids.astype(np.int32)
        if root_ids.size == 0:
            return None
        if np.any(arena.node_cat_outcome[root_ids] != int(NO_OUTCOME)):
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

        legal = np.zeros((root_ids.shape[0], arena.num_actions), dtype=bool)
        legal[row_ids, action_ids] = True
        q_kind = np.zeros((root_ids.shape[0], arena.num_actions), dtype=np.int8)
        q_target_weight = np.zeros((root_ids.shape[0], arena.num_actions), dtype=np.float32)
        q_outcome = np.full((root_ids.shape[0], arena.num_actions), int(NO_OUTCOME), dtype=np.int8)
        q_distance = np.full((root_ids.shape[0], arena.num_actions), int(NO_DISTANCE), dtype=np.int32)
        sparse_cat_outcome = arena.edge_cat_outcome[edge_ids]
        q_kind[row_ids, action_ids] = np.where(
            sparse_cat_outcome != int(NO_OUTCOME),
            int(TARGET_CATEGORICAL),
            int(TARGET_DIRICHLET),
        ).astype(np.int8)
        q_target_weight[row_ids, action_ids] = 1.0
        q_outcome[row_ids, action_ids] = sparse_cat_outcome
        q_distance[row_ids, action_ids] = arena.edge_cat_distance[edge_ids]

        if np.any(sparse_cat_outcome != int(NO_OUTCOME)):
            policies = np.stack(
                [
                    native_policy_target_np(
                        self.rng,
                        alpha_root[ix],
                        legal[ix],
                        q_kind[ix],
                        q_outcome[ix],
                        config.policy_mc_samples,
                    )
                    for ix in range(root_ids.shape[0])
                ],
                axis=0,
            )
        else:
            policies = _posterior_best_policy_target_batch_np(
                self.rng,
                alpha_root,
                legal,
                config.policy_mc_samples,
            )
        q_weight = policies.copy()
        beta_v = arena.node_value_cache_C[root_ids]
        actions = _commit_actions_batch(self.rng, config, policies, alpha_root, legal)
        v_kind = np.full((root_ids.shape[0],), int(TARGET_DIRICHLET), dtype=np.int8)
        v_target_weight = np.ones((root_ids.shape[0],), dtype=np.float32)
        v_outcome = np.full((root_ids.shape[0],), int(NO_OUTCOME), dtype=np.int8)
        v_distance = np.full((root_ids.shape[0],), int(NO_DISTANCE), dtype=np.int32)
        return SearchResult(
            action=jnp.asarray(actions, dtype=jnp.int32),
            action_weights=jnp.asarray(policies, dtype=jnp.float32),
            beta_Q_target=jnp.asarray(beta_q, dtype=jnp.float32),
            beta_V_target=jnp.asarray(beta_v, dtype=jnp.float32),
            q_loss_weight=jnp.asarray(q_weight, dtype=jnp.float32),
            alpha_root=jnp.asarray(alpha_root, dtype=jnp.float32),
            search_loss_mask=jnp.asarray(np.sum(policies, axis=-1) > 0.0),
            diagnostics=self._build_search_diagnostics(config, policies, alpha_root),
            q_target_kind=jnp.asarray(q_kind, dtype=jnp.int8),
            q_target_weight=jnp.asarray(q_target_weight, dtype=jnp.float32),
            q_target_outcome=jnp.asarray(q_outcome, dtype=jnp.int8),
            q_target_distance=jnp.asarray(q_distance, dtype=jnp.int32),
            v_target_kind=jnp.asarray(v_kind, dtype=jnp.int8),
            v_target_weight=jnp.asarray(v_target_weight, dtype=jnp.float32),
            v_target_outcome=jnp.asarray(v_outcome, dtype=jnp.int8),
            v_target_distance=jnp.asarray(v_distance, dtype=jnp.int32),
        )

    def _build_search_diagnostics(
        self,
        config: SearchConfig,
        policies: np.ndarray,
        alpha_root: np.ndarray,
    ) -> SearchDiagnostics:
        if self.arena is None or self.root_node_ids is None:
            raise ValueError("search has not been initialized")
        arena = self.arena
        root_ids = self.root_node_ids.astype(np.int32)
        depth_mean, depth_p50, depth_p90, depth_max = self._completed_path_depth_stats(
            root_ids.shape[0]
        )
        expanded_nodes, terminal_fraction = _arena_root_descendant_stats(arena, root_ids)
        n_down = np.zeros((root_ids.shape[0],), dtype=np.float32)
        root_q_concentration = np.zeros((root_ids.shape[0],), dtype=np.float32)
        for ix, node_id in enumerate(root_ids):
            start = int(arena.node_first_edge[node_id])
            count = int(arena.node_num_edges[node_id])
            if count <= 0:
                continue
            edge_ids = np.arange(start, start + count, dtype=np.int32)
            n_down[ix] = float(np.sum(arena.edge_eval_count_R[edge_ids], dtype=np.uint64))
            legal_alpha = alpha_root[ix, arena.edge_action[edge_ids].astype(np.int32)]
            root_q_concentration[ix] = float(np.mean(np.sum(legal_alpha, axis=-1)))
        gamma = n_down / (float(config.state_posterior_kappa_n) + n_down)
        policy = np.asarray(policies, dtype=np.float32)
        entropy = -np.sum(
            np.where(policy > 0.0, policy * np.log(np.maximum(policy, 1e-12)), 0.0),
            axis=-1,
        ).astype(np.float32)
        return SearchDiagnostics(
            path_depth_mean=jnp.asarray(depth_mean, dtype=jnp.float32),
            path_depth_p50=jnp.asarray(depth_p50, dtype=jnp.float32),
            path_depth_p90=jnp.asarray(depth_p90, dtype=jnp.float32),
            path_depth_max=jnp.asarray(depth_max, dtype=jnp.float32),
            expanded_nodes=jnp.asarray(expanded_nodes, dtype=jnp.float32),
            terminal_fraction=jnp.asarray(terminal_fraction, dtype=jnp.float32),
            root_policy_entropy=jnp.asarray(entropy, dtype=jnp.float32),
            root_gamma=jnp.asarray(gamma, dtype=jnp.float32),
            root_downstream_eval_count=jnp.asarray(n_down, dtype=jnp.float32),
            root_q_concentration=jnp.asarray(root_q_concentration, dtype=jnp.float32),
        )

    def _completed_path_depth_stats(
        self,
        num_roots: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean = np.zeros((num_roots,), dtype=np.float32)
        p50 = np.zeros((num_roots,), dtype=np.float32)
        p90 = np.zeros((num_roots,), dtype=np.float32)
        max_depth = np.zeros((num_roots,), dtype=np.float32)
        if self._completed_path_depths is None or self._completed_path_counts is None:
            return mean, p50, p90, max_depth
        for ix in range(num_roots):
            count = min(
                int(self._completed_path_counts[ix]),
                int(self._completed_path_depths.shape[1]),
            )
            if count <= 0:
                continue
            values = self._completed_path_depths[ix, :count].astype(np.float32)
            values = values[values >= 0]
            if values.size == 0:
                continue
            mean[ix] = float(np.mean(values))
            p50[ix] = float(np.percentile(values, 50))
            p90[ix] = float(np.percentile(values, 90))
            max_depth[ix] = float(np.max(values))
        return mean, p50, p90, max_depth

    def _build_tree_training_data(self, config: SearchConfig) -> TreeTrainingData:
        if self.arena is None or self.root_node_ids is None:
            raise ValueError("search has not been initialized")
        arena = self.arena
        if arena.node_observation is None:
            raise ValueError("tree-node training requires arena observation storage")

        capacity = self.tree_sample_capacity
        if capacity is None:
            capacity = _tree_sample_capacity(self.root_node_ids.shape[0], arena.max_nodes, config)
        obs = np.zeros((capacity, *arena.node_observation.shape[1:]), dtype=arena.node_observation.dtype)
        action_weights = np.zeros((capacity, arena.num_actions), dtype=np.float32)
        played_action = np.zeros((capacity,), dtype=np.int32)
        legal_mask = np.zeros((capacity, arena.num_actions), dtype=bool)
        beta_q = np.zeros((capacity, arena.num_actions, arena.num_outcomes), dtype=np.float32)
        beta_v = np.ones((capacity, arena.num_outcomes), dtype=np.float32)
        q_weight = np.zeros((capacity, arena.num_actions), dtype=np.float32)
        q_kind = np.zeros((capacity, arena.num_actions), dtype=np.int8)
        q_target_weight = np.zeros((capacity, arena.num_actions), dtype=np.float32)
        q_outcome = np.full((capacity, arena.num_actions), int(NO_OUTCOME), dtype=np.int8)
        q_distance = np.full((capacity, arena.num_actions), int(NO_DISTANCE), dtype=np.int32)
        v_kind = np.full((capacity,), int(TARGET_DIRICHLET), dtype=np.int8)
        v_target_weight = np.ones((capacity,), dtype=np.float32)
        v_outcome = np.full((capacity,), int(NO_OUTCOME), dtype=np.int8)
        v_distance = np.full((capacity,), int(NO_DISTANCE), dtype=np.int32)
        value_tgt = np.zeros((capacity,), dtype=np.float32)
        policy_loss_mask = np.zeros((capacity,), dtype=bool)
        value_loss_mask = np.zeros((capacity,), dtype=bool)
        search_loss_mask = np.zeros((capacity,), dtype=bool)
        outcome_mask = np.zeros((capacity,), dtype=bool)

        root_nodes = set(int(x) for x in np.asarray(self.root_node_ids, dtype=np.int32))
        include_roots = bool(config.train_tree_include_root)
        expanded_rows: list[int] = []
        expanded_node_ids: list[int] = []
        min_q_evidence = float(config.train_tree_min_q_evidence)
        row = 0
        for node_id in range(int(arena.num_nodes)):
            if row >= capacity:
                break
            if node_id in root_nodes and not include_roots:
                continue
            status = arena.node_status[node_id]
            if status == STATUS_EXPANDED:
                if arena.node_value_cache_status[node_id] != VALUE_CACHE_CLEAN:
                    continue
                start = int(arena.node_first_edge[node_id])
                count = int(arena.node_num_edges[node_id])
                if count <= 0:
                    continue
                edge_ids = np.arange(start, start + count, dtype=np.int32)
                if not np.any(arena.edge_has_post[edge_ids]):
                    continue
                actions = arena.edge_action[edge_ids].astype(np.int32)
                edge_counts = arena.edge_eval_count_R[edge_ids].astype(np.float32)
                if float(np.sum(edge_counts)) <= min_q_evidence:
                    continue
                obs[row] = arena.node_observation[node_id]
                legal_mask[row, actions] = True
                beta_q[row, actions] = arena.edge_post_alpha[edge_ids]
                beta_v[row] = arena.node_value_cache_C[node_id]
                q_kind[row, actions] = np.where(
                    arena.edge_cat_outcome[edge_ids] != int(NO_OUTCOME),
                    int(TARGET_CATEGORICAL),
                    int(TARGET_DIRICHLET),
                ).astype(np.int8)
                q_target_weight[row, actions] = 1.0
                q_outcome[row, actions] = arena.edge_cat_outcome[edge_ids]
                q_distance[row, actions] = arena.edge_cat_distance[edge_ids]
                if int(arena.node_cat_outcome[node_id]) != int(NO_OUTCOME):
                    v_kind[row] = int(TARGET_CATEGORICAL)
                    v_outcome[row] = arena.node_cat_outcome[node_id]
                    v_distance[row] = arena.node_cat_distance[node_id]
                policy_loss_mask[row] = True
                value_loss_mask[row] = True
                search_loss_mask[row] = True
                expanded_rows.append(row)
                expanded_node_ids.append(node_id)
                row += 1
                continue

        if expanded_rows:
            row_ix = np.asarray(expanded_rows, dtype=np.int32)
            node_ix = np.asarray(expanded_node_ids, dtype=np.int32)
            if np.any(q_kind[row_ix] == int(TARGET_CATEGORICAL)):
                policies = np.stack(
                    [
                        native_policy_target_np(
                            self.rng,
                            beta_q[row],
                            legal_mask[row],
                            q_kind[row],
                            q_outcome[row],
                            config.policy_mc_samples,
                        )
                        for row in row_ix
                    ],
                    axis=0,
                )
            else:
                policies = _posterior_best_policy_target_batch_np(
                    self.rng,
                    beta_q[row_ix],
                    legal_mask[row_ix],
                    config.policy_mc_samples,
                )
            categorical_nodes = arena.node_cat_outcome[node_ix] != int(NO_OUTCOME)
            for local_ix, is_categorical in enumerate(categorical_nodes):
                if not bool(is_categorical):
                    continue
                row = int(row_ix[local_ix])
                action = int(arena.node_cat_action[int(node_ix[local_ix])])
                if action < 0 or action >= arena.num_actions:
                    action = _first_legal_action(legal_mask[row])
                policies[local_ix] = 0.0
                if 0 <= action < arena.num_actions and bool(legal_mask[row, action]):
                    policies[local_ix, action] = 1.0
            action_weights[row_ix] = policies
            q_weight[row_ix] = policies
            beta_v[row_ix] = arena.node_value_cache_C[node_ix]
            value_tgt[row_ix] = outcome_utility(outcome_mean(beta_v[row_ix]))
            played_action[row_ix] = _commit_actions_batch(
                self.rng,
                config,
                policies,
                beta_q[row_ix],
                legal_mask[row_ix],
            )

        return TreeTrainingData(
            obs=jnp.asarray(obs),
            action_weights=jnp.asarray(action_weights, dtype=jnp.float32),
            played_action=jnp.asarray(played_action, dtype=jnp.int32),
            legal_action_mask=jnp.asarray(legal_mask),
            beta_Q_target=jnp.asarray(beta_q, dtype=jnp.float32),
            beta_V_target=jnp.asarray(beta_v, dtype=jnp.float32),
            q_loss_weight=jnp.asarray(q_weight, dtype=jnp.float32),
            value_tgt=jnp.asarray(value_tgt, dtype=jnp.float32),
            policy_loss_mask=jnp.asarray(policy_loss_mask),
            value_loss_mask=jnp.asarray(value_loss_mask),
            search_loss_mask=jnp.asarray(search_loss_mask),
            outcome_mask=jnp.asarray(outcome_mask),
            q_target_kind=jnp.asarray(q_kind, dtype=jnp.int8),
            q_target_weight=jnp.asarray(q_target_weight, dtype=jnp.float32),
            q_target_outcome=jnp.asarray(q_outcome, dtype=jnp.int8),
            q_target_distance=jnp.asarray(q_distance, dtype=jnp.int32),
            v_target_kind=jnp.asarray(v_kind, dtype=jnp.int8),
            v_target_weight=jnp.asarray(v_target_weight, dtype=jnp.float32),
            v_target_outcome=jnp.asarray(v_outcome, dtype=jnp.int8),
            v_target_distance=jnp.asarray(v_distance, dtype=jnp.int32),
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
        fixed_lane_root_ids: np.ndarray | None = None
        fixed_state_batch: Any | None = None
        if config.stable_lane_batch:
            fixed_lane_root_ids = np.repeat(
                np.arange(num_roots, dtype=np.int32),
                max(1, int(config.num_lanes_per_root)),
            ).astype(np.int32)
            fixed_state_batch = _select_state_rows(root_state_batch, fixed_lane_root_ids)
        while np.any(done < config.num_simulations):
            attempts += 1
            if attempts > max_attempts:
                unfinished = np.flatnonzero(done < config.num_simulations).tolist()
                raise RuntimeError(f"arena wavefront posterior search stalled for roots {unfinished}")

            if config.stable_lane_batch:
                assert fixed_lane_root_ids is not None and fixed_state_batch is not None
                lane_root_ids = fixed_lane_root_ids
                state_batch = fixed_state_batch
            else:
                unfinished = np.flatnonzero(done < config.num_simulations).astype(np.int32)
                if unfinished.size == 0:
                    break
                lane_root_ids = np.repeat(unfinished, max(1, int(config.num_lanes_per_root))).astype(np.int32)
                state_batch = _select_state_rows(root_state_batch, lane_root_ids)
            current_node_ids = root_node_ids[lane_root_ids].copy()
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
            with self._timed("backup"):
                for row in terminal_rows:
                    if done[lane_root_ids[row]] >= config.num_simulations:
                        continue
                    node_id = int(current_node_ids[row])
                    if path_len[row] > 0:
                        final_edge = int(path_edges[row, int(path_len[row]) - 1])
                        self._publish_categorical_edge_from_child(
                            final_edge,
                            node_id,
                            config,
                            increment_eval_count=True,
                        )
                        final_parent = int(path_nodes[row, int(path_len[row]) - 1])
                        self._propagate_categorical(final_parent, config)
                    self._record_completed_path_depth(int(lane_root_ids[row]), int(path_len[row]))
                    done[lane_root_ids[row]] += 1

                categorical_rows = active_rows[
                    (status == STATUS_EXPANDED)
                    & (arena.node_cat_outcome[node_ids] != int(NO_OUTCOME))
                ]
                for row in categorical_rows:
                    if done[lane_root_ids[row]] >= config.num_simulations:
                        continue
                    self._propagate_categorical(int(current_node_ids[row]), config)
                    self._record_completed_path_depth(int(lane_root_ids[row]), int(path_len[row]))
                    done[lane_root_ids[row]] += 1

            selectable_mask = (
                (status == STATUS_EXPANDED)
                & (arena.node_num_edges[node_ids] > 0)
                & (arena.node_cat_outcome[node_ids] == int(NO_OUTCOME))
            )
            selectable_rows = active_rows[selectable_mask]
            selectable_state_pos = active_state_pos[selectable_mask]
            if selectable_rows.size == 0:
                break

            with self._timed("node_packing"):
                select_nodes = current_node_ids[selectable_rows]
                for node_id in np.unique(select_nodes):
                    self._try_categorize_node(int(node_id), config)
                selectable_now = arena.node_cat_outcome[select_nodes] == int(NO_OUTCOME)
                if not np.any(selectable_now):
                    continue
                if not np.all(selectable_now):
                    selectable_rows = selectable_rows[selectable_now]
                    selectable_state_pos = selectable_state_pos[selectable_now]
                    select_nodes = select_nodes[selectable_now]
                first_edges = arena.node_first_edge[select_nodes].astype(np.int32)
                edge_counts = arena.node_num_edges[select_nodes].astype(np.int32)
                max_edges = int(np.max(edge_counts))
                offsets = np.arange(max_edges, dtype=np.int32)
                edge_ids = first_edges[:, None] + offsets[None, :]
                mask = offsets[None, :] < edge_counts[:, None]
                safe_edge_ids = np.where(mask, edge_ids, 0)
                child_ids = arena.edge_child_node[safe_edge_ids]
                safe_child_ids = np.where(child_ids == UNKNOWN, 0, child_ids)
                child_status = arena.node_status[safe_child_ids]
                blocked = (child_ids != UNKNOWN) & (
                    (child_status == STATUS_INFLIGHT) | (child_status == STATUS_EXPANDING)
                )
                mask = mask & ~blocked
                mask = mask & (arena.edge_cat_outcome[safe_edge_ids] == int(NO_OUTCOME))
                has_available = np.any(mask, axis=1)
                if not np.any(has_available):
                    break
                if not np.all(has_available):
                    selectable_rows = selectable_rows[has_available]
                    selectable_state_pos = selectable_state_pos[has_available]
                    select_nodes = select_nodes[has_available]
                    first_edges = first_edges[has_available]
                    edge_counts = edge_counts[has_available]
                    edge_ids = edge_ids[has_available]
                    safe_edge_ids = safe_edge_ids[has_available]
                    mask = mask[has_available]
                alpha = np.where(
                    mask[..., None],
                    arena.edge_post_alpha[safe_edge_ids],
                    np.ones((selectable_rows.shape[0], max_edges, arena.num_outcomes), dtype=np.float32),
                )
                actions_padded = np.where(mask, arena.edge_action[safe_edge_ids], 0).astype(np.int32)
            if 0 < selectable_rows.shape[0] < config.np_select_below:
                with self._timed("thompson_selection"):
                    actions, selected_pos = thompson_select_np(
                        self.rng,
                        alpha,
                        actions_padded,
                        mask,
                    )
            else:
                select_alpha = alpha
                select_actions_padded = actions_padded
                select_mask = mask
                if config.pad_jax_select:
                    select_alpha, select_actions_padded, select_mask = _pad_jax_selection_inputs(
                        alpha,
                        actions_padded,
                        mask,
                        target_rows=int(lane_root_ids.shape[0]),
                        target_edges=arena.num_actions,
                    )
                self.jax_key, select_key = jax.random.split(self.jax_key)
                with self._timed("thompson_selection"):
                    selected_actions_result = thompson_select_jax(
                        select_key,
                        jnp.asarray(select_alpha),
                        jnp.asarray(select_actions_padded),
                        jnp.asarray(select_mask),
                    )
                    self._block_if_timing(selected_actions_result)
                with self._timed("device_get"):
                    selected_actions = np.asarray(
                        jax.device_get(selected_actions_result),
                        dtype=np.int32,
                    )
                actions = selected_actions[: selectable_rows.shape[0]]
                selected_pos = _selected_positions(actions_padded, actions, mask)
            selected_edge_ids = edge_ids[np.arange(edge_ids.shape[0]), selected_pos].astype(np.int32)

            depth_ix = path_len[selectable_rows].astype(np.int32)
            path_nodes[selectable_rows, depth_ix] = select_nodes
            path_edges[selectable_rows, depth_ix] = selected_edge_ids
            path_len[selectable_rows] += 1

            lane_count = int(lane_root_ids.shape[0])
            if config.lane_indexed_step:
                selected_state_batch = state_batch
                padded_actions = np.zeros((lane_count,), dtype=np.int32)
                padded_actions[selectable_rows] = actions
                output_indices = selectable_rows.astype(np.int32, copy=False)
            elif _is_prefix_indices(selectable_state_pos):
                selected_state_batch = state_batch
                padded_actions = _pad_actions(actions, lane_count)
                output_indices = np.arange(selectable_rows.shape[0], dtype=np.int32)
            else:
                padded_state_pos, padded_actions = _pad_step_inputs(
                    selectable_state_pos,
                    actions,
                    lane_count,
                )
                selected_state_batch = _select_state_rows(state_batch, padded_state_pos)
                output_indices = np.arange(selectable_rows.shape[0], dtype=np.int32)

            with self._timed("pgx_step_hash"):
                step_info_result = _batched_step_info(self.env)(
                    selected_state_batch,
                    jnp.asarray(padded_actions),
                )
                self._block_if_timing(step_info_result)
            with self._timed("device_get"):
                key_words, terminated, players, terminal_rewards, legal = jax.device_get(
                    step_info_result[1:]
                )
            key_words = np.asarray(key_words, dtype=np.uint32)
            terminated = np.asarray(terminated, dtype=bool)
            players = np.asarray(players, dtype=np.int32)
            terminal_rewards = np.asarray(terminal_rewards, dtype=np.float32)
            legal = np.asarray(legal, dtype=bool)
            next_state_batch = step_info_result[0]

            selected_keys = key_words[output_indices]
            selected_terminated = terminated[output_indices]
            if self._can_fast_path_fresh_leaves(selected_edge_ids, selected_keys, selected_terminated):
                arena.edge_child_key[selected_edge_ids] = selected_keys
                arena.edge_child_node[selected_edge_ids] = UNKNOWN
                with self._timed("leaf_observation_gather"):
                    pending_batch = _make_pending_batch(
                        next_state_batch=next_state_batch,
                        key_words=key_words,
                        players=players,
                        legal=legal,
                        lane_root_ids=lane_root_ids,
                        path_nodes=path_nodes,
                        path_edges=path_edges,
                        path_len=path_len,
                        missing_rows=selectable_rows.astype(np.int32, copy=False),
                        missing_state_indices=output_indices.astype(np.int32, copy=False),
                        recycle_duplicates=True,
                        observation_pad_size=(
                            config.eval_batch_size
                            if config.pad_eval_batches and config.pad_pending_observation_gather
                            else None
                        ),
                    )
                    self._block_if_timing(pending_batch.observations)
                    pending_batches.append(pending_batch)
                break

            with self._timed("child_classification"):
                arena.ensure_key_index()
                next_rows: list[int] = []
                next_state_indices: list[int] = []
                missing_rows: list[int] = []
                missing_state_indices: list[int] = []
                base_update_edges: list[int] = []
                base_update_children: list[int] = []
                terminal_backup_rows: list[int] = []
                terminal_backup_children: list[int] = []
                for ix, row in enumerate(selectable_rows):
                    data_ix = int(output_indices[ix])
                    edge_id = int(selected_edge_ids[ix])
                    child_node_id = arena.key_to_node.get(key_words[data_ix].tobytes(), UNKNOWN)
                    arena.edge_child_key[edge_id] = key_words[data_ix]
                    arena.edge_child_node[edge_id] = child_node_id

                    if terminated[data_ix]:
                        if child_node_id == UNKNOWN:
                            outcome = terminal_outcome_from_reward(float(terminal_rewards[data_ix]), arena.num_outcomes)
                            child_node_id = arena.add_terminal_node(
                                key=key_words[data_ix],
                                current_player=int(players[data_ix]),
                                terminal_outcome=outcome,
                                parent_node_id=int(arena.edge_parent_node[edge_id]),
                                parent_action=int(arena.edge_action[edge_id]),
                                depth=int(arena.node_depth[int(arena.edge_parent_node[edge_id])]) + 1,
                                observation=(
                                    None
                                    if arena.node_observation is None
                                    else np.asarray(
                                        jax.device_get(next_state_batch.observation[data_ix])
                                    )
                                ),
                            )
                            arena.edge_child_node[edge_id] = child_node_id
                        terminal_backup_rows.append(int(row))
                        terminal_backup_children.append(child_node_id)
                        continue

                    if child_node_id == UNKNOWN:
                        child_node_id = arena.add_inflight_node(
                            key=key_words[data_ix],
                            current_player=int(players[data_ix]),
                            parent_node_id=int(arena.edge_parent_node[edge_id]),
                            parent_action=int(arena.edge_action[edge_id]),
                            depth=int(arena.node_depth[int(arena.edge_parent_node[edge_id])]) + 1,
                        )
                        arena.edge_child_node[edge_id] = child_node_id
                        missing_rows.append(int(row))
                        missing_state_indices.append(data_ix)
                        continue

                    child_status = arena.node_status[child_node_id]
                    if child_status in (STATUS_INFLIGHT, STATUS_EXPANDING):
                        continue
                    if child_status == STATUS_TERMINAL:
                        terminal_backup_rows.append(int(row))
                        terminal_backup_children.append(child_node_id)
                    elif int(arena.node_cat_outcome[child_node_id]) != int(NO_OUTCOME):
                        terminal_backup_rows.append(int(row))
                        terminal_backup_children.append(child_node_id)
                    else:
                        base_update_edges.append(edge_id)
                        base_update_children.append(child_node_id)
                        next_rows.append(int(row))
                        next_state_indices.append(data_ix)
                        current_node_ids[row] = child_node_id

            if base_update_edges:
                with self._timed("backup"):
                    base_edge_ids = np.asarray(base_update_edges, dtype=np.int32)
                    base_child_ids = np.asarray(base_update_children, dtype=np.int32)
                    self._update_edge_base_from_children(base_edge_ids, base_child_ids)
                    self._refresh_edges_from_children(base_edge_ids, base_child_ids, config)

            with self._timed("backup"):
                for row, child_node_id in zip(terminal_backup_rows, terminal_backup_children, strict=True):
                    if done[lane_root_ids[row]] >= config.num_simulations:
                        continue
                    final_edge = int(path_edges[row, int(path_len[row]) - 1])
                    self._publish_categorical_edge_from_child(
                        final_edge,
                        int(child_node_id),
                        config,
                        increment_eval_count=True,
                    )
                    final_parent = int(path_nodes[row, int(path_len[row]) - 1])
                    self._propagate_categorical(final_parent, config)
                    self._record_completed_path_depth(int(lane_root_ids[row]), int(path_len[row]))
                    done[lane_root_ids[row]] += 1

            if missing_rows:
                with self._timed("leaf_observation_gather"):
                    pending_batch = _make_pending_batch(
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
                        recycle_duplicates=True,
                        observation_pad_size=(
                            config.eval_batch_size
                            if config.pad_eval_batches and config.pad_pending_observation_gather
                            else None
                        ),
                    )
                    self._block_if_timing(pending_batch.observations)
                    pending_batches.append(pending_batch)

            if not next_rows:
                break
            active_rows = np.asarray(next_rows, dtype=np.int32)
            state_batch = next_state_batch
            active_state_pos = active_rows if config.lane_indexed_step else np.asarray(next_state_indices, dtype=np.int32)

        return pending_batches

    def _can_fast_path_fresh_leaves(
        self,
        selected_edge_ids: np.ndarray,
        key_words: np.ndarray,
        terminated: np.ndarray,
    ) -> bool:
        return False
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
        with self._timed("leaf_observation_gather"):
            pending_batch = _merge_pending_batches(pending)
            self._block_if_timing(pending_batch.observations)
        if (
            pending_batch.eval_indices is not None
            and config.pad_eval_batches
            and int(pending_batch.observations.shape[0]) == int(config.eval_batch_size)
            and pending_batch.size <= int(config.eval_batch_size)
        ):
            with self._timed("nn_eval"):
                eval_result = leaf_evaluator(pending_batch.observations)
                self._block_if_timing(eval_result)
            with self._timed("device_get"):
                logits, value_alpha, q_alpha = jax.device_get(eval_result)
            eval_indices = pending_batch.eval_indices
            logits = logits[eval_indices]
            value_alpha = value_alpha[eval_indices]
            q_alpha = q_alpha[eval_indices]
            with self._timed("expansion"):
                node_observations = (
                    None
                    if arena.node_observation is None
                    else np.asarray(jax.device_get(_pending_selected_observations(pending_batch)))
                )
                parent_edges = pending_batch.path_edges[
                    np.arange(pending_batch.size, dtype=np.int32),
                    pending_batch.path_len.astype(np.int32) - 1,
                ].astype(np.int32)
                parent_nodes = pending_batch.path_nodes[
                    np.arange(pending_batch.size, dtype=np.int32),
                    pending_batch.path_len.astype(np.int32) - 1,
                ].astype(np.int32)
                child_node_ids = arena.add_expanded_nodes_batch(
                    keys=pending_batch.key_words,
                    current_players=pending_batch.players,
                    legal_action_mask=pending_batch.legal,
                    value_alpha=value_alpha,
                    policy_logits=logits,
                    q_alpha=q_alpha,
                    observations=node_observations,
                    assume_unique_new=False,
                    allow_grouped=config.grouped_expansion,
                    parent_node_ids=parent_nodes,
                    parent_actions=arena.edge_action[parent_edges],
                    depths=arena.node_depth[parent_nodes].astype(np.int32) + 1,
                )
            with self._timed("backup"):
                self._backup_pending_rows(
                    child_node_ids=child_node_ids,
                    root_ids=pending_batch.root_ids,
                    path_nodes=pending_batch.path_nodes,
                    path_edges=pending_batch.path_edges,
                    path_len=pending_batch.path_len,
                    done=done,
                    policy_mc_samples=config.policy_mc_samples,
                    num_simulations=config.num_simulations,
                    config=config,
                )
            return
        for start in range(0, pending_batch.size, config.eval_batch_size):
            end = min(start + config.eval_batch_size, pending_batch.size)
            real_count = end - start
            with self._timed("leaf_observation_gather"):
                observations = _eval_observation_batch(
                    pending_batch.observations,
                    start,
                    end,
                    target_size=config.eval_batch_size,
                    pad=config.pad_eval_batches,
                )
                self._block_if_timing(observations)
            with self._timed("nn_eval"):
                eval_result = leaf_evaluator(observations)
                self._block_if_timing(eval_result)
            logits, value_alpha, q_alpha = eval_result
            if int(logits.shape[0]) != real_count:
                logits = logits[:real_count]
                value_alpha = value_alpha[:real_count]
                q_alpha = q_alpha[:real_count]
            with self._timed("device_get"):
                logits, value_alpha, q_alpha = jax.device_get((logits, value_alpha, q_alpha))
            request_slice = slice(start, end)
            with self._timed("expansion"):
                node_observations = (
                    None
                    if arena.node_observation is None
                    else np.asarray(
                        jax.device_get(_pending_selected_observations(pending_batch)[request_slice])
                    )
                )
                slice_path_len = pending_batch.path_len[request_slice].astype(np.int32)
                slice_rows = np.arange(real_count, dtype=np.int32)
                parent_edges = pending_batch.path_edges[request_slice][
                    slice_rows,
                    slice_path_len - 1,
                ].astype(np.int32)
                parent_nodes = pending_batch.path_nodes[request_slice][
                    slice_rows,
                    slice_path_len - 1,
                ].astype(np.int32)
                child_node_ids = arena.add_expanded_nodes_batch(
                    keys=pending_batch.key_words[request_slice],
                    current_players=pending_batch.players[request_slice],
                    legal_action_mask=pending_batch.legal[request_slice],
                    value_alpha=value_alpha,
                    policy_logits=logits,
                    q_alpha=q_alpha,
                    observations=node_observations,
                    assume_unique_new=False,
                    allow_grouped=config.grouped_expansion,
                    parent_node_ids=parent_nodes,
                    parent_actions=arena.edge_action[parent_edges],
                    depths=arena.node_depth[parent_nodes].astype(np.int32) + 1,
                )
            with self._timed("backup"):
                self._backup_pending_rows(
                    child_node_ids=child_node_ids,
                    root_ids=pending_batch.root_ids[request_slice],
                    path_nodes=pending_batch.path_nodes[request_slice],
                    path_edges=pending_batch.path_edges[request_slice],
                    path_len=pending_batch.path_len[request_slice],
                    done=done,
                    policy_mc_samples=config.policy_mc_samples,
                    num_simulations=config.num_simulations,
                    config=config,
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
        policy_mc_samples: int,
        num_simulations: int,
        config: SearchConfig,
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
        fresh = arena.edge_cat_outcome[final_edges] == int(NO_OUTCOME)
        if not np.any(fresh):
            np.add.at(done, active_root_ids, 1)
            return
        if not np.all(fresh):
            stale_root_ids = active_root_ids[~fresh]
            np.add.at(done, stale_root_ids, 1)
            active_child_ids = active_child_ids[fresh]
            active_root_ids = active_root_ids[fresh]
            active_path_len = active_path_len[fresh]
            final_edges = final_edges[fresh]
            final_parents = final_parents[fresh]

        categorical_children = arena.node_cat_outcome[active_child_ids] != int(NO_OUTCOME)
        if np.any(categorical_children):
            cat_edges = final_edges[categorical_children]
            cat_children = active_child_ids[categorical_children]
            for edge_id, child_id, parent_id in zip(
                cat_edges,
                cat_children,
                final_parents[categorical_children],
                strict=True,
            ):
                self._publish_categorical_edge_from_child(
                    int(edge_id),
                    int(child_id),
                    config,
                    increment_eval_count=True,
                )
                self._propagate_categorical(int(parent_id), config)
            for root_id, depth in zip(
                active_root_ids[categorical_children],
                active_path_len[categorical_children],
                strict=True,
            ):
                self._record_completed_path_depth(int(root_id), int(depth))
            np.add.at(done, active_root_ids[categorical_children], 1)
            keep = ~categorical_children
            if not np.any(keep):
                return
            active_child_ids = active_child_ids[keep]
            active_root_ids = active_root_ids[keep]
            active_path_len = active_path_len[keep]
            final_edges = final_edges[keep]
            final_parents = final_parents[keep]

        arena.edge_child_node[final_edges] = active_child_ids
        leaf_value = _leaf_beta(arena.node_value_alpha[active_child_ids], config)
        parent_players = arena.node_current_player[final_parents]
        child_players = arena.node_current_player[active_child_ids]
        aligned_leaf = _align_rows(leaf_value, parent_players != child_players)
        self._publish_edge_posts(
            final_edges,
            aligned_leaf,
            np.ones((final_edges.shape[0],), dtype=np.uint32),
            increment_eval_count=True,
        )
        for root_id, depth in zip(active_root_ids, active_path_len, strict=True):
            self._record_completed_path_depth(int(root_id), int(depth))
        np.add.at(done, active_root_ids, 1)
        for parent_id in np.unique(final_parents):
            self._repair_path_to_root(
                int(parent_id),
                config,
                policy_mc_samples=policy_mc_samples,
            )

    def _state_search_posterior_batch(
        self,
        node_ids: np.ndarray,
        policy_mc_samples: int,
        state_posterior_kappa_n: float = 9.0,
    ) -> np.ndarray:
        assert self.arena is not None
        arena = self.arena
        node_ids = np.asarray(node_ids, dtype=np.int32)
        beta = np.zeros((node_ids.shape[0], arena.num_outcomes), dtype=np.float32)
        if node_ids.size == 0:
            return beta

        edge_counts = arena.node_num_edges[node_ids].astype(np.int32)
        no_edges = edge_counts <= 0
        if np.any(no_edges):
            beta[no_edges] = arena.node_value_cache_C[node_ids[no_edges]]

        for edge_count in np.unique(edge_counts[~no_edges]):
            positions = np.flatnonzero(edge_counts == edge_count).astype(np.int32)
            group_ids = node_ids[positions]
            first_edges = arena.node_first_edge[group_ids].astype(np.int32)
            edge_ids = first_edges[:, None] + np.arange(int(edge_count), dtype=np.int32)[None, :]
            alpha = arena.edge_post_alpha[edge_ids]
            legal = np.ones((group_ids.shape[0], int(edge_count)), dtype=bool)
            pi_search = np.stack(
                [
                    native_policy_target_np(
                        self.rng,
                        alpha[ix],
                        legal[ix],
                        _edge_target_kind_np(arena.edge_cat_outcome[edge_ids[ix]]),
                        arena.edge_cat_outcome[edge_ids[ix]],
                        int(policy_mc_samples),
                    )
                    for ix in range(group_ids.shape[0])
                ],
                axis=0,
            )
            e_v = np.sum(pi_search[..., None] * alpha, axis=1)
            n_down = np.sum(arena.edge_eval_count_R[edge_ids], axis=1).astype(np.float32)
            gamma = n_down / (float(state_posterior_kappa_n) + n_down)
            beta[positions] = (
                (1.0 - gamma[:, None]) * arena.node_value_alpha[group_ids]
                + gamma[:, None] * e_v
            )
        return _positive(beta)

    def _update_edge_base_from_child(self, edge_id: int, child_node_id: int) -> None:
        assert self.arena is not None
        arena = self.arena
        if (
            arena.node_status[int(child_node_id)] == STATUS_TERMINAL
            or int(arena.node_cat_outcome[int(child_node_id)]) != int(NO_OUTCOME)
        ):
            return
        if bool(arena.edge_has_post[int(edge_id)]):
            return
        parent_node_id = int(arena.edge_parent_node[edge_id])
        parent_player = int(arena.node_current_player[parent_node_id])
        child_player = int(arena.node_current_player[child_node_id])
        value = arena.node_value_alpha[child_node_id]
        arena.edge_base_alpha[edge_id] = value[::-1] if parent_player != child_player else value
        arena.edge_post_alpha[edge_id] = _positive(arena.edge_base_alpha[edge_id])

    def _update_edge_base_from_children(self, edge_ids: np.ndarray, child_node_ids: np.ndarray) -> None:
        assert self.arena is not None
        arena = self.arena
        edge_ids = np.asarray(edge_ids, dtype=np.int32)
        child_node_ids = np.asarray(child_node_ids, dtype=np.int32)
        keep = (
            (arena.node_status[child_node_ids] != STATUS_TERMINAL)
            & (arena.node_cat_outcome[child_node_ids] == int(NO_OUTCOME))
            & ~arena.edge_has_post[edge_ids]
        )
        if not np.any(keep):
            return
        edge_ids = edge_ids[keep]
        child_node_ids = child_node_ids[keep]
        parent_node_ids = arena.edge_parent_node[edge_ids].astype(np.int32)
        parent_players = arena.node_current_player[parent_node_ids]
        child_players = arena.node_current_player[child_node_ids]
        value = arena.node_value_alpha[child_node_ids]
        arena.edge_base_alpha[edge_ids] = _align_rows(value, parent_players != child_players)
        arena.edge_post_alpha[edge_ids] = _positive(arena.edge_base_alpha[edge_ids])

    def _publish_edge_posts(
        self,
        edge_ids: np.ndarray,
        beta: np.ndarray,
        eval_counts: np.ndarray,
        *,
        increment_eval_count: bool = False,
    ) -> None:
        assert self.arena is not None
        arena = self.arena
        edge_ids = np.asarray(edge_ids, dtype=np.int32).reshape((-1,))
        beta = _positive(np.asarray(beta, dtype=np.float32).reshape((edge_ids.shape[0], arena.num_outcomes)))
        eval_counts = np.asarray(eval_counts, dtype=np.uint32).reshape((edge_ids.shape[0],))
        arena.edge_B[edge_ids] = beta
        arena.edge_has_post[edge_ids] = True
        if increment_eval_count:
            np.add.at(arena.edge_eval_count_R, edge_ids, eval_counts)
        else:
            arena.edge_eval_count_R[edge_ids] = eval_counts
        arena.edge_version[edge_ids] += np.uint32(1)
        arena.edge_child_cache_version[edge_ids] = -1
        arena.edge_post_alpha[edge_ids] = beta
        arena.edge_E[edge_ids] = beta - arena.edge_base_alpha[edge_ids]
        arena.edge_visits[edge_ids] += np.uint32(1)
        parent_ids = np.unique(arena.edge_parent_node[edge_ids].astype(np.int32))
        self._mark_nodes_dirty(parent_ids)

    def _publish_categorical_edges(
        self,
        edge_ids: np.ndarray,
        outcomes: np.ndarray,
        distances: np.ndarray,
        *,
        increment_eval_count: bool = False,
    ) -> np.ndarray:
        assert self.arena is not None
        arena = self.arena
        edge_ids = np.asarray(edge_ids, dtype=np.int32).reshape((-1,))
        outcomes = np.asarray(outcomes, dtype=np.int8).reshape((edge_ids.shape[0],))
        distances = np.asarray(distances, dtype=np.int32).reshape((edge_ids.shape[0],))
        if edge_ids.size == 0:
            return np.zeros((0,), dtype=bool)

        previous_outcome = arena.edge_cat_outcome[edge_ids].copy()
        previous_distance = arena.edge_cat_distance[edge_ids].copy()
        changed = (previous_outcome != outcomes) | (previous_distance != distances)
        if not np.any(changed):
            return changed

        changed_edges = edge_ids[changed]
        changed_outcomes = outcomes[changed]
        changed_distances = distances[changed]
        arena.edge_cat_outcome[changed_edges] = changed_outcomes
        arena.edge_cat_distance[changed_edges] = changed_distances
        proxies = np.stack(
            [
                categorical_proxy_np(
                    int(outcome),
                    arena.num_outcomes,
                    epsilon=1e-6,
                )
                for outcome in changed_outcomes
            ],
            axis=0,
        )
        arena.edge_B[changed_edges] = proxies
        arena.edge_post_alpha[changed_edges] = proxies
        arena.edge_E[changed_edges] = proxies - arena.edge_base_alpha[changed_edges]
        arena.edge_has_post[changed_edges] = True
        first_publish = previous_outcome[changed] == int(NO_OUTCOME)
        if increment_eval_count:
            np.add.at(arena.edge_eval_count_R, changed_edges, np.ones_like(changed_edges, dtype=np.uint32))
        else:
            np.add.at(
                arena.edge_eval_count_R,
                changed_edges[first_publish],
                np.ones((int(np.sum(first_publish)),), dtype=np.uint32),
            )
        np.add.at(arena.edge_visits, changed_edges, np.ones_like(changed_edges, dtype=np.uint32))
        arena.edge_version[changed_edges] += np.uint32(1)
        arena.edge_child_cache_version[changed_edges] = -1
        parent_ids = np.unique(arena.edge_parent_node[changed_edges].astype(np.int32))
        self._mark_nodes_dirty(parent_ids)
        return changed

    def _publish_categorical_edge_from_child(
        self,
        edge_id: int,
        child_node_id: int,
        config: SearchConfig | None,
        *,
        increment_eval_count: bool = False,
    ) -> bool:
        assert self.arena is not None
        arena = self.arena
        edge_id = int(edge_id)
        child_id = int(child_node_id)
        if child_id == UNKNOWN:
            return False
        if int(arena.node_cat_outcome[child_id]) == int(NO_OUTCOME):
            if config is None:
                return False
            self._try_categorize_node(child_id, config)
        if int(arena.node_cat_outcome[child_id]) == int(NO_OUTCOME):
            return False
        parent_id = int(arena.edge_parent_node[edge_id])
        outcome = _align_outcome_index(
            int(arena.node_cat_outcome[child_id]),
            int(arena.node_current_player[child_id]),
            int(arena.node_current_player[parent_id]),
            arena.num_outcomes,
        )
        distance = int(arena.node_cat_distance[child_id]) + 1
        self._publish_categorical_edges(
            np.asarray([edge_id], dtype=np.int32),
            np.asarray([outcome], dtype=np.int8),
            np.asarray([distance], dtype=np.int32),
            increment_eval_count=increment_eval_count,
        )
        return True

    def _publish_categorical_node(
        self,
        node_id: int,
        outcome: int,
        distance: int,
        action: int,
    ) -> bool:
        assert self.arena is not None
        arena = self.arena
        node_id = int(node_id)
        changed = (
            int(arena.node_cat_outcome[node_id]) != int(outcome)
            or int(arena.node_cat_distance[node_id]) != int(distance)
            or int(arena.node_cat_action[node_id]) != int(action)
        )
        arena.node_cat_outcome[node_id] = np.int8(outcome)
        arena.node_cat_distance[node_id] = np.int32(distance)
        arena.node_cat_action[node_id] = np.int32(action)
        arena.node_value_cache_C[node_id] = categorical_proxy_np(
            int(outcome),
            arena.num_outcomes,
            epsilon=1e-6,
        )
        start = int(arena.node_first_edge[node_id])
        count = int(arena.node_num_edges[node_id])
        if count > 0:
            edge_ids = start + np.arange(count, dtype=np.int32)
            arena.node_downstream_eval_count[node_id] = np.uint32(
                np.sum(arena.edge_eval_count_R[edge_ids], dtype=np.uint64)
            )
        else:
            arena.node_downstream_eval_count[node_id] = np.uint32(0)
        arena.node_value_cache_status[node_id] = VALUE_CACHE_CLEAN
        arena.node_summary_alpha[node_id] = arena.node_value_cache_C[node_id]
        if changed:
            arena.node_value_cache_version[node_id] += np.uint32(1)
            parent_id = int(arena.node_parent_node[node_id])
            if parent_id != UNKNOWN:
                self._mark_nodes_dirty(np.asarray([parent_id], dtype=np.int32))
        return True

    def _try_categorize_node(self, node_id: int, config: SearchConfig) -> bool:
        assert self.arena is not None
        arena = self.arena
        node_id = int(node_id)
        if int(arena.node_cat_outcome[node_id]) != int(NO_OUTCOME):
            return True
        status = arena.node_status[node_id]
        if status == STATUS_TERMINAL:
            outcome = int(arena.node_terminal_outcome[node_id])
            if outcome == int(NO_OUTCOME):
                return False
            return self._publish_categorical_node(node_id, outcome, 0, UNKNOWN)
        if status != STATUS_EXPANDED:
            return False
        start = int(arena.node_first_edge[node_id])
        count = int(arena.node_num_edges[node_id])
        if count <= 0:
            return False
        edge_ids = start + np.arange(count, dtype=np.int32)
        for edge_id in edge_ids:
            child_id = int(arena.edge_child_node[edge_id])
            if child_id != UNKNOWN:
                self._publish_categorical_edge_from_child(int(edge_id), child_id, config)

        outcomes = arena.edge_cat_outcome[edge_ids].astype(np.int32)
        distances = arena.edge_cat_distance[edge_ids].astype(np.int32)
        known = outcomes != int(NO_OUTCOME)
        win_index = arena.num_outcomes - 1
        win_mask = known & (outcomes == win_index)
        if np.any(win_mask):
            action = self._choose_categorical_distance_edge(
                edge_ids[win_mask],
                distances[win_mask],
                prefer_short=True,
            )
            return self._publish_categorical_node(
                node_id,
                win_index,
                int(arena.edge_cat_distance[action]),
                int(arena.edge_action[action]),
            )
        if not np.all(known):
            return False
        if arena.num_outcomes == 3:
            draw_mask = outcomes == 1
            if np.any(draw_mask):
                edge_id = self._choose_categorical_draw_edge(
                    edge_ids[draw_mask],
                    config,
                )
                return self._publish_categorical_node(
                    node_id,
                    1,
                    int(arena.edge_cat_distance[edge_id]),
                    int(arena.edge_action[edge_id]),
                )
        edge_id = self._choose_categorical_distance_edge(
            edge_ids,
            distances,
            prefer_short=False,
        )
        return self._publish_categorical_node(
            node_id,
            0,
            int(arena.edge_cat_distance[edge_id]),
            int(arena.edge_action[edge_id]),
        )

    def _propagate_categorical(self, start_node_id: int, config: SearchConfig) -> None:
        assert self.arena is not None
        arena = self.arena
        node_id = int(start_node_id)
        seen: set[int] = set()
        while node_id != UNKNOWN and node_id not in seen:
            seen.add(node_id)
            self._try_categorize_node(node_id, config)
            if int(arena.node_cat_outcome[node_id]) == int(NO_OUTCOME):
                return
            parent_id = int(arena.node_parent_node[node_id])
            if parent_id == UNKNOWN:
                return
            edge_id = _edge_id_for_action(arena, parent_id, int(arena.node_parent_action[node_id]))
            if edge_id == UNKNOWN:
                return
            self._publish_categorical_edge_from_child(edge_id, node_id, config)
            node_id = parent_id

    def _choose_categorical_distance_edge(
        self,
        edge_ids: np.ndarray,
        distances: np.ndarray,
        *,
        prefer_short: bool,
    ) -> int:
        assert self.arena is not None
        edge_ids = np.asarray(edge_ids, dtype=np.int32)
        distances = np.asarray(distances, dtype=np.int32)
        if edge_ids.size == 0:
            return UNKNOWN
        best = np.min(distances) if prefer_short else np.max(distances)
        candidates = edge_ids[distances == best]
        actions = self.arena.edge_action[candidates]
        return int(candidates[int(np.argmin(actions))])

    def _choose_categorical_draw_edge(
        self,
        edge_ids: np.ndarray,
        config: SearchConfig,
    ) -> int:
        assert self.arena is not None
        arena = self.arena
        edge_ids = np.asarray(edge_ids, dtype=np.int32)
        if edge_ids.size == 0:
            return UNKNOWN
        distances = arena.edge_cat_distance[edge_ids].astype(np.int32)
        if config.categorical_draw_rule == "fastest_draw":
            return self._choose_categorical_distance_edge(edge_ids, distances, prefer_short=True)
        if config.categorical_draw_rule == "slowest_draw":
            return self._choose_categorical_distance_edge(edge_ids, distances, prefer_short=False)
        if config.categorical_draw_rule == "fixed_order":
            actions = arena.edge_action[edge_ids]
            return int(edge_ids[int(np.argmin(actions))])
        logits = arena.edge_logit[edge_ids]
        return int(edge_ids[int(np.argmax(logits))])

    def _mark_nodes_dirty(self, node_ids: np.ndarray) -> None:
        assert self.arena is not None
        node_ids = np.unique(np.asarray(node_ids, dtype=np.int32))
        node_ids = node_ids[node_ids != UNKNOWN]
        if node_ids.size == 0:
            return
        self.arena.node_value_cache_status[node_ids] = VALUE_CACHE_DIRTY
        self.arena.node_edge_epoch[node_ids] += np.uint32(1)

    def _refresh_edges_from_children(
        self,
        edge_ids: np.ndarray,
        child_node_ids: np.ndarray,
        config: SearchConfig,
    ) -> np.ndarray:
        assert self.arena is not None
        arena = self.arena
        edge_ids = np.asarray(edge_ids, dtype=np.int32).reshape((-1,))
        child_node_ids = np.asarray(child_node_ids, dtype=np.int32).reshape((-1,))
        refreshed = np.zeros((edge_ids.shape[0],), dtype=bool)
        for ix, (edge_id, child_id) in enumerate(zip(edge_ids, child_node_ids, strict=True)):
            refreshed[ix] = self._refresh_edge_from_child(int(edge_id), int(child_id), config)
        return refreshed

    def _refresh_edge_from_child(
        self,
        edge_id: int,
        child_node_id: int,
        config: SearchConfig,
    ) -> bool:
        assert self.arena is not None
        arena = self.arena
        child_id = int(child_node_id)
        edge_id = int(edge_id)
        if child_id == UNKNOWN:
            return True
        child_status = arena.node_status[child_id]
        if self._publish_categorical_edge_from_child(edge_id, child_id, config):
            return True
        if child_status == STATUS_TERMINAL:
            return True
        if child_status != STATUS_EXPANDED:
            return True
        first = int(arena.node_first_edge[child_id])
        count = int(arena.node_num_edges[child_id])
        if count <= 0 or not np.any(arena.edge_has_post[first : first + count]):
            return True
        parent_id = int(arena.edge_parent_node[edge_id])
        if arena.node_value_cache_status[child_id] != VALUE_CACHE_CLEAN:
            self._mark_nodes_dirty(np.asarray([parent_id], dtype=np.int32))
            return False
        if int(arena.edge_child_cache_version[edge_id]) == int(arena.node_value_cache_version[child_id]):
            return True
        parent_player = int(arena.node_current_player[parent_id])
        child_player = int(arena.node_current_player[child_id])
        beta = arena.node_value_cache_C[child_id]
        if parent_player != child_player:
            beta = beta[::-1]
        self._publish_edge_posts(
            np.asarray([edge_id], dtype=np.int32),
            np.asarray([beta], dtype=np.float32),
            np.asarray([1 + int(arena.node_downstream_eval_count[child_id])], dtype=np.uint32),
        )
        arena.edge_child_cache_version[edge_id] = np.int64(arena.node_value_cache_version[child_id])
        return True

    def _try_repair_node(
        self,
        node_id: int,
        config: SearchConfig,
        *,
        policy_mc_samples: int | None = None,
    ) -> bool:
        assert self.arena is not None
        arena = self.arena
        node_id = int(node_id)
        if arena.node_status[node_id] != STATUS_EXPANDED:
            return False
        if int(arena.node_cat_outcome[node_id]) != int(NO_OUTCOME):
            arena.node_value_cache_status[node_id] = VALUE_CACHE_CLEAN
            return True
        start = int(arena.node_first_edge[node_id])
        count = int(arena.node_num_edges[node_id])
        if count <= 0:
            arena.node_value_cache_C[node_id] = arena.node_value_alpha[node_id]
            arena.node_downstream_eval_count[node_id] = np.uint32(0)
            arena.node_value_cache_status[node_id] = VALUE_CACHE_CLEAN
            arena.node_summary_alpha[node_id] = arena.node_value_cache_C[node_id]
            return True
        edge_ids = start + np.arange(count, dtype=np.int32)
        if arena.node_value_cache_status[node_id] == VALUE_CACHE_CLEAN:
            return True
        if arena.node_value_cache_status[node_id] == VALUE_CACHE_UPDATING:
            return False
        arena.node_value_cache_status[node_id] = VALUE_CACHE_UPDATING
        epoch = int(arena.node_edge_epoch[node_id])
        for edge_id in edge_ids:
            child_id = int(arena.edge_child_node[edge_id])
            if not self._refresh_edge_from_child(int(edge_id), child_id, config):
                arena.node_value_cache_status[node_id] = VALUE_CACHE_DIRTY
                return False
        if self._try_categorize_node(node_id, config):
            return True
        if int(arena.node_edge_epoch[node_id]) != epoch:
            arena.node_value_cache_status[node_id] = VALUE_CACHE_DIRTY
            return False
        if not np.any(arena.edge_has_post[edge_ids]):
            arena.node_value_cache_C[node_id] = arena.node_value_alpha[node_id]
            arena.node_downstream_eval_count[node_id] = np.uint32(0)
            arena.node_value_cache_status[node_id] = VALUE_CACHE_CLEAN
            arena.node_summary_alpha[node_id] = arena.node_value_cache_C[node_id]
            return True
        samples = int(config.policy_mc_samples if policy_mc_samples is None else policy_mc_samples)
        alpha = arena.edge_post_alpha[edge_ids]
        legal = np.ones((count,), dtype=bool)
        pi_search = native_policy_target_np(
            self.rng,
            alpha,
            legal,
            _edge_target_kind_np(arena.edge_cat_outcome[edge_ids]),
            arena.edge_cat_outcome[edge_ids],
            samples,
        )
        e_v = np.sum(pi_search[:, None] * alpha, axis=0)
        n_down = int(np.sum(arena.edge_eval_count_R[edge_ids], dtype=np.uint64))
        gamma = float(n_down) / (float(config.state_posterior_kappa_n) + float(n_down))
        cache = _positive((1.0 - gamma) * arena.node_value_alpha[node_id] + gamma * e_v)
        if int(arena.node_edge_epoch[node_id]) != epoch:
            arena.node_value_cache_status[node_id] = VALUE_CACHE_DIRTY
            return False
        arena.node_value_cache_C[node_id] = cache
        arena.node_downstream_eval_count[node_id] = np.uint32(n_down)
        arena.node_summary_alpha[node_id] = cache
        arena.node_value_cache_version[node_id] += np.uint32(1)
        arena.node_value_cache_status[node_id] = VALUE_CACHE_CLEAN
        parent_id = int(arena.node_parent_node[node_id])
        if parent_id != UNKNOWN:
            self._mark_nodes_dirty(np.asarray([parent_id], dtype=np.int32))
        return True

    def _repair_path_to_root(
        self,
        start_node_id: int,
        config: SearchConfig,
        *,
        policy_mc_samples: int | None = None,
    ) -> None:
        assert self.arena is not None
        node_id = int(start_node_id)
        seen: set[int] = set()
        while node_id != UNKNOWN and node_id not in seen:
            seen.add(node_id)
            if self.arena.node_value_cache_status[node_id] != VALUE_CACHE_CLEAN:
                if not self._try_repair_node(node_id, config, policy_mc_samples=policy_mc_samples):
                    return
            node_id = int(self.arena.node_parent_node[node_id])

    def _repair_dirty_frontier(self, config: SearchConfig) -> None:
        assert self.arena is not None
        arena = self.arena
        for _ in range(max(1, int(arena.num_nodes))):
            node_ids = np.arange(int(arena.num_nodes), dtype=np.int32)
            dirty = node_ids[
                (arena.node_status[: arena.num_nodes] == STATUS_EXPANDED)
                & (arena.node_value_cache_status[: arena.num_nodes] != VALUE_CACHE_CLEAN)
            ]
            if dirty.size == 0:
                return
            changed = False
            for node_id in dirty[np.argsort(arena.node_depth[dirty])[::-1]]:
                before = int(arena.node_value_cache_version[node_id])
                self._try_repair_node(int(node_id), config)
                changed = changed or int(arena.node_value_cache_version[node_id]) != before
            if not changed:
                return

    def _backup_path(
        self,
        path_nodes: np.ndarray | None,
        path_edges: np.ndarray | None,
        path_len: int,
        *,
        leaf_node_id: int,
        leaf_value: np.ndarray,
        policy_mc_samples: int,
        config: SearchConfig | None = None,
    ) -> None:
        if path_nodes is None or path_edges is None or path_len <= 0:
            return
        assert self.arena is not None
        arena = self.arena
        final_edge_id = int(path_edges[path_len - 1])
        final_parent_id = int(path_nodes[path_len - 1])
        if int(arena.node_cat_outcome[int(leaf_node_id)]) != int(NO_OUTCOME):
            self._publish_categorical_edge_from_child(
                final_edge_id,
                int(leaf_node_id),
                config,
                increment_eval_count=True,
            )
            if config is not None:
                self._propagate_categorical(final_parent_id, config)
            return
        parent_player = int(arena.node_current_player[final_parent_id])
        leaf_player = int(arena.node_current_player[leaf_node_id])
        aligned = leaf_value[::-1] if parent_player != leaf_player else leaf_value
        self._publish_edge_posts(
            np.asarray([final_edge_id], dtype=np.int32),
            np.asarray([aligned], dtype=np.float32),
            np.asarray([1], dtype=np.uint32),
            increment_eval_count=True,
        )
        if config is not None:
            self._repair_path_to_root(
                final_parent_id,
                config,
                policy_mc_samples=policy_mc_samples,
            )

def run_arena_posterior_tree_search(
    *,
    env: Any,
    root_states: list[Any],
    leaf_evaluator: LeafEvaluator,
    rng_key: jax.Array,
    config: RuntimeSearchConfig,
) -> SearchResult:
    from .search import search_config_from_runtime

    search_config = search_config_from_runtime(config, num_roots=len(root_states))
    search = BatchedPosteriorArenaSearch(env=env, rng_key=rng_key)
    return search.search_batch(
        root_states,
        leaf_evaluator,
        search_config,
    )


def _default_max_nodes(num_roots: int, config: SearchConfig) -> int:
    lanes = max(1, int(config.num_lanes_per_root))
    return max(16, int(num_roots) * (int(config.num_simulations) + 2) * lanes * 2)


def _tree_sample_capacity(num_roots: int, max_nodes: int, config: SearchConfig) -> int:
    configured = config.train_tree_max_nodes_per_step
    if configured is not None:
        return max(1, min(int(configured), int(max_nodes)))
    expected_nodes = int(num_roots) * (int(config.num_simulations) + 1)
    return max(1, min(expected_nodes, int(max_nodes)))


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
    observation_pad_size: int | None = None,
) -> _PendingBatch:
    if recycle_duplicates and missing_state_indices.shape[0] > 1:
        keep = _first_unique_indices(key_words[missing_state_indices])
        missing_rows = missing_rows[keep]
        missing_state_indices = missing_state_indices[keep]
    max_path_len = int(np.max(path_len[missing_rows])) if missing_rows.size else 0
    source_observations = next_state_batch.observation
    eval_indices: np.ndarray | None = None
    if (
        observation_pad_size is not None
        and int(source_observations.shape[0]) == int(observation_pad_size)
        and missing_state_indices.shape[0] <= int(observation_pad_size)
    ):
        observations = source_observations
        eval_indices = missing_state_indices.astype(np.int32, copy=True)
    else:
        observations = _select_array_rows_padded(
            source_observations,
            missing_state_indices,
            target_size=observation_pad_size,
        )
    return _PendingBatch(
        observations=observations,
        key_words=key_words[missing_state_indices].copy(),
        players=players[missing_state_indices].astype(np.int32, copy=True),
        legal=legal[missing_state_indices].copy(),
        root_ids=lane_root_ids[missing_rows].astype(np.int32, copy=True),
        path_nodes=path_nodes[missing_rows, :max_path_len].copy(),
        path_edges=path_edges[missing_rows, :max_path_len].copy(),
        path_len=path_len[missing_rows].astype(np.int16, copy=True),
        eval_indices=eval_indices,
    )


def _merge_pending_batches(pending: list[_PendingBatch]) -> _PendingBatch:
    if len(pending) == 1:
        return pending[0]
    total_size = sum(batch.size for batch in pending)
    if total_size == 0:
        raise ValueError("cannot merge empty pending batches")
    max_path_width = max(batch.path_nodes.shape[1] for batch in pending)
    observations = _merge_padded_observations(pending, total_size)
    if observations is None:
        observations = jnp.concatenate([_pending_selected_observations(batch) for batch in pending], axis=0)
    key_words = np.concatenate([batch.key_words for batch in pending], axis=0)
    players = np.concatenate([batch.players for batch in pending], axis=0)
    legal = np.concatenate([batch.legal for batch in pending], axis=0)
    root_ids = np.concatenate([batch.root_ids for batch in pending], axis=0)
    path_len = np.concatenate([batch.path_len for batch in pending], axis=0)
    path_nodes = np.full((total_size, max_path_width), UNKNOWN, dtype=np.int32)
    path_edges = np.full((total_size, max_path_width), UNKNOWN, dtype=np.int32)
    offset = 0
    for batch in pending:
        width = batch.path_nodes.shape[1]
        next_offset = offset + batch.size
        path_nodes[offset:next_offset, :width] = batch.path_nodes
        path_edges[offset:next_offset, :width] = batch.path_edges
        offset = next_offset
    return _PendingBatch(
        observations=observations,
        key_words=key_words,
        players=players,
        legal=legal,
        root_ids=root_ids,
        path_nodes=path_nodes,
        path_edges=path_edges,
        path_len=path_len,
    )


def _merge_padded_observations(pending: list[_PendingBatch], total_size: int) -> jax.Array | None:
    target_size = int(pending[0].observations.shape[0])
    if target_size <= 0 or int(total_size) > target_size:
        return None
    if any(int(batch.observations.shape[0]) != target_size for batch in pending):
        return None
    if not any(batch.eval_indices is not None or batch.size < target_size for batch in pending):
        return None

    sources = jnp.concatenate([batch.observations for batch in pending], axis=0)
    indices = np.empty((target_size,), dtype=np.int32)
    offset = 0
    source_offset = 0
    for batch in pending:
        if batch.eval_indices is None:
            local_indices = np.arange(batch.size, dtype=np.int32)
        else:
            local_indices = np.asarray(batch.eval_indices, dtype=np.int32)
        next_offset = offset + batch.size
        indices[offset:next_offset] = source_offset + local_indices[: batch.size]
        offset = next_offset
        source_offset += target_size
    indices[offset:] = indices[0] if offset else 0
    return _take_rows_jit(sources, jnp.asarray(indices, dtype=jnp.int32))


def _pending_selected_observations(batch: _PendingBatch) -> jax.Array:
    if batch.eval_indices is None:
        return batch.observations[: batch.size]
    return _select_array_rows(batch.observations, batch.eval_indices)


def _eval_observation_batch(
    observations: jax.Array,
    start: int,
    end: int,
    *,
    target_size: int,
    pad: bool,
) -> jax.Array:
    real_count = int(end - start)
    if real_count <= 0:
        raise ValueError("cannot evaluate an empty pending batch")
    if not pad or real_count >= int(target_size):
        return observations[start:end]
    if start == 0 and int(observations.shape[0]) == int(target_size):
        return observations
    indices = np.empty((int(target_size),), dtype=np.int32)
    indices[:real_count] = np.arange(start, end, dtype=np.int32)
    indices[real_count:] = int(start)
    return _take_rows_jit(observations, jnp.asarray(indices, dtype=jnp.int32))


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
        q_loss_weight=result.q_loss_weight[inverse_jax],
        alpha_root=result.alpha_root[inverse_jax],
        tree_data=result.tree_data,
        search_loss_mask=(
            None
            if result.search_loss_mask is None
            else result.search_loss_mask[inverse_jax]
        ),
        diagnostics=(
            None
            if result.diagnostics is None
            else jax.tree_util.tree_map(lambda x: x[inverse_jax], result.diagnostics)
        ),
        q_target_kind=(
            None if result.q_target_kind is None else result.q_target_kind[inverse_jax]
        ),
        q_target_weight=(
            None if result.q_target_weight is None else result.q_target_weight[inverse_jax]
        ),
        q_target_outcome=(
            None if result.q_target_outcome is None else result.q_target_outcome[inverse_jax]
        ),
        q_target_distance=(
            None if result.q_target_distance is None else result.q_target_distance[inverse_jax]
        ),
        v_target_kind=(
            None if result.v_target_kind is None else result.v_target_kind[inverse_jax]
        ),
        v_target_weight=(
            None if result.v_target_weight is None else result.v_target_weight[inverse_jax]
        ),
        v_target_outcome=(
            None if result.v_target_outcome is None else result.v_target_outcome[inverse_jax]
        ),
        v_target_distance=(
            None if result.v_target_distance is None else result.v_target_distance[inverse_jax]
        ),
    )


def _arena_root_descendant_stats(
    arena: PosteriorArena,
    root_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    root_ids = np.asarray(root_ids, dtype=np.int32)
    num_roots = int(root_ids.shape[0])
    expanded = np.zeros((num_roots,), dtype=np.float32)
    terminal = np.zeros((num_roots,), dtype=np.float32)
    total = np.zeros((num_roots,), dtype=np.float32)
    if num_roots == 0 or int(arena.num_nodes) == 0:
        return expanded, terminal

    owner = np.full((int(arena.num_nodes),), UNKNOWN, dtype=np.int32)
    for root_pos, node_id in enumerate(root_ids):
        if 0 <= int(node_id) < owner.shape[0]:
            owner[int(node_id)] = np.int32(root_pos)

    for node_id in range(int(arena.num_nodes)):
        if owner[node_id] != UNKNOWN:
            continue
        chain: list[int] = []
        current = node_id
        while current != UNKNOWN and owner[current] == UNKNOWN:
            chain.append(current)
            current = int(arena.node_parent_node[current])
        root_pos = UNKNOWN if current == UNKNOWN else int(owner[current])
        for chained_node in chain:
            owner[chained_node] = np.int32(root_pos)

    for node_id in range(int(arena.num_nodes)):
        root_pos = int(owner[node_id])
        if root_pos == UNKNOWN:
            continue
        total[root_pos] += 1.0
        status = arena.node_status[node_id]
        if status == STATUS_EXPANDED:
            expanded[root_pos] += 1.0
        elif status == STATUS_TERMINAL:
            terminal[root_pos] += 1.0

    terminal_fraction = terminal / np.maximum(total, 1.0)
    return expanded, terminal_fraction.astype(np.float32)


def _selected_positions(actions_padded: np.ndarray, actions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    matches = (actions_padded == actions[:, None]) & mask
    return np.argmax(matches, axis=1).astype(np.int32)


def _pad_jax_selection_inputs(
    alpha: np.ndarray,
    actions_padded: np.ndarray,
    mask: np.ndarray,
    *,
    target_rows: int,
    target_edges: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, edges = mask.shape
    out_rows = max(rows, int(target_rows))
    out_edges = max(edges, int(target_edges))
    if out_rows == rows and out_edges == edges:
        return alpha, actions_padded, mask
    padded_alpha = np.ones((out_rows, out_edges, alpha.shape[-1]), dtype=np.float32)
    padded_actions = np.zeros((out_rows, out_edges), dtype=np.int32)
    padded_mask = np.zeros((out_rows, out_edges), dtype=bool)
    padded_alpha[:rows, :edges] = alpha
    padded_actions[:rows, :edges] = actions_padded
    padded_mask[:rows, :edges] = mask
    return padded_alpha, padded_actions, padded_mask


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


def _is_prefix_indices(indices: np.ndarray) -> bool:
    indices = np.asarray(indices, dtype=np.int32)
    return indices.size == 0 or np.array_equal(indices, np.arange(indices.shape[0], dtype=np.int32))


def _pad_actions(actions: np.ndarray, size: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int32)
    if actions.shape[0] >= size:
        return actions
    padded = np.empty((size,), dtype=np.int32)
    padded[: actions.shape[0]] = actions
    padded[actions.shape[0] :] = actions[0] if actions.shape[0] else 0
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
    return _pad_indices(state_indices, size), _pad_actions(actions, size)


def _block_until_ready(value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


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


def _align_outcome_index(
    outcome: int,
    source_player: int,
    target_player: int,
    num_outcomes: int,
) -> int:
    outcome = int(outcome)
    if outcome < 0 or int(source_player) == int(target_player):
        return outcome
    return int(num_outcomes) - 1 - outcome


def _edge_target_kind_np(edge_cat_outcome: np.ndarray) -> np.ndarray:
    return np.where(
        np.asarray(edge_cat_outcome) != int(NO_OUTCOME),
        int(TARGET_CATEGORICAL),
        int(TARGET_DIRICHLET),
    ).astype(np.int8)


def _edge_id_for_action(arena: PosteriorArena, node_id: int, action: int) -> int:
    start = int(arena.node_first_edge[node_id])
    count = int(arena.node_num_edges[node_id])
    if count <= 0:
        return UNKNOWN
    edge_ids = start + np.arange(count, dtype=np.int32)
    matches = edge_ids[arena.edge_action[edge_ids] == int(action)]
    return int(matches[0]) if matches.size else UNKNOWN


def _first_legal_action(legal: np.ndarray) -> int:
    actions = np.flatnonzero(np.asarray(legal, dtype=bool))
    return int(actions[0]) if actions.size else 0


def _leaf_beta(alpha_v: np.ndarray, config: SearchConfig) -> np.ndarray:
    alpha_v = _positive(alpha_v)
    if config.leaf_value_mode == "alpha":
        return alpha_v
    if config.leaf_value_mode == "mean":
        return np.asarray(config.kappa_leaf, dtype=np.float32) * outcome_mean(alpha_v)
    raise ValueError(f"unknown leaf_value_mode: {config.leaf_value_mode!r}")


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
    raise ValueError(f"unknown final_action_mode: {config.final_action_mode!r}")


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
    raise ValueError(f"unknown final_action_mode: {config.final_action_mode!r}")


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
        gamma = rng.standard_gamma(gamma_shape, dtype=np.float32)
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
            next_state.rewards[next_state.current_player],
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


def _select_array_rows(array: jax.Array, rows: np.ndarray) -> jax.Array:
    rows = np.asarray(rows, dtype=np.int32)
    if rows.size == 0:
        return array[:0]
    if _is_prefix_indices(rows):
        return array[: rows.shape[0]]
    return _take_rows_jit(array, jnp.asarray(rows, dtype=jnp.int32))


def _select_array_rows_padded(
    array: jax.Array,
    rows: np.ndarray,
    *,
    target_size: int | None,
) -> jax.Array:
    rows = np.asarray(rows, dtype=np.int32)
    if target_size is None or rows.shape[0] >= int(target_size):
        return _select_array_rows(array, rows)
    if rows.shape[0] == 0:
        return array[:0]
    indices = np.empty((int(target_size),), dtype=np.int32)
    indices[: rows.shape[0]] = rows
    indices[rows.shape[0] :] = rows[0]
    return _take_rows_jit(array, jnp.asarray(indices, dtype=jnp.int32))


def _state_batch_size(state: Any) -> int:
    leaves = jax.tree_util.tree_leaves(state)
    if not leaves:
        return 0
    return int(leaves[0].shape[0])


def _key_id(key: np.ndarray) -> bytes:
    return np.ascontiguousarray(key, dtype=np.uint32).reshape((4,)).tobytes()


def _positive(alpha: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(alpha, dtype=np.float32), np.float32(1e-6))
