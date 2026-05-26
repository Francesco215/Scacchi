from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NamedTuple

import jax
import numpy as np


KEY_WORDS = 4
NO_CHILD_KEY = np.zeros((KEY_WORDS,), dtype=np.uint32)

EVAL_UNEXPANDED = 0
EVAL_INFLIGHT = 1
EVAL_EXPANDING = 2
EVAL_EXPANDED = 3
EVAL_TERMINAL = 4

VALUE_CACHE_DIRTY = 0
VALUE_CACHE_CLEAN = 1
VALUE_CACHE_UPDATING = 2

FinalActionMode = Literal[
    "posterior_argmax",
    "posterior_sample",
]


@dataclass(frozen=True, slots=True)
class StateKey:
    words: tuple[int, int, int, int]

    @classmethod
    def from_array(cls, value: Any) -> "StateKey":
        arr = np.asarray(jax.device_get(value), dtype=np.uint32).reshape((KEY_WORDS,))
        return cls(tuple(int(x) for x in arr))

    @classmethod
    def zero(cls) -> "StateKey":
        return cls((0, 0, 0, 0))

    @property
    def hi_lo_hex(self) -> tuple[str, str]:
        hi = (self.words[0] << 32) | self.words[1]
        lo = (self.words[2] << 32) | self.words[3]
        return f"{hi:016x}", f"{lo:016x}"

    @property
    def hex(self) -> str:
        hi, lo = self.hi_lo_hex
        return hi + lo

    def to_array(self) -> np.ndarray:
        return np.asarray(self.words, dtype=np.uint32)

    def is_zero(self) -> bool:
        return self.words == (0, 0, 0, 0)


@dataclass(slots=True)
class NodeBlob:
    key: StateKey
    current_player: int
    legal_actions: np.ndarray
    value_alpha: np.ndarray
    policy_logits: np.ndarray
    q_alpha: np.ndarray
    edge_base_alpha: np.ndarray
    child_keys: np.ndarray
    visits: np.ndarray
    edge_B: np.ndarray | None = None
    edge_has_post: np.ndarray | None = None
    edge_eval_count_R: np.ndarray | None = None
    edge_version: np.ndarray | None = None
    edge_child_cache_version: np.ndarray | None = None
    terminal_outcome: int = -1
    status: int = EVAL_EXPANDED
    game_id: int = 0
    model_id: int = 0
    dirty_version: int = 0
    parent_key: StateKey | None = None
    parent_node_id: int = -1
    parent_action: int = -1
    depth: int = 0
    value_cache_C: np.ndarray | None = None
    downstream_eval_count: int = 0
    value_cache_status: int = VALUE_CACHE_CLEAN
    value_cache_version: int = 0
    edge_epoch: int = 0
    pi_search: np.ndarray | None = None
    state_summary_alpha: np.ndarray | None = None
    edge_evidence_E: np.ndarray | None = field(default=None, repr=False)
    action_to_index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.legal_actions = np.asarray(self.legal_actions, dtype=np.uint32)
        self.value_alpha = _positive(self.value_alpha)
        self.policy_logits = np.asarray(self.policy_logits, dtype=np.float16)
        self.q_alpha = _positive(self.q_alpha)
        self.edge_base_alpha = _positive(self.edge_base_alpha)
        self.child_keys = np.asarray(self.child_keys, dtype=np.uint32).reshape((-1, KEY_WORDS))
        self.visits = np.asarray(self.visits, dtype=np.uint32)
        edge_shape = self.edge_base_alpha.shape
        legacy_evidence = (
            None
            if self.edge_evidence_E is None
            else np.asarray(self.edge_evidence_E, dtype=np.float32).reshape(edge_shape)
        )
        if self.edge_B is None:
            self.edge_B = (
                _positive(self.edge_base_alpha + legacy_evidence)
                if legacy_evidence is not None
                else self.edge_base_alpha.copy()
            )
        else:
            self.edge_B = _positive(np.asarray(self.edge_B, dtype=np.float32).reshape(edge_shape))
        if self.edge_has_post is None:
            if legacy_evidence is None:
                self.edge_has_post = np.zeros(edge_shape[:1], dtype=bool)
            else:
                self.edge_has_post = np.sum(np.abs(legacy_evidence), axis=-1) > 0
        else:
            self.edge_has_post = np.asarray(self.edge_has_post, dtype=bool).reshape(edge_shape[:1])
        if self.edge_eval_count_R is None:
            self.edge_eval_count_R = np.where(self.edge_has_post, self.visits, 0).astype(np.uint32)
        else:
            self.edge_eval_count_R = np.asarray(self.edge_eval_count_R, dtype=np.uint32).reshape(edge_shape[:1])
        if self.edge_version is None:
            self.edge_version = np.zeros(edge_shape[:1], dtype=np.uint32)
        else:
            self.edge_version = np.asarray(self.edge_version, dtype=np.uint32).reshape(edge_shape[:1])
        if self.edge_child_cache_version is None:
            self.edge_child_cache_version = np.full(edge_shape[:1], -1, dtype=np.int64)
        else:
            self.edge_child_cache_version = np.asarray(
                self.edge_child_cache_version,
                dtype=np.int64,
            ).reshape(edge_shape[:1])
        if self.parent_key is None:
            self.parent_key = StateKey.zero()
        if self.value_cache_C is None:
            self.value_cache_C = self.value_alpha.copy()
        else:
            self.value_cache_C = _positive(self.value_cache_C)
        if self.pi_search is None:
            self.pi_search = fast_pi_search(self.edge_post_alpha)
        else:
            self.pi_search = np.asarray(self.pi_search, dtype=np.float16)
        if self.state_summary_alpha is None:
            self.state_summary_alpha = _positive(self.value_cache_C)
        else:
            self.state_summary_alpha = _positive(self.state_summary_alpha)
        self.edge_evidence_E = self.edge_evidence_delta
        self.action_to_index = {int(action): ix for ix, action in enumerate(self.legal_actions)}

    @property
    def edge_post_alpha(self) -> np.ndarray:
        return _positive(
            np.where(
                np.asarray(self.edge_has_post, dtype=bool)[:, None],
                np.asarray(self.edge_B, dtype=np.float32),
                self.edge_base_alpha,
            )
        )

    @property
    def edge_evidence_delta(self) -> np.ndarray:
        return np.where(
            np.asarray(self.edge_has_post, dtype=bool)[:, None],
            np.asarray(self.edge_B, dtype=np.float32) - self.edge_base_alpha,
            np.zeros_like(self.edge_base_alpha),
        )

    @property
    def expanded(self) -> bool:
        return self.status == EVAL_EXPANDED

    @property
    def terminal(self) -> bool:
        return self.status == EVAL_TERMINAL

    @property
    def has_child_evidence(self) -> bool:
        return bool(np.any(np.asarray(self.edge_has_post, dtype=bool)))

    def mark_dirty(self) -> None:
        self.dirty_version += 1
        self.edge_epoch += 1
        self.value_cache_status = VALUE_CACHE_DIRTY
        self.pi_search = fast_pi_search(self.edge_post_alpha)

    def publish_edge_post(self, index: int, beta: np.ndarray, eval_count: int) -> None:
        ix = int(index)
        self.edge_B[ix] = _positive(beta)
        self.edge_has_post[ix] = True
        self.edge_eval_count_R[ix] = np.uint32(max(0, int(eval_count)))
        self.edge_version[ix] += np.uint32(1)
        self.visits[ix] += np.uint32(1)
        self.edge_evidence_E = self.edge_evidence_delta
        self.mark_dirty()

    def publish_clean_cache(
        self,
        value_cache: np.ndarray,
        downstream_eval_count: int,
        pi_search: np.ndarray,
    ) -> None:
        self.value_cache_C = _positive(value_cache)
        self.downstream_eval_count = int(max(0, downstream_eval_count))
        self.pi_search = np.asarray(pi_search, dtype=np.float16)
        self.state_summary_alpha = self.value_cache_C.copy()
        self.value_cache_status = VALUE_CACHE_CLEAN
        self.value_cache_version += 1

    @classmethod
    def expanded_node(
        cls,
        *,
        key: StateKey,
        current_player: int,
        legal_action_mask: np.ndarray,
        value_alpha: np.ndarray,
        policy_logits: np.ndarray,
        q_alpha: np.ndarray,
        parent_key: StateKey | None = None,
        parent_node_id: int = -1,
        parent_action: int = -1,
        depth: int = 0,
        game_id: int = 0,
        model_id: int = 0,
    ) -> "NodeBlob":
        legal_actions = np.flatnonzero(np.asarray(legal_action_mask, dtype=bool)).astype(np.uint32)
        sparse_q = np.asarray(q_alpha, dtype=np.float32)[legal_actions]
        return cls(
            key=key,
            current_player=int(current_player),
            legal_actions=legal_actions,
            value_alpha=value_alpha,
            policy_logits=np.asarray(policy_logits)[legal_actions],
            q_alpha=sparse_q,
            edge_base_alpha=sparse_q.copy(),
            child_keys=np.zeros((legal_actions.shape[0], KEY_WORDS), dtype=np.uint32),
            visits=np.zeros((legal_actions.shape[0],), dtype=np.uint32),
            edge_B=sparse_q.copy(),
            edge_has_post=np.zeros((legal_actions.shape[0],), dtype=bool),
            edge_eval_count_R=np.zeros((legal_actions.shape[0],), dtype=np.uint32),
            edge_version=np.zeros((legal_actions.shape[0],), dtype=np.uint32),
            edge_child_cache_version=np.full((legal_actions.shape[0],), -1, dtype=np.int64),
            game_id=game_id,
            model_id=model_id,
            parent_key=parent_key,
            parent_node_id=parent_node_id,
            parent_action=parent_action,
            depth=depth,
            value_cache_C=_positive(value_alpha),
            downstream_eval_count=0,
            value_cache_status=VALUE_CACHE_CLEAN,
            value_cache_version=0,
            edge_epoch=0,
        )

    @classmethod
    def terminal_node(
        cls,
        *,
        key: StateKey,
        current_player: int,
        terminal_outcome: int,
        num_outcomes: int = 3,
        parent_key: StateKey | None = None,
        parent_node_id: int = -1,
        parent_action: int = -1,
        depth: int = 0,
        game_id: int = 0,
        model_id: int = 0,
    ) -> "NodeBlob":
        value_alpha = np.zeros((num_outcomes,), dtype=np.float32)
        value_alpha[int(terminal_outcome)] = 1.0
        return cls(
            key=key,
            current_player=int(current_player),
            legal_actions=np.zeros((0,), dtype=np.uint32),
            value_alpha=value_alpha,
            policy_logits=np.zeros((0,), dtype=np.float16),
            q_alpha=np.zeros((0, num_outcomes), dtype=np.float32),
            edge_base_alpha=np.zeros((0, num_outcomes), dtype=np.float32),
            child_keys=np.zeros((0, KEY_WORDS), dtype=np.uint32),
            visits=np.zeros((0,), dtype=np.uint32),
            edge_B=np.zeros((0, num_outcomes), dtype=np.float32),
            edge_has_post=np.zeros((0,), dtype=bool),
            edge_eval_count_R=np.zeros((0,), dtype=np.uint32),
            edge_version=np.zeros((0,), dtype=np.uint32),
            edge_child_cache_version=np.zeros((0,), dtype=np.int64),
            terminal_outcome=int(terminal_outcome),
            status=EVAL_TERMINAL,
            game_id=game_id,
            model_id=model_id,
            parent_key=parent_key,
            parent_node_id=parent_node_id,
            parent_action=parent_action,
            depth=depth,
            value_cache_C=value_alpha,
        )

    @classmethod
    def inflight_node(
        cls,
        *,
        key: StateKey,
        current_player: int = 0,
        num_outcomes: int = 3,
        parent_key: StateKey | None = None,
        parent_node_id: int = -1,
        parent_action: int = -1,
        depth: int = 0,
        game_id: int = 0,
        model_id: int = 0,
    ) -> "NodeBlob":
        return cls(
            key=key,
            current_player=int(current_player),
            legal_actions=np.zeros((0,), dtype=np.uint32),
            value_alpha=np.ones((num_outcomes,), dtype=np.float32),
            policy_logits=np.zeros((0,), dtype=np.float16),
            q_alpha=np.zeros((0, num_outcomes), dtype=np.float32),
            edge_base_alpha=np.zeros((0, num_outcomes), dtype=np.float32),
            child_keys=np.zeros((0, KEY_WORDS), dtype=np.uint32),
            visits=np.zeros((0,), dtype=np.uint32),
            edge_B=np.zeros((0, num_outcomes), dtype=np.float32),
            edge_has_post=np.zeros((0,), dtype=bool),
            edge_eval_count_R=np.zeros((0,), dtype=np.uint32),
            edge_version=np.zeros((0,), dtype=np.uint32),
            edge_child_cache_version=np.zeros((0,), dtype=np.int64),
            status=EVAL_INFLIGHT,
            game_id=game_id,
            model_id=model_id,
            parent_key=parent_key,
            parent_node_id=parent_node_id,
            parent_action=parent_action,
            depth=depth,
        )


@dataclass(frozen=True, slots=True)
class SearchConfig:
    num_simulations: int
    max_depth: int = 128
    num_lanes_per_root: int = 1
    eval_batch_size: int = 1024
    leaf_value_mode: Literal["alpha", "mean"] = "alpha"
    kappa_leaf: float = 1.0
    kappa_terminal: float = 8.0
    epsilon_terminal: float = 1e-6
    categorical_epsilon: float = 1e-4
    categorical_draw_rule: Literal["policy_prior", "fastest_draw", "slowest_draw", "fixed_order"] = "policy_prior"
    state_posterior_kappa_n: float = 9.0
    policy_mc_samples: int = 32
    tau_internal: float = 1.0
    final_action_mode: FinalActionMode = "posterior_argmax"
    pad_eval_batches: bool = True
    pad_jax_select: bool = False
    np_select_below: int = 1024
    grouped_expansion: bool = True
    lane_indexed_step: bool = True
    stable_lane_batch: bool = True
    pad_pending_observation_gather: bool = True
    train_tree_nodes: bool = False
    train_tree_include_root: bool = False
    train_tree_include_terminal: bool = False
    train_tree_min_q_evidence: float = 0.0
    train_tree_max_nodes_per_step: int | None = None


class EvalBatch(NamedTuple):
    observations: jax.Array
    keys: tuple[StateKey, ...]


class EvalResult(NamedTuple):
    logits: jax.Array
    value_alpha: jax.Array
    q_alpha: jax.Array


LeafEvaluator = Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]]


class TreeTrainingData(NamedTuple):
    obs: jax.Array
    action_weights: jax.Array
    played_action: jax.Array
    legal_action_mask: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    value_tgt: jax.Array
    policy_loss_mask: jax.Array
    value_loss_mask: jax.Array
    search_loss_mask: jax.Array
    outcome_mask: jax.Array
    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None

    @property
    def q_evidence_mass(self) -> jax.Array:
        return self.q_loss_weight


class SearchDiagnostics(NamedTuple):
    path_depth_mean: jax.Array
    path_depth_p50: jax.Array
    path_depth_p90: jax.Array
    path_depth_max: jax.Array
    expanded_nodes: jax.Array
    terminal_fraction: jax.Array
    root_policy_entropy: jax.Array
    root_gamma: jax.Array
    root_downstream_eval_count: jax.Array
    root_q_concentration: jax.Array


class SearchResult(NamedTuple):
    action: jax.Array
    action_weights: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    alpha_root: jax.Array
    tree_data: TreeTrainingData | None = None
    search_loss_mask: jax.Array | None = None
    diagnostics: SearchDiagnostics | None = None
    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None

    @property
    def q_evidence_mass(self) -> jax.Array:
        return self.q_loss_weight


class PathStep(NamedTuple):
    key: StateKey
    action: int


def align_wdl(alpha: np.ndarray, source_player: int, target_player: int) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float32)
    if int(source_player) != int(target_player):
        return alpha[..., ::-1].copy()
    return alpha


def outcome_mean(alpha: np.ndarray) -> np.ndarray:
    alpha = _positive(alpha)
    return alpha / np.sum(alpha, axis=-1, keepdims=True)


def outcome_utility(dist: np.ndarray) -> np.ndarray:
    return dist[..., -1] - dist[..., 0]


def fast_pi_search(alpha: np.ndarray, tau: float = 1.0) -> np.ndarray:
    if alpha.shape[0] == 0:
        return np.zeros((0,), dtype=np.float16)
    q_mean = outcome_utility(outcome_mean(alpha))
    tau = max(float(tau), 1e-6)
    logits = q_mean / tau
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)
    return probs.astype(np.float16)


def state_summary(pi_search: np.ndarray, edge_post_alpha: np.ndarray) -> np.ndarray:
    if edge_post_alpha.shape[0] == 0:
        return np.ones((edge_post_alpha.shape[-1] if edge_post_alpha.ndim else 3,), dtype=np.float32)
    return _positive(np.sum(np.asarray(pi_search, dtype=np.float32)[:, None] * edge_post_alpha, axis=0))


def terminal_outcome_from_reward(reward: float, num_outcomes: int) -> int:
    rounded = int(np.rint(reward))
    if num_outcomes == 2:
        return int(np.clip((rounded + 1) // 2, 0, num_outcomes - 1))
    if num_outcomes == 3:
        return int(np.clip(rounded + 1, 0, num_outcomes - 1))
    raise ValueError(f"unsupported outcome count: {num_outcomes}")


def _positive(alpha: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(alpha, dtype=np.float32), np.float32(1e-6))
