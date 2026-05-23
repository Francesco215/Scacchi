from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, NamedTuple

import jax
import numpy as np


KEY_WORDS = 4
NO_CHILD_KEY = np.zeros((KEY_WORDS,), dtype=np.uint32)

FinalActionMode = Literal["argmax_q_mean", "posterior_argmax", "posterior_sample"]
DuplicateLeafMode = Literal["recycle_lane", "park_lane"]


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
    def redis_hex(self) -> str:
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
    edge_evidence_E: np.ndarray
    child_keys: np.ndarray
    visits: np.ndarray
    terminal_outcome: int = -1
    status: int = 1
    game_id: int = 0
    model_id: int = 0
    dirty_version: int = 0
    pi_search: np.ndarray | None = None
    state_summary_alpha: np.ndarray | None = None
    action_to_index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.legal_actions = np.asarray(self.legal_actions, dtype=np.uint32)
        self.value_alpha = _positive(self.value_alpha)
        self.policy_logits = np.asarray(self.policy_logits, dtype=np.float16)
        self.q_alpha = _positive(self.q_alpha)
        self.edge_base_alpha = _positive(self.edge_base_alpha)
        self.edge_evidence_E = np.asarray(self.edge_evidence_E, dtype=np.float32)
        self.child_keys = np.asarray(self.child_keys, dtype=np.uint32).reshape((-1, KEY_WORDS))
        self.visits = np.asarray(self.visits, dtype=np.uint32)
        if self.pi_search is None:
            self.pi_search = fast_pi_search(self.edge_base_alpha + self.edge_evidence_E)
        else:
            self.pi_search = np.asarray(self.pi_search, dtype=np.float16)
        if self.state_summary_alpha is None:
            self.state_summary_alpha = state_summary(self.pi_search, self.edge_post_alpha)
        else:
            self.state_summary_alpha = _positive(self.state_summary_alpha)
        self.action_to_index = {int(action): ix for ix, action in enumerate(self.legal_actions)}

    @property
    def edge_post_alpha(self) -> np.ndarray:
        return _positive(self.edge_base_alpha + self.edge_evidence_E)

    @property
    def expanded(self) -> bool:
        return self.status == 1

    @property
    def terminal(self) -> bool:
        return self.status == 2

    def mark_dirty(self) -> None:
        self.dirty_version += 1
        self.pi_search = fast_pi_search(self.edge_post_alpha)
        self.state_summary_alpha = state_summary(self.pi_search, self.edge_post_alpha)

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
            edge_evidence_E=np.zeros_like(sparse_q, dtype=np.float32),
            child_keys=np.zeros((legal_actions.shape[0], KEY_WORDS), dtype=np.uint32),
            visits=np.zeros((legal_actions.shape[0],), dtype=np.uint32),
            game_id=game_id,
            model_id=model_id,
        )

    @classmethod
    def terminal_node(
        cls,
        *,
        key: StateKey,
        current_player: int,
        terminal_outcome: int,
        num_outcomes: int = 3,
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
            edge_evidence_E=np.zeros((0, num_outcomes), dtype=np.float32),
            child_keys=np.zeros((0, KEY_WORDS), dtype=np.uint32),
            visits=np.zeros((0,), dtype=np.uint32),
            terminal_outcome=int(terminal_outcome),
            status=2,
            game_id=game_id,
            model_id=model_id,
        )

    @classmethod
    def inflight_node(
        cls,
        *,
        key: StateKey,
        current_player: int = 0,
        num_outcomes: int = 3,
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
            edge_evidence_E=np.zeros((0, num_outcomes), dtype=np.float32),
            child_keys=np.zeros((0, KEY_WORDS), dtype=np.uint32),
            visits=np.zeros((0,), dtype=np.uint32),
            status=0,
            game_id=game_id,
            model_id=model_id,
        )


@dataclass(frozen=True, slots=True)
class SearchConfig:
    num_simulations: int
    max_depth: int = 128
    num_lanes_per_root: int = 1
    eval_batch_size: int = 1024
    c_leaf: float = 1.0
    c_terminal: float = 8.0
    c_state: float = 0.1
    c_value_search: float = 1.0
    policy_mc_samples: int = 32
    tau_internal: float = 1.0
    duplicate_leaf_mode: DuplicateLeafMode = "recycle_lane"
    final_action_mode: FinalActionMode = "argmax_q_mean"
    pad_eval_batches: bool = True
    pad_jax_select: bool = False
    np_select_below: int = 1024
    grouped_expansion: bool = True
    lane_indexed_step: bool = True
    stable_lane_batch: bool = True
    pad_pending_observation_gather: bool = True
    redis_inflight_ttl_ms: int = 30000


class EvalBatch(NamedTuple):
    observations: jax.Array
    keys: tuple[StateKey, ...]


class EvalResult(NamedTuple):
    logits: jax.Array
    value_alpha: jax.Array
    q_alpha: jax.Array


LeafEvaluator = Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]]


class SearchResult(NamedTuple):
    action: jax.Array
    action_weights: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_evidence_mass: jax.Array
    alpha_root: jax.Array


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
