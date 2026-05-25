from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from .backup import (
    backup_path,
    repair_dirty_frontier,
    terminal_dirichlet,
    update_edge_base_from_child,
    update_parent_child_edge,
)
from .pack import pack_nodes_for_selection
from .selection import posterior_best_policy_target_np, thompson_select_jax
from .state_hash import canonical_state_key, state_keys_to_host
from .store import InMemoryNodeStore, NodeStore
from .types import (
    EVAL_EXPANDING,
    EVAL_INFLIGHT,
    LeafEvaluator,
    NodeBlob,
    PathStep,
    SearchConfig,
    SearchDiagnostics,
    SearchResult,
    StateKey,
    TreeTrainingData,
    outcome_mean,
    terminal_outcome_from_reward,
)


class WavefrontPosteriorTreeBatchOutput(NamedTuple):
    action: jax.Array
    action_weights: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    alpha_root: jax.Array
    trees: tuple[Any, ...]
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


@dataclass(slots=True)
class _EvalRequest:
    root_id: int
    leaf_key: StateKey
    leaf_state: Any
    path: list[PathStep]


@dataclass(slots=True)
class _Lane:
    root_id: int
    state: Any
    key: StateKey
    path: list[PathStep]


_STEP_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, Any]] = {}
_KEY_CACHE: dict[type, Any] = {}


class BatchedPosteriorSearch:
    def __init__(
        self,
        *,
        env: Any,
        store: NodeStore | None = None,
        rng_key: jax.Array | None = None,
    ) -> None:
        self.env = env
        self.store = store if store is not None else InMemoryNodeStore()
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        seed = int(
            jax.device_get(
                jax.random.randint(rng_key, (), minval=0, maxval=np.iinfo(np.int32).max)
            )
        )
        self.rng = np.random.default_rng(seed)
        self.jax_key = rng_key
        self.num_actions: int | None = None
        self.num_outcomes: int | None = None

    def initialize_roots(
        self,
        root_states: list[Any],
        root_eval_result: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[StateKey, ...]:
        if not root_states:
            raise ValueError("root_states must not be empty")
        logits, value_alpha, q_alpha = jax.device_get(root_eval_result)
        self.num_actions = int(logits.shape[-1])
        self.num_outcomes = int(value_alpha.shape[-1])
        root_keys = _compute_keys(root_states)
        existing = self.store.get_many(root_keys)
        nodes: list[NodeBlob] = []
        for ix, (state, key) in enumerate(zip(root_states, root_keys, strict=True)):
            current = _current_player(state)
            if existing.get(key) is not None and existing[key].status not in (
                EVAL_INFLIGHT,
                EVAL_EXPANDING,
            ):
                continue
            if _terminated(state):
                outcome = terminal_outcome_from_reward(
                    float(_as_numpy(state.rewards)[current]),
                    self.num_outcomes,
                )
                nodes.append(
                    NodeBlob.terminal_node(
                        key=key,
                        current_player=current,
                        terminal_outcome=outcome,
                        num_outcomes=self.num_outcomes,
                    )
                )
                continue
            nodes.append(
                NodeBlob.expanded_node(
                    key=key,
                    current_player=current,
                    legal_action_mask=_as_numpy(state.legal_action_mask),
                    value_alpha=value_alpha[ix],
                    policy_logits=logits[ix],
                    q_alpha=q_alpha[ix],
                )
            )
        self.store.put_many(nodes)
        return root_keys

    def search_batch(
        self,
        root_states: list[Any],
        leaf_evaluator: LeafEvaluator,
        config: SearchConfig,
    ) -> SearchResult:
        root_observations = jnp.stack([state.observation for state in root_states], axis=0)
        root_eval_result = leaf_evaluator(root_observations)
        root_keys = self.initialize_roots(root_states, root_eval_result)
        self._run_wavefront(root_states, root_keys, leaf_evaluator, config)
        return self.finish_search(root_keys, config)

    def finish_search(
        self,
        root_keys: tuple[StateKey, ...],
        config: SearchConfig,
    ) -> SearchResult:
        if self.num_actions is None or self.num_outcomes is None:
            raise ValueError("search has not been initialized")
        repair_dirty_frontier(
            self.store,
            rng=self.rng,
            num_samples=config.backup_mc_samples,
            state_posterior_kappa_n=config.state_posterior_kappa_n,
        )
        actions = []
        policies = []
        beta_q = []
        beta_v = []
        q_weight = []
        alpha_roots = []
        root_nodes = []
        for root_key in root_keys:
            root = self.store.get_many([root_key])[root_key]
            if root is None:
                raise KeyError(f"missing root node {root_key.redis_hex}")
            root_nodes.append(root)
            alpha_dense = np.zeros((self.num_actions, self.num_outcomes), dtype=np.float32)
            legal = np.zeros((self.num_actions,), dtype=bool)
            for ix, action in enumerate(root.legal_actions):
                alpha_dense[int(action)] = root.edge_post_alpha[ix]
                legal[int(action)] = True
            policy = posterior_best_policy_target_np(
                self.rng,
                alpha_dense,
                legal,
                config.policy_mc_samples,
            )
            action = _commit_action(self.rng, config, policy, alpha_dense, legal)
            actions.append(action)
            policies.append(policy)
            beta_q.append(alpha_dense)
            beta_v.append(root.value_cache_C)
            q_weight.append(policy)
            alpha_roots.append(alpha_dense)
        self.store.flush_dirty()
        policy_array = np.stack(policies, axis=0)
        alpha_array = np.stack(alpha_roots, axis=0)
        search_loss_mask = np.sum(policy_array, axis=-1) > 0.0
        return SearchResult(
            action=jnp.asarray(actions, dtype=jnp.int32),
            action_weights=jnp.asarray(policy_array, dtype=jnp.float32),
            beta_Q_target=jnp.asarray(np.stack(beta_q, axis=0), dtype=jnp.float32),
            beta_V_target=jnp.asarray(np.stack(beta_v, axis=0), dtype=jnp.float32),
            q_loss_weight=jnp.asarray(np.stack(q_weight, axis=0), dtype=jnp.float32),
            alpha_root=jnp.asarray(alpha_array, dtype=jnp.float32),
            search_loss_mask=jnp.asarray(search_loss_mask),
            diagnostics=_store_search_diagnostics(root_nodes, policy_array, alpha_array, config),
        )

    def _run_wavefront(
        self,
        root_states: list[Any],
        root_keys: tuple[StateKey, ...],
        leaf_evaluator: LeafEvaluator,
        config: SearchConfig,
    ) -> None:
        done = np.zeros((len(root_states),), dtype=np.int32)
        max_attempts = max(1, len(root_states) * config.num_simulations * (config.max_depth + 4) * 4)
        attempts = 0
        while np.any(done < config.num_simulations):
            attempts += 1
            if attempts > max_attempts:
                unfinished = np.flatnonzero(done < config.num_simulations).tolist()
                raise RuntimeError(f"wavefront posterior search stalled for roots {unfinished}")
            lanes = [
                _Lane(root_id=ix, state=root_states[ix], key=root_keys[ix], path=[])
                for ix in range(len(root_states))
                for _ in range(max(1, int(config.num_lanes_per_root)))
                if done[ix] < config.num_simulations
            ]
            if not lanes:
                break
            pending = self._traverse_lanes(lanes, done, config)
            self._evaluate_pending(pending, leaf_evaluator, done, config)
            self.store.flush_dirty()

    def _traverse_lanes(
        self,
        lanes: list[_Lane],
        done: np.ndarray,
        config: SearchConfig,
    ) -> list[_EvalRequest]:
        active = lanes
        pending: list[_EvalRequest] = []
        for _ in range(config.max_depth):
            active = [lane for lane in active if done[lane.root_id] < config.num_simulations]
            if not active:
                break
            nodes_by_key = self.store.get_many([lane.key for lane in active])
            selectable: list[tuple[_Lane, NodeBlob]] = []
            next_active: list[_Lane] = []
            for lane in active:
                node = nodes_by_key[lane.key]
                if node is None or node.status in (EVAL_INFLIGHT, EVAL_EXPANDING):
                    continue
                if node.terminal:
                    if done[lane.root_id] >= config.num_simulations:
                        continue
                    if lane.path:
                        backup_path(
                            self.store,
                            path=lane.path,
                            leaf_node=node,
                            leaf_value=terminal_dirichlet(
                                node.terminal_outcome,
                                self.num_outcomes or 3,
                                kappa_terminal=config.kappa_terminal,
                                epsilon_terminal=config.epsilon_terminal,
                            ),
                            rng=self.rng,
                            backup_mc_samples=config.backup_mc_samples,
                            state_posterior_kappa_n=config.state_posterior_kappa_n,
                        )
                    done[lane.root_id] += 1
                    continue
                if node.legal_actions.shape[0] == 0:
                    continue
                selectable.append((lane, node))
            if not selectable:
                break

            select_lanes = [item[0] for item in selectable]
            select_nodes = [item[1] for item in selectable]
            packed = pack_nodes_for_selection(select_nodes)
            self.jax_key, select_key = jax.random.split(self.jax_key)
            actions = np.asarray(
                jax.device_get(
                    thompson_select_jax(
                        select_key,
                        packed.edge_post_alpha,
                        packed.legal_actions,
                        packed.action_mask,
                    )
                ),
                dtype=np.int32,
            )
            state_batch = _stack_states([lane.state for lane in select_lanes])
            next_state_batch = _batched_step(self.env)(state_batch, jnp.asarray(actions))
            child_keys = state_keys_to_host(_batched_key_fn(next_state_batch))
            next_states = _split_batched_state(next_state_batch)
            child_nodes = self.store.get_many(child_keys)

            for lane, parent_node, action, next_state, child_key in zip(
                select_lanes,
                select_nodes,
                actions,
                next_states,
                child_keys,
                strict=True,
            ):
                path = lane.path + [PathStep(parent_node.key, int(action))]
                update_parent_child_edge(
                    self.store,
                    parent_key=parent_node.key,
                    action=int(action),
                    child_key=child_key,
                )

                if _terminated(next_state):
                    terminal_node = _terminal_node_from_state(child_key, next_state, self.num_outcomes or 3)
                    terminal_node.parent_key = parent_node.key
                    terminal_node.parent_action = int(action)
                    terminal_node.depth = int(parent_node.depth) + 1
                    if child_nodes[child_key] is None:
                        self.store.put_many([terminal_node])
                    else:
                        terminal_node = child_nodes[child_key]
                    backup_path(
                        self.store,
                        path=path,
                        leaf_node=terminal_node,
                        leaf_value=terminal_dirichlet(
                            terminal_node.terminal_outcome,
                            self.num_outcomes or 3,
                            kappa_terminal=config.kappa_terminal,
                            epsilon_terminal=config.epsilon_terminal,
                        ),
                        rng=self.rng,
                        backup_mc_samples=config.backup_mc_samples,
                        state_posterior_kappa_n=config.state_posterior_kappa_n,
                    )
                    done[lane.root_id] += 1
                    continue

                child = child_nodes[child_key]
                if child is None:
                    pending.append(_EvalRequest(lane.root_id, child_key, next_state, path))
                    continue
                if child.status in (EVAL_INFLIGHT, EVAL_EXPANDING):
                    continue
                update_edge_base_from_child(
                    self.store,
                    parent_key=parent_node.key,
                    action=int(action),
                    child=child,
                )
                if child.terminal:
                    backup_path(
                        self.store,
                        path=path,
                        leaf_node=child,
                        leaf_value=terminal_dirichlet(
                            child.terminal_outcome,
                            self.num_outcomes or 3,
                            kappa_terminal=config.kappa_terminal,
                            epsilon_terminal=config.epsilon_terminal,
                        ),
                        rng=self.rng,
                        backup_mc_samples=config.backup_mc_samples,
                        state_posterior_kappa_n=config.state_posterior_kappa_n,
                    )
                    done[lane.root_id] += 1
                else:
                    next_active.append(_Lane(lane.root_id, next_state, child_key, path))
            active = next_active
        return pending

    def _evaluate_pending(
        self,
        pending: list[_EvalRequest],
        leaf_evaluator: LeafEvaluator,
        done: np.ndarray,
        config: SearchConfig,
    ) -> None:
        if not pending:
            return
        unique: dict[StateKey, _EvalRequest] = {}
        for request in pending:
            if done[request.root_id] >= config.num_simulations:
                continue
            unique.setdefault(request.leaf_key, request)
        claim = self.store.claim_many_inflight(
            unique.keys(),
            ttl_ms=config.redis_inflight_ttl_ms,
        )
        claimed_requests = [unique[key] for key in claim.claimed if key in unique]
        for start in range(0, len(claimed_requests), config.eval_batch_size):
            batch = claimed_requests[start : start + config.eval_batch_size]
            if not batch:
                continue
            observations = jnp.stack([request.leaf_state.observation for request in batch], axis=0)
            logits, value_alpha, q_alpha = jax.device_get(leaf_evaluator(observations))
            nodes = []
            for ix, request in enumerate(batch):
                node = NodeBlob.expanded_node(
                    key=request.leaf_key,
                    current_player=_current_player(request.leaf_state),
                    legal_action_mask=_as_numpy(request.leaf_state.legal_action_mask),
                    value_alpha=value_alpha[ix],
                    policy_logits=logits[ix],
                    q_alpha=q_alpha[ix],
                    parent_key=request.path[-1].key,
                    parent_action=request.path[-1].action,
                    depth=_node_depth_for_parent(self.store, request.path[-1].key) + 1,
                )
                nodes.append(node)
            self.store.put_many(nodes)
            for request, node in zip(batch, nodes, strict=True):
                if done[request.root_id] >= config.num_simulations:
                    continue
                final = request.path[-1]
                update_edge_base_from_child(
                    self.store,
                    parent_key=final.key,
                    action=final.action,
                    child=node,
                )
                backup_path(
                    self.store,
                    path=request.path,
                    leaf_node=node,
                    leaf_value=_leaf_beta(node.value_alpha, config),
                    rng=self.rng,
                    backup_mc_samples=config.backup_mc_samples,
                    state_posterior_kappa_n=config.state_posterior_kappa_n,
                )
                done[request.root_id] += 1


def run_wavefront_posterior_tree_search(
    *,
    env: Any,
    root_states: list[Any],
    leaf_evaluator: LeafEvaluator,
    rng_key: jax.Array,
    config: Any,
    store: NodeStore | None = None,
) -> WavefrontPosteriorTreeBatchOutput:
    if store is None and getattr(config, "wavefront_backend", "arena") == "arena":
        from .arena_search import run_arena_posterior_tree_search

        result = run_arena_posterior_tree_search(
            env=env,
            root_states=root_states,
            leaf_evaluator=leaf_evaluator,
            rng_key=rng_key,
            config=config,
        )
        return WavefrontPosteriorTreeBatchOutput(
            action=result.action,
            action_weights=result.action_weights,
            beta_Q_target=result.beta_Q_target,
            beta_V_target=result.beta_V_target,
            q_loss_weight=result.q_loss_weight,
            alpha_root=result.alpha_root,
            trees=(),
            tree_data=result.tree_data,
            search_loss_mask=result.search_loss_mask,
            diagnostics=result.diagnostics,
            q_target_kind=result.q_target_kind,
            q_target_weight=result.q_target_weight,
            q_target_outcome=result.q_target_outcome,
            q_target_distance=result.q_target_distance,
            v_target_kind=result.v_target_kind,
            v_target_weight=result.v_target_weight,
            v_target_outcome=result.v_target_outcome,
            v_target_distance=result.v_target_distance,
        )

    search_config = search_config_from_any(config, num_roots=len(root_states))
    search = BatchedPosteriorSearch(env=env, store=store, rng_key=rng_key)
    result = search.search_batch(root_states, leaf_evaluator, search_config)
    return WavefrontPosteriorTreeBatchOutput(
        action=result.action,
        action_weights=result.action_weights,
        beta_Q_target=result.beta_Q_target,
        beta_V_target=result.beta_V_target,
        q_loss_weight=result.q_loss_weight,
        alpha_root=result.alpha_root,
        trees=(),
        tree_data=result.tree_data,
        search_loss_mask=result.search_loss_mask,
        diagnostics=result.diagnostics,
        q_target_kind=result.q_target_kind,
        q_target_weight=result.q_target_weight,
        q_target_outcome=result.q_target_outcome,
        q_target_distance=result.q_target_distance,
        v_target_kind=result.v_target_kind,
        v_target_weight=result.v_target_weight,
        v_target_outcome=result.v_target_outcome,
        v_target_distance=result.v_target_distance,
    )


def run_wavefront_posterior_tree_search_state_batch(
    *,
    env: Any,
    root_state_batch: Any,
    leaf_evaluator: LeafEvaluator,
    rng_key: jax.Array,
    config: Any,
    store: NodeStore | None = None,
) -> WavefrontPosteriorTreeBatchOutput:
    num_roots = int(np.asarray(jax.device_get(root_state_batch.current_player)).shape[0])
    if store is None and getattr(config, "wavefront_backend", "arena") == "arena":
        from .arena_search import BatchedPosteriorArenaSearch

        search_config = search_config_from_any(config, num_roots=num_roots)
        search = BatchedPosteriorArenaSearch(env=env, rng_key=rng_key)
        result = search.search_state_batch(
            root_state_batch,
            leaf_evaluator,
            search_config,
            max_nodes=getattr(config, "wavefront_max_nodes", None),
            max_edges=getattr(config, "wavefront_max_edges", None),
        )
        return WavefrontPosteriorTreeBatchOutput(
            action=result.action,
            action_weights=result.action_weights,
            beta_Q_target=result.beta_Q_target,
            beta_V_target=result.beta_V_target,
            q_loss_weight=result.q_loss_weight,
            alpha_root=result.alpha_root,
            trees=(),
            tree_data=result.tree_data,
            search_loss_mask=result.search_loss_mask,
            diagnostics=result.diagnostics,
            q_target_kind=result.q_target_kind,
            q_target_weight=result.q_target_weight,
            q_target_outcome=result.q_target_outcome,
            q_target_distance=result.q_target_distance,
            v_target_kind=result.v_target_kind,
            v_target_weight=result.v_target_weight,
            v_target_outcome=result.v_target_outcome,
            v_target_distance=result.v_target_distance,
        )

    return run_wavefront_posterior_tree_search(
        env=env,
        root_states=_split_batched_state(root_state_batch),
        leaf_evaluator=leaf_evaluator,
        rng_key=rng_key,
        config=config,
        store=store,
    )


def search_config_from_any(config: Any, *, num_roots: int = 1) -> SearchConfig:
    eval_batch_size = getattr(config, "search_eval_batch_size", None)
    if eval_batch_size is None:
        eval_batch_size = max(1, int(num_roots))
    final_action_mode = getattr(config, "wavefront_final_action_mode", "posterior_argmax")
    return SearchConfig(
        num_simulations=int(getattr(config, "num_simulations")),
        max_depth=int(getattr(config, "wavefront_max_depth", getattr(config, "max_depth", 128))),
        num_lanes_per_root=int(getattr(config, "wavefront_num_lanes_per_root", 1)),
        eval_batch_size=max(1, int(eval_batch_size)),
        leaf_value_mode=getattr(config, "leaf_value_mode", "alpha"),
        kappa_leaf=float(getattr(config, "kappa_leaf", 1.0)),
        kappa_terminal=float(getattr(config, "kappa_terminal", 8.0)),
        epsilon_terminal=float(getattr(config, "epsilon_terminal", 1e-6)),
        categorical_epsilon=float(getattr(config, "categorical_epsilon", 1e-4)),
        categorical_draw_rule=getattr(config, "categorical_draw_rule", "policy_prior"),
        state_posterior_kappa_n=float(getattr(config, "state_posterior_kappa_n", 9.0)),
        policy_mc_samples=int(getattr(config, "policy_mc_samples")),
        backup_mc_samples=int(getattr(config, "backup_mc_samples", getattr(config, "policy_mc_samples"))),
        duplicate_leaf_mode=getattr(config, "duplicate_leaf_mode", "recycle_lane"),
        final_action_mode=final_action_mode,
        pad_eval_batches=bool(getattr(config, "wavefront_pad_eval_batches", True)),
        pad_jax_select=bool(getattr(config, "wavefront_pad_jax_select", False)),
        np_select_below=max(0, int(getattr(config, "wavefront_np_select_below", 1024))),
        grouped_expansion=bool(getattr(config, "wavefront_grouped_expansion", True)),
        lane_indexed_step=bool(getattr(config, "wavefront_lane_indexed_step", True)),
        stable_lane_batch=bool(getattr(config, "wavefront_stable_lane_batch", True)),
        pad_pending_observation_gather=bool(
            getattr(config, "wavefront_pad_pending_observation_gather", True)
        ),
        train_tree_nodes=bool(getattr(config, "train_tree_nodes", False)),
        train_tree_include_root=bool(getattr(config, "train_tree_include_root", False)),
        train_tree_include_terminal=bool(getattr(config, "train_tree_include_terminal", False)),
        train_tree_min_q_evidence=float(getattr(config, "train_tree_min_q_evidence", 0.0)),
        train_tree_max_nodes_per_step=getattr(config, "train_tree_max_nodes_per_step", None),
    )


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


def _store_search_diagnostics(
    roots: list[NodeBlob],
    policies: np.ndarray,
    alpha_root: np.ndarray,
    config: SearchConfig,
) -> SearchDiagnostics:
    num_roots = len(roots)
    n_down = np.asarray(
        [
            float(np.sum(root.edge_eval_count_R, dtype=np.uint64))
            for root in roots
        ],
        dtype=np.float32,
    )
    expanded_nodes = np.asarray(
        [
            float(np.sum(np.asarray(root.edge_has_post, dtype=bool)))
            for root in roots
        ],
        dtype=np.float32,
    )
    gamma = n_down / (float(config.state_posterior_kappa_n) + n_down)
    policy = np.asarray(policies, dtype=np.float32)
    entropy = -np.sum(
        np.where(policy > 0.0, policy * np.log(np.maximum(policy, 1e-12)), 0.0),
        axis=-1,
    ).astype(np.float32)
    concentration = np.zeros((num_roots,), dtype=np.float32)
    for ix, root in enumerate(roots):
        if root.legal_actions.shape[0] == 0:
            continue
        legal_alpha = alpha_root[ix, root.legal_actions.astype(np.int32)]
        concentration[ix] = float(np.mean(np.sum(legal_alpha, axis=-1)))
    zeros = np.zeros((num_roots,), dtype=np.float32)
    return SearchDiagnostics(
        path_depth_mean=jnp.asarray(zeros),
        path_depth_p50=jnp.asarray(zeros),
        path_depth_p90=jnp.asarray(zeros),
        path_depth_max=jnp.asarray(zeros),
        expanded_nodes=jnp.asarray(expanded_nodes),
        terminal_fraction=jnp.asarray(zeros),
        root_policy_entropy=jnp.asarray(entropy),
        root_gamma=jnp.asarray(gamma, dtype=jnp.float32),
        root_downstream_eval_count=jnp.asarray(n_down, dtype=jnp.float32),
        root_q_concentration=jnp.asarray(concentration, dtype=jnp.float32),
    )


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


def _batched_key_fn(state: Any) -> jax.Array:
    key = type(state)
    fn = _KEY_CACHE.get(key)
    if fn is None:
        fn = jax.jit(jax.vmap(canonical_state_key))
        _KEY_CACHE[key] = fn
    return fn(state)


def _compute_keys(states: list[Any]) -> tuple[StateKey, ...]:
    return state_keys_to_host(_batched_key_fn(_stack_states(states)))


def _terminal_node_from_state(key: StateKey, state: Any, num_outcomes: int) -> NodeBlob:
    current = _current_player(state)
    reward = float(_as_numpy(state.rewards)[current])
    return NodeBlob.terminal_node(
        key=key,
        current_player=current,
        terminal_outcome=terminal_outcome_from_reward(reward, num_outcomes),
        num_outcomes=num_outcomes,
    )


def _leaf_beta(alpha_v: np.ndarray, config: SearchConfig) -> np.ndarray:
    alpha_v = np.maximum(np.asarray(alpha_v, dtype=np.float32), np.float32(1e-6))
    if config.leaf_value_mode == "alpha":
        return alpha_v
    if config.leaf_value_mode == "mean":
        return np.asarray(config.kappa_leaf, dtype=np.float32) * outcome_mean(alpha_v)
    raise ValueError(f"unknown leaf_value_mode: {config.leaf_value_mode!r}")


def _node_depth_for_parent(store: NodeStore, parent_key: StateKey) -> int:
    parent = store.get_many([parent_key]).get(parent_key)
    return 0 if parent is None else int(parent.depth)


def _stack_states(states: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _split_batched_state(state: Any) -> list[Any]:
    batch_size = int(_as_numpy(state.current_player).shape[0])
    return [jax.tree_util.tree_map(lambda x, ix=ix: x[ix], state) for ix in range(batch_size)]


def _as_numpy(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _current_player(state: Any) -> int:
    return int(_as_numpy(state.current_player).item())


def _terminated(state: Any) -> bool:
    return bool(_as_numpy(state.terminated).item())
