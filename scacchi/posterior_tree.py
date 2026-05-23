from __future__ import annotations

from dataclasses import dataclass, field
import weakref
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .dirichlet_tree.types import TreeTrainingData


NO_CHILD = -1
POSTERIOR_TREE_POLICIES = {"posterior_tree", "posterior_tree_wavefront", "dirichlet_thompson"}


class EvalRequest(NamedTuple):
    tree_index: int
    leaf_id: int
    path: tuple[tuple[int, int], ...]
    state: Any


class StepRequest(NamedTuple):
    tree_index: int
    parent_id: int
    action: int
    path: tuple[tuple[int, int], ...]
    state: Any


class PosteriorTreeBatchOutput(NamedTuple):
    action: jax.Array
    action_weights: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_evidence_mass: jax.Array
    alpha_root: jax.Array
    trees: tuple["PosteriorTree", ...]
    tree_data: TreeTrainingData | None = None


@dataclass
class PosteriorNode:
    state: Any
    legal_action_mask: np.ndarray
    current_player: int
    terminal: bool
    parent: int = NO_CHILD
    action_from_parent: int = NO_CHILD
    expanded: bool = False
    in_flight: bool = False
    prior_logits: np.ndarray | None = None
    alpha_v: np.ndarray | None = None
    alpha_q: np.ndarray | None = None
    children: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    evidence: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    visits: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))


class PosteriorTree:
    def __init__(
        self,
        *,
        env: Any,
        root_state: Any,
        root_logits: np.ndarray,
        root_alpha_v: np.ndarray,
        root_alpha_q: np.ndarray,
        tree_index: int,
        rng: np.random.Generator,
        c_leaf: float,
        c_terminal: float,
        c_state: float,
        c_value_search: float,
        policy_mc_samples: int,
        backup_mc_samples: int,
        commit: str,
    ):
        self.env = env
        self.tree_index = tree_index
        self.rng = rng
        self.c_leaf = float(c_leaf)
        self.c_terminal = float(c_terminal)
        self.c_state = float(c_state)
        self.c_value_search = float(c_value_search)
        self.policy_mc_samples = int(policy_mc_samples)
        self.backup_mc_samples = int(backup_mc_samples)
        self.commit = commit
        self.num_actions = int(root_alpha_q.shape[-2])
        self.num_outcomes = int(root_alpha_q.shape[-1])
        self.done = 0
        self.inflight = 0
        self.nodes: list[PosteriorNode] = []
        self._add_node(
            state=root_state,
            parent=NO_CHILD,
            action_from_parent=NO_CHILD,
            expanded=True,
            in_flight=False,
            prior_logits=root_logits,
            alpha_v=root_alpha_v,
            alpha_q=root_alpha_q,
        )

    def _add_node(
        self,
        *,
        state: Any,
        parent: int,
        action_from_parent: int,
        expanded: bool,
        in_flight: bool,
        prior_logits: np.ndarray | None = None,
        alpha_v: np.ndarray | None = None,
        alpha_q: np.ndarray | None = None,
    ) -> int:
        node = PosteriorNode(
            state=state,
            legal_action_mask=_legal_action_mask(state, self.num_actions),
            current_player=_current_player(state),
            terminal=_terminated(state),
            parent=parent,
            action_from_parent=action_from_parent,
            expanded=expanded,
            in_flight=in_flight,
            prior_logits=None if prior_logits is None else _as_numpy(prior_logits),
            alpha_v=None if alpha_v is None else _positive_alpha(_as_numpy(alpha_v)),
            alpha_q=None if alpha_q is None else _positive_alpha(_as_numpy(alpha_q)),
            children=np.full((self.num_actions,), NO_CHILD, dtype=np.int32),
            evidence=np.zeros((self.num_actions, self.num_outcomes), dtype=np.float32),
            visits=np.zeros((self.num_actions,), dtype=np.int32),
        )
        self.nodes.append(node)
        return len(self.nodes) - 1

    def edge_base(self, node_id: int, action: int) -> np.ndarray:
        node = self.nodes[node_id]
        child_id = int(node.children[action])
        if child_id != NO_CHILD:
            child = self.nodes[child_id]
            if child.expanded and not child.terminal and child.alpha_v is not None:
                return _positive_alpha(self._align(child_id, node_id, child.alpha_v))
        if node.alpha_q is None:
            raise ValueError("expanded node is missing alpha_q")
        return _positive_alpha(node.alpha_q[action])

    def edge_posterior(self, node_id: int, action: int) -> np.ndarray:
        node = self.nodes[node_id]
        return _positive_alpha(self.edge_base(node_id, action) + node.evidence[action])

    def thompson_select(self, node_id: int) -> int:
        node = self.nodes[node_id]
        scores = np.full((self.num_actions,), -np.inf, dtype=np.float64)
        legal_actions = np.flatnonzero(node.legal_action_mask)
        if legal_actions.size == 0:
            raise ValueError("cannot select from a node with no legal actions")
        for action in legal_actions:
            phi = self.rng.dirichlet(self.edge_posterior(node_id, int(action)))
            scores[action] = outcome_utility_np(phi)
        return int(np.argmax(scores))

    def state_search_posterior(self, node_id: int) -> np.ndarray:
        node = self.nodes[node_id]
        if node.terminal or not node.expanded:
            if node.alpha_v is None:
                return np.ones((self.num_outcomes,), dtype=np.float32)
            return _positive_alpha(node.alpha_v)
        alpha = np.stack(
            [self.edge_posterior(node_id, action) for action in range(self.num_actions)],
            axis=0,
        )
        policy = posterior_best_policy_target_np(
            self.rng,
            alpha,
            node.legal_action_mask,
            self.backup_mc_samples,
        )
        return _positive_alpha(np.sum(policy[:, None] * alpha, axis=0))

    def next_step_request(self) -> StepRequest | None:
        node_id = 0
        path: list[tuple[int, int]] = []
        while True:
            node = self.nodes[node_id]
            if node.terminal:
                d_leaf = self._terminal_outcome(node_id)
                self.backup_path(tuple(path), node_id, d_leaf, self.c_terminal)
                self.done += 1
                return None
            if not node.expanded:
                return None

            action = self.thompson_select(node_id)
            path.append((node_id, action))
            child_id = int(node.children[action])
            if child_id != NO_CHILD:
                child = self.nodes[child_id]
                if child.terminal or child.expanded:
                    node_id = child_id
                    continue
                return None

            return StepRequest(self.tree_index, node_id, action, tuple(path), node.state)

    def consume_step_result(self, request: StepRequest, child_state: Any) -> EvalRequest | None:
        parent = self.nodes[request.parent_id]
        child_id = self._add_node(
            state=child_state,
            parent=request.parent_id,
            action_from_parent=request.action,
            expanded=False,
            in_flight=False,
        )
        parent.children[request.action] = child_id
        child = self.nodes[child_id]
        if child.terminal:
            d_leaf = self._terminal_outcome(child_id)
            self.backup_path(request.path, child_id, d_leaf, self.c_terminal)
            self.done += 1
            return None

        child.in_flight = True
        self.inflight += 1
        return EvalRequest(self.tree_index, child_id, request.path, child_state)

    def next_request(self) -> EvalRequest | None:
        request = self.next_step_request()
        if request is None:
            return None
        child_state = self.env.step(request.state, jnp.asarray(request.action, dtype=jnp.int32))
        return self.consume_step_result(request, child_state)

    def consume_result(
        self,
        request: EvalRequest,
        *,
        logits: np.ndarray,
        alpha_v: np.ndarray,
        alpha_q: np.ndarray,
    ) -> None:
        node = self.nodes[request.leaf_id]
        node.prior_logits = _as_numpy(logits)
        node.alpha_v = _positive_alpha(_as_numpy(alpha_v))
        node.alpha_q = _positive_alpha(_as_numpy(alpha_q))
        node.expanded = True
        if node.in_flight:
            node.in_flight = False
            self.inflight -= 1
        d_leaf = outcome_mean_np(node.alpha_v)
        self.backup_path(request.path, request.leaf_id, d_leaf, self.c_leaf)
        self.done += 1

    def backup_path(
        self,
        path: tuple[tuple[int, int], ...],
        leaf_id: int,
        d_leaf: np.ndarray,
        leaf_weight: float,
    ) -> None:
        if not path:
            return

        final_parent, final_action = path[-1]
        final_node = self.nodes[final_parent]
        d_parent = self._align(leaf_id, final_parent, d_leaf)
        final_node.evidence[final_action] += np.asarray(leaf_weight, dtype=np.float32) * d_parent
        final_node.visits[final_action] += 1

        for parent_id, action in reversed(path[:-1]):
            parent = self.nodes[parent_id]
            child_id = int(parent.children[action])
            if child_id == NO_CHILD:
                continue
            beta_child = self.state_search_posterior(child_id)
            parent.evidence[action] += np.asarray(self.c_state, dtype=np.float32) * self._align(
                child_id,
                parent_id,
                beta_child,
            )
            parent.visits[action] += 1

    def finish(self) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
        root = self.nodes[0]
        alpha_root = np.stack(
            [self.edge_posterior(0, action) for action in range(self.num_actions)],
            axis=0,
        )
        policy_target = posterior_best_policy_target_np(
            self.rng,
            alpha_root,
            root.legal_action_mask,
            self.policy_mc_samples,
        )
        action = self._commit_action(policy_target, alpha_root, root.legal_action_mask)
        q_evidence_mass = np.sum(root.evidence, axis=-1)
        beta_q = alpha_root
        value_proxy = np.sum(policy_target[:, None] * alpha_root, axis=0)
        if root.alpha_v is None:
            raise ValueError("root is missing alpha_v")
        beta_v = _positive_alpha(root.alpha_v + self.c_value_search * value_proxy)
        return action, policy_target, beta_q, beta_v, q_evidence_mass, alpha_root

    def _commit_action(
        self,
        policy_target: np.ndarray,
        alpha_root: np.ndarray,
        legal_action_mask: np.ndarray,
    ) -> int:
        mode = "posterior_argmax" if self.commit == "posterior_best" else self.commit
        if mode == "posterior_sample":
            probs = np.where(legal_action_mask, policy_target, 0.0)
            prob_sum = float(np.sum(probs))
            if prob_sum <= 0:
                probs = legal_action_mask.astype(np.float64)
                prob_sum = float(np.sum(probs))
            return int(self.rng.choice(self.num_actions, p=probs / prob_sum))
        if mode == "scalar_q_argmax" or mode == "search_action":
            scores = outcome_utility_np(outcome_mean_np(alpha_root))
            return int(np.argmax(np.where(legal_action_mask, scores, -np.inf)))
        if mode == "posterior_argmax":
            return int(np.argmax(np.where(legal_action_mask, policy_target, -np.inf)))
        raise ValueError(f"unknown selfplay_action_source: {self.commit!r}")

    def _terminal_outcome(self, node_id: int) -> np.ndarray:
        node = self.nodes[node_id]
        rewards = _as_numpy(node.state.rewards)
        reward = float(rewards[node.current_player])
        return terminal_outcome_from_reward_np(reward, self.num_outcomes)

    def _align(self, source_node_id: int, target_node_id: int, value: np.ndarray) -> np.ndarray:
        source = self.nodes[source_node_id]
        target = self.nodes[target_node_id]
        if source.current_player != target.current_player:
            return flip_outcome_np(value)
        return np.asarray(value, dtype=np.float32)


LeafEvaluator = Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]]
BatchedStep = Callable[[Any, jax.Array], Any]


_BATCHED_STEP_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, BatchedStep]] = {}


def is_posterior_tree_policy(search_policy: str) -> bool:
    return search_policy in POSTERIOR_TREE_POLICIES


def run_posterior_tree_search(
    *,
    env: Any,
    root_states: list[Any],
    leaf_evaluator: LeafEvaluator,
    rng_key: jax.Array,
    config: Any,
) -> PosteriorTreeBatchOutput:
    if getattr(config, "search_policy", "posterior_tree") == "posterior_tree_wavefront":
        from .dirichlet_tree.search import run_wavefront_posterior_tree_search

        return run_wavefront_posterior_tree_search(
            env=env,
            root_states=root_states,
            leaf_evaluator=leaf_evaluator,
            rng_key=rng_key,
            config=config,
        )

    if not root_states:
        raise ValueError("root_states must not be empty")

    root_observations = jnp.stack([state.observation for state in root_states], axis=0)
    root_logits, root_alpha_v, root_alpha_q = leaf_evaluator(root_observations)
    root_logits, root_alpha_v, root_alpha_q = jax.device_get(
        (root_logits, root_alpha_v, root_alpha_q)
    )

    seed = int(
        jax.device_get(
            jax.random.randint(rng_key, (), minval=0, maxval=np.iinfo(np.int32).max)
        )
    )
    rng = np.random.default_rng(seed)
    trees = tuple(
        PosteriorTree(
            env=env,
            root_state=state,
            root_logits=root_logits[ix],
            root_alpha_v=root_alpha_v[ix],
            root_alpha_q=root_alpha_q[ix],
            tree_index=ix,
            rng=rng,
            c_leaf=getattr(config, "c_leaf"),
            c_terminal=getattr(config, "c_terminal"),
            c_state=getattr(config, "c_state", 0.1),
            c_value_search=getattr(config, "c_value_search", 1.0),
            policy_mc_samples=getattr(config, "policy_mc_samples"),
            backup_mc_samples=getattr(config, "backup_mc_samples", getattr(config, "policy_mc_samples")),
            commit=getattr(config, "selfplay_action_source"),
        )
        for ix, state in enumerate(root_states)
    )

    _run_search_loop(
        env,
        trees,
        leaf_evaluator=leaf_evaluator,
        num_simulations=int(getattr(config, "num_simulations")),
        eval_batch_size=_eval_batch_size(config, len(trees)),
        inflight_limit=int(getattr(config, "inflight_limit", 1)),
    )

    finished = [tree.finish() for tree in trees]
    actions, policies, beta_q, beta_v, q_mass, alpha_root = zip(*finished, strict=True)
    return PosteriorTreeBatchOutput(
        action=_device_put_cpu(jnp.asarray(np.asarray(actions), dtype=jnp.int32)),
        action_weights=_device_put_cpu(jnp.asarray(np.stack(policies, axis=0))),
        beta_Q_target=_device_put_cpu(jnp.asarray(np.stack(beta_q, axis=0))),
        beta_V_target=_device_put_cpu(jnp.asarray(np.stack(beta_v, axis=0))),
        q_evidence_mass=_device_put_cpu(jnp.asarray(np.stack(q_mass, axis=0))),
        alpha_root=_device_put_cpu(jnp.asarray(np.stack(alpha_root, axis=0))),
        trees=trees,
        tree_data=None,
    )


def run_posterior_tree_search_state_batch(
    *,
    env: Any,
    root_state_batch: Any,
    leaf_evaluator: LeafEvaluator,
    rng_key: jax.Array,
    config: Any,
) -> PosteriorTreeBatchOutput:
    if getattr(config, "search_policy", "posterior_tree") == "posterior_tree_wavefront":
        from .dirichlet_tree.search import run_wavefront_posterior_tree_search_state_batch

        return run_wavefront_posterior_tree_search_state_batch(
            env=env,
            root_state_batch=root_state_batch,
            leaf_evaluator=leaf_evaluator,
            rng_key=rng_key,
            config=config,
        )

    return run_posterior_tree_search(
        env=env,
        root_states=split_batched_state(root_state_batch),
        leaf_evaluator=leaf_evaluator,
        rng_key=rng_key,
        config=config,
    )


def _run_search_loop(
    env: Any,
    trees: tuple[PosteriorTree, ...],
    *,
    leaf_evaluator: LeafEvaluator,
    num_simulations: int,
    eval_batch_size: int,
    inflight_limit: int,
) -> None:
    while any(tree.done < num_simulations for tree in trees):
        step_requests, made_progress = _build_step_batch(
            trees,
            num_simulations=num_simulations,
            inflight_limit=inflight_limit,
        )
        if not any(request is not None for request in step_requests):
            if made_progress:
                continue
            if all(tree.done >= num_simulations for tree in trees):
                break
            unfinished = [tree.tree_index for tree in trees if tree.done < num_simulations]
            raise RuntimeError(f"posterior tree search stalled for roots {unfinished}")

        active_requests = [request for request in step_requests if request is not None]
        fallback_request = active_requests[0]
        states = [
            request.state if request is not None else fallback_request.state
            for request in step_requests
        ]
        actions = jnp.asarray(
            [
                request.action if request is not None else fallback_request.action
                for request in step_requests
            ],
            dtype=jnp.int32,
        )
        active_mask = jnp.asarray([request is not None for request in step_requests], dtype=bool)
        batched_states = _device_put_cpu(_stack_states(states))
        actions = _device_put_cpu(actions)
        active_mask = _device_put_cpu(active_mask)
        stepped_batch = _batched_step(env)(batched_states, actions)
        stepped_batch = _select_active_states(stepped_batch, batched_states, active_mask)
        stepped_states = split_batched_state(stepped_batch)

        eval_requests: list[EvalRequest] = []
        for request, child_state in zip(step_requests, stepped_states, strict=True):
            if request is None:
                continue
            eval_request = trees[request.tree_index].consume_step_result(request, child_state)
            if eval_request is not None:
                eval_requests.append(eval_request)

        _consume_eval_requests(
            trees,
            eval_requests,
            leaf_evaluator=leaf_evaluator,
            eval_batch_size=eval_batch_size,
        )


def _build_step_batch(
    trees: tuple[PosteriorTree, ...],
    *,
    num_simulations: int,
    inflight_limit: int,
) -> tuple[list[StepRequest | None], bool]:
    requests: list[StepRequest | None] = []
    made_progress = False
    for tree in trees:
        if tree.done >= num_simulations or tree.inflight >= inflight_limit:
            requests.append(None)
            continue
        before = (tree.done, tree.inflight, len(tree.nodes))
        request = tree.next_step_request()
        after = (tree.done, tree.inflight, len(tree.nodes))
        requests.append(request)
        if request is not None or after != before:
            made_progress = True
    return requests, made_progress


def _consume_eval_requests(
    trees: tuple[PosteriorTree, ...],
    requests: list[EvalRequest],
    *,
    leaf_evaluator: LeafEvaluator,
    eval_batch_size: int,
) -> None:
    for start in range(0, len(requests), eval_batch_size):
        batch = requests[start : start + eval_batch_size]
        observations = jnp.stack([request.state.observation for request in batch], axis=0)
        logits, alpha_v, alpha_q = leaf_evaluator(observations)
        logits, alpha_v, alpha_q = jax.device_get((logits, alpha_v, alpha_q))
        for ix, request in enumerate(batch):
            trees[request.tree_index].consume_result(
                request,
                logits=logits[ix],
                alpha_v=alpha_v[ix],
                alpha_q=alpha_q[ix],
            )


def _batched_step(env: Any) -> BatchedStep:
    cache_key = id(env)
    cached = _BATCHED_STEP_CACHE.get(cache_key)
    if cached is not None:
        env_ref, step = cached
        if env_ref is None or env_ref() is env:
            return step
    try:
        env_ref = weakref.ref(env)
    except TypeError:
        env_ref = None
    step = jax.jit(jax.vmap(env.step))
    _BATCHED_STEP_CACHE[cache_key] = (env_ref, step)
    return step


def posterior_best_policy_target_np(
    rng: np.random.Generator,
    alpha: np.ndarray,
    legal_action_mask: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    target = np.zeros((alpha.shape[-2],), dtype=np.float32)
    legal_actions = np.flatnonzero(legal_action_mask)
    if legal_actions.size == 0:
        return target
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    legal_alpha = _positive_alpha(alpha[legal_actions])
    hits = np.zeros((legal_actions.size,), dtype=np.float32)
    for _ in range(num_samples):
        sampled = np.stack([rng.dirichlet(legal_alpha[ix]) for ix in range(legal_actions.size)])
        best = int(np.argmax(outcome_utility_np(sampled)))
        hits[best] += 1.0
    target[legal_actions] = hits / float(num_samples)
    target_sum = float(np.sum(target))
    if target_sum <= 0:
        target[legal_actions] = 1.0 / float(legal_actions.size)
    else:
        target /= target_sum
    return target


def outcome_mean_np(alpha: np.ndarray) -> np.ndarray:
    alpha = _positive_alpha(alpha)
    return alpha / np.sum(alpha, axis=-1, keepdims=True)


def outcome_utility_np(outcome_dist: np.ndarray) -> np.ndarray:
    return outcome_dist[..., -1] - outcome_dist[..., 0]


def flip_outcome_np(outcome_dist: np.ndarray) -> np.ndarray:
    return np.asarray(outcome_dist, dtype=np.float32)[..., ::-1]


def terminal_outcome_from_reward_np(reward: float, num_outcomes: int) -> np.ndarray:
    rounded = int(np.rint(reward))
    if num_outcomes == 2:
        index = (rounded + 1) // 2
    elif num_outcomes == 3:
        index = rounded + 1
    else:
        raise ValueError(f"unsupported outcome count: {num_outcomes}")
    index = int(np.clip(index, 0, num_outcomes - 1))
    outcome = np.zeros((num_outcomes,), dtype=np.float32)
    outcome[index] = 1.0
    return outcome


def _stack_states(states: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _cpu_device() -> jax.Device:
    try:
        return jax.devices("cpu")[0]
    except RuntimeError as exc:
        raise RuntimeError(
            "posterior_tree CPU env stepping requires the JAX CPU platform. "
            "Use JAX_PLATFORMS=cuda,cpu when running with a GPU."
        ) from exc


def _device_put_cpu(value: Any) -> Any:
    return jax.device_put(value, _cpu_device())


def _select_active_states(
    stepped_state: Any,
    original_state: Any,
    active_mask: jax.Array,
) -> Any:
    def select_leaf(stepped_leaf: jax.Array, original_leaf: jax.Array) -> jax.Array:
        mask = jnp.reshape(
            active_mask,
            active_mask.shape + (1,) * (stepped_leaf.ndim - 1),
        )
        return jnp.where(mask, stepped_leaf, original_leaf)

    return jax.tree_util.tree_map(select_leaf, stepped_state, original_state)


def split_batched_state(state: Any) -> list[Any]:
    batch_size = int(_as_numpy(state.current_player).shape[0])
    return [jax.tree_util.tree_map(lambda x, ix=ix: x[ix], state) for ix in range(batch_size)]


def _eval_batch_size(config: Any, num_trees: int) -> int:
    configured = getattr(config, "search_eval_batch_size", None)
    if configured is None:
        return max(1, num_trees)
    return max(1, int(configured))


def _as_numpy(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _positive_alpha(alpha: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(alpha, dtype=np.float32), np.float32(1e-6))


def _current_player(state: Any) -> int:
    return int(_as_numpy(state.current_player).item())


def _terminated(state: Any) -> bool:
    return bool(_as_numpy(state.terminated).item())


def _legal_action_mask(state: Any, num_actions: int) -> np.ndarray:
    if _terminated(state):
        return np.zeros((num_actions,), dtype=bool)
    return np.asarray(_as_numpy(state.legal_action_mask), dtype=bool)
