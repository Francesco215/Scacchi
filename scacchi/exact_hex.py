from __future__ import annotations

from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .dirichlet_tree.native import (
    INF_DISTANCE,
    NO_DISTANCE,
    NO_OUTCOME,
    TARGET_CATEGORICAL,
    categorical_proxy_np,
    native_fields_from_beta,
)


_SOLVERS: dict[int, "_ExactHexSolver"] = {}


class _ExactHexSolver:
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.num_cells = self.size * self.size
        self.full_mask = (1 << self.num_cells) - 1
        self.neighbors = tuple(self._build_neighbors())
        center = (self.size - 1) / 2.0
        self.move_order = tuple(
            sorted(
                range(self.num_cells),
                key=lambda ix: abs(ix // self.size - center)
                + abs(ix % self.size - center),
            )
        )

    def _build_neighbors(self) -> list[tuple[int, ...]]:
        neighbors: list[tuple[int, ...]] = []
        for action in range(self.num_cells):
            row = action // self.size
            col = action % self.size
            adjacent: list[int] = []
            for dr, dc in ((0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1)):
                nr = row + dr
                nc = col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    adjacent.append(nr * self.size + nc)
            neighbors.append(tuple(adjacent))
        return neighbors

    def has_win(self, mask: int, color: int) -> bool:
        if color == 0:
            starts = [
                row * self.size
                for row in range(self.size)
                if mask & (1 << (row * self.size))
            ]

            def is_target(cell: int) -> bool:
                return cell % self.size == self.size - 1

        else:
            starts = [col for col in range(self.size) if mask & (1 << col)]

            def is_target(cell: int) -> bool:
                return cell // self.size == self.size - 1

        seen = 0
        stack = list(starts)
        for cell in starts:
            seen |= 1 << cell
        while stack:
            cell = stack.pop()
            if is_target(cell):
                return True
            for neighbor in self.neighbors[cell]:
                bit = 1 << neighbor
                if mask & bit and not seen & bit:
                    seen |= bit
                    stack.append(neighbor)
        return False

    @lru_cache(maxsize=None)
    def solve(self, my_mask: int, opp_mask: int, color: int) -> int:
        if self.has_win(opp_mask, 1 - color):
            return -1
        if self.has_win(my_mask, color):
            return 1
        occupied = my_mask | opp_mask
        if occupied == self.full_mask:
            return -1

        for action in self.move_order:
            action_bit = 1 << action
            if occupied & action_bit:
                continue
            next_my_mask = my_mask | action_bit
            if self.has_win(next_my_mask, color):
                return 1
            if -self.solve(opp_mask, next_my_mask, 1 - color) == 1:
                return 1
        return -1

    def action_values(self, my_mask: int, opp_mask: int, color: int) -> np.ndarray:
        values = np.full((self.num_cells,), -1, dtype=np.int8)
        occupied = my_mask | opp_mask
        for action in range(self.num_cells):
            action_bit = 1 << action
            if occupied & action_bit:
                continue
            next_my_mask = my_mask | action_bit
            values[action] = (
                1
                if self.has_win(next_my_mask, color)
                else -self.solve(opp_mask, next_my_mask, 1 - color)
            )
        return values


def _solver(size: int) -> _ExactHexSolver:
    size = int(size)
    if size not in _SOLVERS:
        _SOLVERS[size] = _ExactHexSolver(size)
    return _SOLVERS[size]


def _mask_from_plane(plane: np.ndarray) -> int:
    mask = 0
    for ix, occupied in enumerate(np.asarray(plane, dtype=bool).reshape(-1)):
        if bool(occupied):
            mask |= 1 << ix
    return mask


def _dirichlet_outcome(value: int, kappa: float, epsilon: float) -> np.ndarray:
    alpha = np.full((3,), float(epsilon), dtype=np.float32)
    if value > 0:
        alpha[2] += float(kappa)
    elif value < 0:
        alpha[0] += float(kappa)
    else:
        alpha[1] += float(kappa)
    return alpha


def _outcome_index(value: int) -> int:
    return int(np.clip(int(value) + 1, 0, 2))


def exact_hex_actions(
    obs: jax.Array,
    legal_action_mask: jax.Array,
    config: Any,
    rng_key: jax.Array | None = None,
) -> jax.Array:
    board_size = int(getattr(config, "board_size"))
    solver = _solver(board_size)
    num_cells = board_size * board_size
    obs_host = np.asarray(jax.device_get(obs))
    legal_host = np.asarray(jax.device_get(legal_action_mask), dtype=bool)
    flat_obs = obs_host.reshape((-1, *obs_host.shape[-3:]))
    flat_legal = legal_host.reshape((-1, legal_host.shape[-1]))
    actions = np.zeros((flat_obs.shape[0],), dtype=np.int32)
    sample = getattr(config, "final_action_mode", "posterior_argmax") == "posterior_sample"
    rng = None
    if sample and rng_key is not None:
        seed = int(
            jax.device_get(
                jax.random.randint(rng_key, (), 0, np.iinfo(np.int32).max)
            )
        )
        rng = np.random.default_rng(seed)

    for row, (row_obs, row_legal) in enumerate(zip(flat_obs, flat_legal, strict=True)):
        legal_cells = row_legal[:num_cells]
        if not bool(np.any(legal_cells)):
            legal_actions = np.nonzero(row_legal)[0]
            actions[row] = int(legal_actions[0]) if legal_actions.size else 0
            continue
        my_mask = _mask_from_plane(row_obs[..., 0])
        opp_mask = _mask_from_plane(row_obs[..., 1])
        color = int(bool(row_obs[0, 0, 2]))
        action_values = solver.action_values(my_mask, opp_mask, color)
        legal_values = np.where(legal_cells, action_values, -2)
        best = np.nonzero(legal_cells & (action_values == int(np.max(legal_values))))[0]
        if best.size == 0:
            best = np.nonzero(legal_cells)[0]
        if sample and rng is not None:
            actions[row] = int(best[int(rng.integers(best.size))])
        else:
            actions[row] = int(best[0])

    return jnp.asarray(actions.reshape(legal_host.shape[:-1]), dtype=jnp.int32)


def relabel_selfplay_with_exact_hex(
    data: Any,
    config: Any,
    rng_key: jax.Array | None = None,
) -> Any:
    board_size = int(getattr(config, "board_size"))
    if board_size > 4:
        raise ValueError("exact Hex bootstrap is intended only for board_size <= 4.")
    if int(getattr(config, "num_outcomes", 3)) != 3:
        raise ValueError("exact Hex bootstrap requires WDL3 targets.")

    solver = _solver(board_size)
    num_cells = board_size * board_size
    obs = np.asarray(jax.device_get(data.obs))
    legal = np.asarray(jax.device_get(data.legal_action_mask), dtype=bool)
    action_shape = legal.shape
    num_actions = int(action_shape[-1])
    policy = np.zeros((*action_shape,), dtype=np.float32)
    q_weight = np.zeros((*action_shape,), dtype=np.float32)
    beta_q = np.zeros((*action_shape, 3), dtype=np.float32)
    beta_v = np.zeros((*action_shape[:-1], 3), dtype=np.float32)
    search_loss_mask = np.zeros(action_shape[:-1], dtype=bool)
    categorical_epsilon = float(getattr(config, "categorical_epsilon", 1e-4))
    q_kind = np.zeros((*action_shape,), dtype=np.int8)
    q_target_weight = np.zeros((*action_shape,), dtype=np.float32)
    q_outcome = np.full((*action_shape,), int(NO_OUTCOME), dtype=np.int8)
    q_distance = np.full((*action_shape,), int(NO_DISTANCE), dtype=np.int32)
    v_kind = np.zeros(action_shape[:-1], dtype=np.int8)
    v_target_weight = np.zeros(action_shape[:-1], dtype=np.float32)
    v_outcome = np.full(action_shape[:-1], int(NO_OUTCOME), dtype=np.int8)
    v_distance = np.full(action_shape[:-1], int(NO_DISTANCE), dtype=np.int32)

    flat_obs = obs.reshape((-1, *obs.shape[-3:]))
    flat_legal = legal.reshape((-1, num_actions))
    flat_policy = policy.reshape((-1, num_actions))
    flat_q_weight = q_weight.reshape((-1, num_actions))
    flat_beta_q = beta_q.reshape((-1, num_actions, 3))
    flat_beta_v = beta_v.reshape((-1, 3))
    flat_search_mask = search_loss_mask.reshape((-1,))
    flat_q_kind = q_kind.reshape((-1, num_actions))
    flat_q_target_weight = q_target_weight.reshape((-1, num_actions))
    flat_q_outcome = q_outcome.reshape((-1, num_actions))
    flat_q_distance = q_distance.reshape((-1, num_actions))
    flat_v_kind = v_kind.reshape((-1,))
    flat_v_target_weight = v_target_weight.reshape((-1,))
    flat_v_outcome = v_outcome.reshape((-1,))
    flat_v_distance = v_distance.reshape((-1,))

    for row, (row_obs, row_legal) in enumerate(zip(flat_obs, flat_legal, strict=True)):
        legal_cells = row_legal[:num_cells]
        if not bool(np.any(legal_cells)):
            continue
        my_mask = _mask_from_plane(row_obs[..., 0])
        opp_mask = _mask_from_plane(row_obs[..., 1])
        color = int(bool(row_obs[0, 0, 2]))
        action_values = solver.action_values(my_mask, opp_mask, color)
        legal_values = np.where(legal_cells, action_values, -2)
        state_value = int(np.max(legal_values))
        best = legal_cells & (action_values == state_value)
        best_count = int(np.sum(best))
        if best_count <= 0:
            continue

        flat_policy[row, :num_cells] = best.astype(np.float32) / float(best_count)
        flat_q_weight[row, :num_cells] = flat_policy[row, :num_cells]
        state_outcome = _outcome_index(state_value)
        flat_beta_v[row] = categorical_proxy_np(
            state_outcome,
            3,
            epsilon=categorical_epsilon,
        )
        flat_v_kind[row] = int(TARGET_CATEGORICAL)
        flat_v_target_weight[row] = 1.0
        flat_v_outcome[row] = np.int8(state_outcome)
        flat_v_distance[row] = np.int32(INF_DISTANCE)
        flat_search_mask[row] = True
        for action in np.nonzero(legal_cells)[0]:
            action_outcome = _outcome_index(int(action_values[action]))
            flat_beta_q[row, action] = categorical_proxy_np(
                action_outcome,
                3,
                epsilon=categorical_epsilon,
            )
            flat_q_kind[row, action] = int(TARGET_CATEGORICAL)
            flat_q_target_weight[row, action] = 1.0
            flat_q_outcome[row, action] = np.int8(action_outcome)
            flat_q_distance[row, action] = np.int32(INF_DISTANCE)

    relabeled = data._replace(
        action_weights=jnp.asarray(policy, dtype=data.action_weights.dtype),
        beta_Q_target=jnp.asarray(beta_q, dtype=data.beta_Q_target.dtype),
        beta_V_target=jnp.asarray(beta_v, dtype=data.beta_V_target.dtype),
        q_loss_weight=jnp.asarray(q_weight, dtype=data.q_loss_weight.dtype),
        search_loss_mask=jnp.asarray(search_loss_mask),
        tree_data=None,
        q_target_kind=jnp.asarray(q_kind, dtype=jnp.int8),
        q_target_weight=jnp.asarray(q_target_weight, dtype=data.q_loss_weight.dtype),
        q_target_outcome=jnp.asarray(q_outcome, dtype=jnp.int8),
        q_target_distance=jnp.asarray(q_distance, dtype=jnp.int32),
        v_target_kind=jnp.asarray(v_kind, dtype=jnp.int8),
        v_target_weight=jnp.asarray(v_target_weight, dtype=data.beta_V_target.dtype),
        v_target_outcome=jnp.asarray(v_outcome, dtype=jnp.int8),
        v_target_distance=jnp.asarray(v_distance, dtype=jnp.int32),
    )
    extra_batch_size = int(getattr(config, "exact_hex_solver_extra_batch_size", 0))
    if extra_batch_size <= 0:
        return relabeled
    return _append_random_exact_samples(
        relabeled,
        config,
        solver,
        extra_batch_size,
        rng_key,
    )


def _append_random_exact_samples(
    data: Any,
    config: Any,
    solver: _ExactHexSolver,
    extra_batch_size: int,
    rng_key: jax.Array | None,
) -> Any:
    if rng_key is None:
        seed = 0
    else:
        seed = int(
            jax.device_get(
                jax.random.randint(rng_key, (), 0, np.iinfo(np.int32).max)
            )
        )
    rng = np.random.default_rng(seed)
    obs_dtype = data.obs.dtype
    target_dtype = data.action_weights.dtype
    time_steps = int(data.obs.shape[0])
    board_size = int(getattr(config, "board_size"))
    num_cells = board_size * board_size
    num_actions = int(data.action_weights.shape[-1])
    obs = np.zeros(
        (time_steps, extra_batch_size, board_size, board_size, 4),
        dtype=bool,
    )
    legal = np.zeros((time_steps, extra_batch_size, num_actions), dtype=bool)
    policy = np.zeros((time_steps, extra_batch_size, num_actions), dtype=np.float32)
    beta_q = np.zeros((time_steps, extra_batch_size, num_actions, 3), dtype=np.float32)
    beta_v = np.zeros((time_steps, extra_batch_size, 3), dtype=np.float32)
    q_weight = np.zeros((time_steps, extra_batch_size, num_actions), dtype=np.float32)
    search_loss_mask = np.zeros((time_steps, extra_batch_size), dtype=bool)
    played_action = np.zeros((time_steps, extra_batch_size), dtype=np.int32)
    categorical_epsilon = float(getattr(config, "categorical_epsilon", 1e-4))
    q_kind = np.zeros((time_steps, extra_batch_size, num_actions), dtype=np.int8)
    q_target_weight = np.zeros((time_steps, extra_batch_size, num_actions), dtype=np.float32)
    q_outcome = np.full(
        (time_steps, extra_batch_size, num_actions),
        int(NO_OUTCOME),
        dtype=np.int8,
    )
    q_distance = np.full(
        (time_steps, extra_batch_size, num_actions),
        int(NO_DISTANCE),
        dtype=np.int32,
    )
    v_kind = np.zeros((time_steps, extra_batch_size), dtype=np.int8)
    v_target_weight = np.zeros((time_steps, extra_batch_size), dtype=np.float32)
    v_outcome = np.full((time_steps, extra_batch_size), int(NO_OUTCOME), dtype=np.int8)
    v_distance = np.full((time_steps, extra_batch_size), int(NO_DISTANCE), dtype=np.int32)

    for t in range(time_steps):
        for b in range(extra_batch_size):
            color0_mask, color1_mask, color = _random_nonterminal_position(
                solver,
                rng,
            )
            if color == 0:
                my_mask, opp_mask = color0_mask, color1_mask
            else:
                my_mask, opp_mask = color1_mask, color0_mask
            _write_observation(obs[t, b], my_mask, opp_mask, color, board_size)
            legal_cells = _legal_cells(my_mask | opp_mask, num_cells)
            legal[t, b, :num_cells] = legal_cells
            action_values = solver.action_values(my_mask, opp_mask, color)
            legal_values = np.where(legal_cells, action_values, -2)
            state_value = int(np.max(legal_values))
            best = legal_cells & (action_values == state_value)
            best_count = int(np.sum(best))
            if best_count <= 0:
                continue
            policy[t, b, :num_cells] = best.astype(np.float32) / float(best_count)
            q_weight[t, b, :num_cells] = policy[t, b, :num_cells]
            state_outcome = _outcome_index(state_value)
            beta_v[t, b] = categorical_proxy_np(
                state_outcome,
                3,
                epsilon=categorical_epsilon,
            )
            v_kind[t, b] = int(TARGET_CATEGORICAL)
            v_target_weight[t, b] = 1.0
            v_outcome[t, b] = np.int8(state_outcome)
            v_distance[t, b] = np.int32(INF_DISTANCE)
            search_loss_mask[t, b] = True
            best_actions = np.nonzero(best)[0]
            played_action[t, b] = int(best_actions[rng.integers(best_actions.shape[0])])
            for action in np.nonzero(legal_cells)[0]:
                action_outcome = _outcome_index(int(action_values[action]))
                beta_q[t, b, action] = categorical_proxy_np(
                    action_outcome,
                    3,
                    epsilon=categorical_epsilon,
                )
                q_kind[t, b, action] = int(TARGET_CATEGORICAL)
                q_target_weight[t, b, action] = 1.0
                q_outcome[t, b, action] = np.int8(action_outcome)
                q_distance[t, b, action] = np.int32(INF_DISTANCE)

    def concat_batch(original: jax.Array, extra: np.ndarray) -> jax.Array:
        return jnp.concatenate(
            [original, jnp.asarray(extra, dtype=original.dtype)],
            axis=1,
        )

    native_defaults = native_fields_from_beta(data.beta_Q_target, data.beta_V_target)

    def concat_optional_native(name: str, extra: np.ndarray) -> jax.Array:
        original = getattr(data, name)
        if original is None:
            original = native_defaults[name]
        return concat_batch(original, extra)

    return data._replace(
        obs=concat_batch(data.obs, obs.astype(np.asarray(jax.device_get(data.obs)).dtype)).astype(obs_dtype),
        reward=concat_batch(data.reward, np.zeros((time_steps, extra_batch_size), dtype=np.float32)),
        terminated=concat_batch(
            data.terminated,
            np.zeros((time_steps, extra_batch_size), dtype=bool),
        ),
        action_weights=concat_batch(data.action_weights, policy.astype(np.float32)).astype(target_dtype),
        played_action=concat_batch(data.played_action, played_action),
        legal_action_mask=concat_batch(data.legal_action_mask, legal),
        beta_Q_target=concat_batch(data.beta_Q_target, beta_q.astype(np.float32)).astype(target_dtype),
        beta_V_target=concat_batch(data.beta_V_target, beta_v.astype(np.float32)).astype(target_dtype),
        q_loss_weight=concat_batch(data.q_loss_weight, q_weight.astype(np.float32)).astype(target_dtype),
        discount=concat_batch(
            data.discount,
            -np.ones((time_steps, extra_batch_size), dtype=np.float32),
        ),
        search_loss_mask=concat_batch(data.search_loss_mask, search_loss_mask),
        search_diagnostics=None,
        q_target_kind=concat_optional_native("q_target_kind", q_kind),
        q_target_weight=concat_optional_native("q_target_weight", q_target_weight),
        q_target_outcome=concat_optional_native("q_target_outcome", q_outcome),
        q_target_distance=concat_optional_native("q_target_distance", q_distance),
        v_target_kind=concat_optional_native("v_target_kind", v_kind),
        v_target_weight=concat_optional_native("v_target_weight", v_target_weight),
        v_target_outcome=concat_optional_native("v_target_outcome", v_outcome),
        v_target_distance=concat_optional_native("v_target_distance", v_distance),
    )


def _legal_cells(occupied_mask: int, num_cells: int) -> np.ndarray:
    return np.asarray(
        [not bool(occupied_mask & (1 << action)) for action in range(num_cells)],
        dtype=bool,
    )


def _write_observation(
    obs: np.ndarray,
    my_mask: int,
    opp_mask: int,
    color: int,
    board_size: int,
) -> None:
    for action in range(board_size * board_size):
        row = action // board_size
        col = action % board_size
        bit = 1 << action
        obs[row, col, 0] = bool(my_mask & bit)
        obs[row, col, 1] = bool(opp_mask & bit)
        obs[row, col, 2] = bool(color)
        obs[row, col, 3] = False


def _random_nonterminal_position(
    solver: _ExactHexSolver,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    color0_mask = 0
    color1_mask = 0
    color = 0
    max_prefix = int(rng.integers(0, solver.num_cells))
    for _ in range(max_prefix):
        occupied = color0_mask | color1_mask
        legal = [action for action in range(solver.num_cells) if not occupied & (1 << action)]
        if not legal:
            break
        action = int(legal[int(rng.integers(len(legal)))])
        bit = 1 << action
        if color == 0:
            next_mask = color0_mask | bit
            if solver.has_win(next_mask, 0):
                break
            color0_mask = next_mask
        else:
            next_mask = color1_mask | bit
            if solver.has_win(next_mask, 1):
                break
            color1_mask = next_mask
        color = 1 - color
    return color0_mask, color1_mask, color
