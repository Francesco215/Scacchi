from typing import NamedTuple

import jax
import jax.numpy as jnp

from scacchi.evaluations import (
    _poison_eval_returns,
    _searchable_eval_state,
    _step_active_eval_rows,
)


class ToyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    terminated: jax.Array
    rewards: jax.Array


class ToyEnv:
    def step(self, state: ToyState, action: jax.Array) -> ToyState:
        reward = jnp.where(action == 1, 1.0, -10.0)
        return state._replace(
            observation=state.observation + 1.0,
            legal_action_mask=jnp.zeros_like(state.legal_action_mask),
            terminated=jnp.array(True),
            rewards=jnp.array([reward, -reward], dtype=state.rewards.dtype),
        )


def test_searchable_eval_state_adds_dummy_legal_action_only_for_terminal_rows():
    state = ToyState(
        observation=jnp.zeros((2, 1), dtype=jnp.float32),
        legal_action_mask=jnp.array(
            [
                [False, True, False],
                [False, False, False],
            ]
        ),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        terminated=jnp.array([False, True]),
        rewards=jnp.zeros((2, 2), dtype=jnp.float32),
    )

    search_state = _searchable_eval_state(state)

    assert jnp.array_equal(
        search_state.legal_action_mask,
        jnp.array(
            [
                [False, True, False],
                [True, False, False],
            ]
        ),
    )


def test_step_active_eval_rows_reports_invalid_actions_and_skips_step():
    state = ToyState(
        observation=jnp.array([[0.0], [5.0]], dtype=jnp.float32),
        legal_action_mask=jnp.array(
            [
                [False, True, False],
                [False, False, False],
            ]
        ),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        terminated=jnp.array([False, True]),
        rewards=jnp.array([[0.0, 0.0], [7.0, -7.0]], dtype=jnp.float32),
    )

    next_state, active, invalid_action = _step_active_eval_rows(
        ToyEnv(),
        state,
        jnp.array([2, 2], dtype=jnp.int32),
    )

    assert jnp.array_equal(active, jnp.array([True, False]))
    assert bool(invalid_action)
    assert jnp.array_equal(next_state.terminated, jnp.array([False, True]))
    assert jnp.array_equal(next_state.observation, jnp.array([[0.0], [5.0]]))
    assert jnp.array_equal(
        next_state.rewards,
        jnp.array([[0.0, 0.0], [7.0, -7.0]], dtype=jnp.float32),
    )


def test_step_active_eval_rows_ignores_invalid_terminal_row_actions():
    state = ToyState(
        observation=jnp.array([[0.0], [5.0]], dtype=jnp.float32),
        legal_action_mask=jnp.array(
            [
                [False, True, False],
                [False, False, False],
            ]
        ),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        terminated=jnp.array([False, True]),
        rewards=jnp.array([[0.0, 0.0], [7.0, -7.0]], dtype=jnp.float32),
    )

    next_state, active, invalid_action = _step_active_eval_rows(
        ToyEnv(),
        state,
        jnp.array([1, 99], dtype=jnp.int32),
    )

    assert jnp.array_equal(active, jnp.array([True, False]))
    assert not bool(invalid_action)
    assert jnp.array_equal(next_state.observation, jnp.array([[1.0], [5.0]]))


def test_poison_eval_returns_makes_all_returns_nan_after_invalid_action():
    returns = _poison_eval_returns(jnp.array([1.0, -1.0]), jnp.array(True))

    assert jnp.isnan(returns).all()
