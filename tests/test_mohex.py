import jax
import jax.numpy as jnp
import pytest

from scacchi.envs import make_env
from scacchi.mohex import MOHEX_BINARY, MoHexProcess


def _build_state(env, seq):
    state = env.init(jax.random.PRNGKey(0))
    for action in seq:
        state = env.step(state, jnp.int32(action))
    return state


def _state_key(state):
    return (
        tuple(map(int, jax.device_get(state._x.board).tolist())),
        int(jax.device_get(state._x.color)),
        int(jax.device_get(state.current_player)),
        tuple(map(int, jax.device_get(state._player_order).tolist())),
    )


def _solve_state(env, state, cache):
    key = _state_key(state)
    if key in cache:
        return cache[key]

    current_player = int(jax.device_get(state.current_player))
    best_value = -2.0
    best_actions = []
    for action, is_legal in enumerate(jax.device_get(state.legal_action_mask)):
        if not is_legal:
            continue
        next_state = env.step(state, jnp.int32(action))
        reward = float(jax.device_get(next_state.rewards[current_player]))
        if bool(jax.device_get(next_state.terminated)):
            value = reward
        else:
            value = -_solve_state(env, next_state, cache)[0]
        if value > best_value + 1e-9:
            best_value = value
            best_actions = [action]
        elif abs(value - best_value) < 1e-9:
            best_actions.append(action)

    cache[key] = (best_value, tuple(best_actions))
    return cache[key]


@pytest.mark.skipif(not MOHEX_BINARY.exists(), reason="MoHex binary is not built")
def test_mohex_matches_perfect_play_on_solved_3x3_positions():
    env = make_env("hex", 3)
    cache = {}
    positions = [
        [2, 4, 6, 8, 1, 7, 5],
        [3, 5, 7, 0, 4, 1],
        [0, 2, 6, 1, 7, 5],
    ]

    mohex = MoHexProcess(str(MOHEX_BINARY))
    try:
        for seq in positions:
            state = _build_state(env, seq)
            value, optimal_actions = _solve_state(env, state, cache)
            mohex_action = mohex.genmove(state, 3)

            print(
                {
                    "seq": seq,
                    "value": value,
                    "optimal_actions": optimal_actions,
                    "mohex_action": mohex_action,
                }
            )

            assert bool(jax.device_get(state.legal_action_mask[mohex_action]))
            assert mohex_action in optimal_actions
    finally:
        mohex.close()
