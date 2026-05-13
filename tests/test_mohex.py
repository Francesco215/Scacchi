from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

import scacchi.mohex as mohex_module
from scacchi.envs import make_env
from scacchi.mohex import MOHEX_BINARY, MoHexProcess, _state_to_sgf


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


def test_state_to_sgf_uses_internal_hex_color_not_external_player_id():
    env = make_env("hex", 3)
    state = env.init(jax.random.PRNGKey(0))
    state = state.replace(_player_order=jnp.array([1, 0]), current_player=jnp.int32(1))

    state = env.step(state, jnp.int32(0))
    state = env.step(state, jnp.int32(4))

    assert int(jax.device_get(state.current_player)) == 1
    assert int(jax.device_get(state._x.color)) == 0
    assert _state_to_sgf(state, 3) == "(;AP[Scacchi]FF[4]GM[11]SZ[3];B[a1];W[b2])"


def test_hex_env_disables_swap_rule():
    env = make_env("hex", 3)
    state = env.init(jax.random.PRNGKey(0))
    state = env.step(state, jnp.int32(4))

    assert not bool(jax.device_get(state.legal_action_mask[-1]))


def test_mohex_selector_concedes_after_retry_failure(monkeypatch, tmp_path):
    binary = tmp_path / "mohex"
    binary.write_text("")
    monkeypatch.setattr(mohex_module, "mohex_binary", lambda: binary)

    class FailingMoHex:
        def __init__(self, *args, **kwargs):
            pass

        def genmove(self, state, board_size):
            raise ValueError("bad move")

        def close(self):
            pass

    monkeypatch.setattr(mohex_module, "MoHexProcess", FailingMoHex)

    env = make_env("hex", 3)
    state = env.init(jax.random.PRNGKey(0))
    current_player = int(jax.device_get(state.current_player))
    batch_state = jax.tree_util.tree_map(lambda x: x[None], state)
    config = SimpleNamespace(
        board_size=3,
        mohex_num_processes=1,
        mohex_num_threads=1,
        mohex_dfpn_threads=1,
    )
    choose, close = mohex_module.make_mohex_action_selector(
        env,
        config,
        my_player=1 - current_player,
    )
    try:
        actions = choose(batch_state)
        chosen = int(jax.device_get(actions[0]))
        assert bool(jax.device_get(state.legal_action_mask[chosen]))
    finally:
        close()


@pytest.mark.skipif(not MOHEX_BINARY.exists(), reason="MoHex binary is not built")
def test_mohex_genmove_crashes_on_known_fillin_position():
    # Known benzene-vanilla bug: this position kills reg_genmove with
    # "Fillin caused win!". If MoHex stops crashing, this test will fail.
    env = make_env("hex", 5)
    state = _build_state(env, [8, 1, 17, 19])

    mohex = MoHexProcess(
        str(MOHEX_BINARY),
        max_memory=67108864,
        max_time=1.0,
        max_games=1000,
        num_threads=1,
        dfpn_threads=1,
        parallel_solver=False,
    )
    try:
        with pytest.raises(RuntimeError, match="MoHex exited unexpectedly"):
            mohex.genmove(state, 5)
    finally:
        mohex.close()


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
