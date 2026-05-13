from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
import pgx.hex


def make_env(env_id: str, board_size: int | None = None):
    if env_id != "hex" or board_size is None:
        return pgx.make(env_id)

    env = pgx.hex.Hex(size=board_size)

    def _init(key):
        player_order = jnp.array([[0, 1], [1, 0]])[
            jax.random.bernoulli(key).astype(jnp.int32)
        ]
        x = env._game.init()._replace(board=jnp.zeros(board_size * board_size, dtype=jnp.int32))
        return pgx.hex.State(
            current_player=player_order[x.color],
            legal_action_mask=jnp.ones(board_size * board_size + 1, dtype=jnp.bool_).at[-1].set(jnp.bool_(False)),
            _player_order=player_order,
            _x=x,
        )

    def _step(state, action, key):
        del key
        x = env._game.step(state._x, action)
        return state.replace(
            current_player=state._player_order[x.color],
            legal_action_mask=env._game.legal_action_mask(x).at[-1].set(jnp.bool_(False)),
            terminated=env._game.is_terminal(x),
            rewards=env._game.rewards(x)[state._player_order],
            _x=x,
        )

    env._init = _init
    env._step = _step
    return env
