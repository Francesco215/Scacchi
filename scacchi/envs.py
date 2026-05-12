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

    env._init = _init
    return env