from __future__ import annotations

import re
from typing import Any, cast

import jax
import jax.numpy as jnp
import pgx
import pgx.gardner_chess
import pgx.go
import pgx.hex


_GARDNER_ROT90_INDEX = jnp.array(
    [
        [4, 9, 14, 19, 24],
        [3, 8, 13, 18, 23],
        [2, 7, 12, 17, 22],
        [1, 6, 11, 16, 21],
        [0, 5, 10, 15, 20],
    ],
    dtype=jnp.int32,
)


def _gardner_chess_observe_tpu_safe(state, player_id):
    gc = pgx.gardner_chess
    color = jax.lax.select(
        state.current_player == player_id,
        state._turn,
        1 - state._turn,
    )
    ones = jnp.ones((1, 5, 5), dtype=jnp.float32)
    state = jax.lax.cond(
        state.current_player == player_id,
        lambda: state,
        lambda: gc._flip(state),
    )

    def make(i):
        board = state._board_history[i][_GARDNER_ROT90_INDEX]

        def piece_feat(p):
            return (board == p).astype(jnp.float32)

        my_pieces = jax.vmap(piece_feat)(jnp.arange(1, 7))
        opp_pieces = jax.vmap(piece_feat)(-jnp.arange(1, 7))
        h = state._hash_history[i, :]
        rep = (state._hash_history == h).all(axis=1).sum() - 1
        rep = jax.lax.select((h == 0).all(), 0, rep)
        rep0 = ones * (rep == 0)
        rep1 = ones * (rep >= 1)
        return jnp.concatenate([my_pieces, opp_pieces, rep0, rep1], axis=0)

    board_feat = jax.vmap(make)(jnp.arange(8)).reshape(-1, 5, 5)
    color = color * ones
    total_move_cnt = (state._step_count / gc.MAX_TERMINATION_STEPS) * ones
    no_prog_cnt = (state._halfmove_count.astype(jnp.float32) / 100.0) * ones
    return jnp.concatenate(
        [board_feat, color, total_move_cnt, no_prog_cnt],
        axis=0,
    ).transpose((1, 2, 0))


class _TpuSafeGardnerChess(pgx.gardner_chess.GardnerChess):
    def _observe(self, state, player_id):
        assert isinstance(state, pgx.gardner_chess.State)
        return _gardner_chess_observe_tpu_safe(state, player_id)


def _parse_go_size(env_id: str) -> int | None:
    match = re.fullmatch(r"go_(\d+)x(\d+)", env_id)
    if match is None:
        return None
    lhs, rhs = match.groups()
    if lhs != rhs:
        return None
    return int(lhs)


def make_env(env_id: str, board_size: int | None = None):
    if env_id == "gardner_chess":
        return _TpuSafeGardnerChess()

    if env_id == "go":
        assert board_size is not None
        return pgx.go.Go(size=board_size, komi=7.5)

    go_size = _parse_go_size(env_id)
    if go_size is not None:
        return pgx.go.Go(size=go_size, komi=7.5)

    if env_id != "hex":
        return pgx.make(cast(Any, env_id))

    assert board_size is not None

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

    cast(Any, env)._init = _init
    cast(Any, env)._step = _step
    return env
