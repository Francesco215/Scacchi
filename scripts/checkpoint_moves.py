"""Print one Hex self-play move list from the latest saved checkpoint.

Example:
    uv run python scripts/checkpoint_moves.py checkpoints/hex5_chat_demo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import nnx
import jax
import numpy as np

from scacchi.checkpoint import config_from_checkpoint, from_pretrained
from scacchi.envs import make_env
from scacchi.play import play_training
from scacchi.play_search import make_search_player


def _column_name(index: int) -> str:
    if index < 0:
        raise ValueError(f"column index must be nonnegative; got {index}")
    name = ""
    while True:
        index, remainder = divmod(index, 26)
        name = chr(ord("A") + remainder) + name
        if index == 0:
            return name
        index -= 1


def hex_action_to_coordinate(action: int, board_size: int) -> str:
    if not 0 <= action < board_size * board_size:
        raise ValueError(
            f"Hex action {action} is outside a {board_size}x{board_size} board"
        )
    row, column = divmod(action, board_size)
    return f"{_column_name(column)}{row + 1}"


def _first_game_actions(
    actions: Any,
    terminated: Any,
) -> tuple[int, ...]:
    actions = np.asarray(jax.device_get(actions))
    terminated = np.asarray(jax.device_get(terminated), dtype=np.bool_)
    terminal_steps = np.flatnonzero(terminated)
    if terminal_steps.size == 0:
        raise RuntimeError(
            "self-play did not terminate within selfplay.max_num_steps"
        )
    end = int(terminal_steps[0]) + 1
    return tuple(int(action) for action in actions[:end])


def moves_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    seed: int = 0,
) -> tuple[str, ...]:
    config = config_from_checkpoint(checkpoint_path)
    if config.env.id != "hex" or config.env.board_size is None:
        raise ValueError(
            "checkpoint_moves.py currently supports Hex checkpoints only; "
            f"got env.id={config.env.id!r}"
        )

    env = make_env(config.env.id, config.env.board_size)
    model = from_pretrained(str(checkpoint_path), env, rngs=nnx.Rngs(seed))

    @nnx.jit
    def play_one_game(model: nnx.Module, rng_key: jax.Array):
        player = make_search_player(
            env,
            model,
            config.selfplay.search,
            config.selfplay.action_commitment,
            q_supervision_config=config.training.losses.q_supervision,
        )
        data = play_training(
            env,
            player,
            rng_key,
            batch_size=1,
            max_num_steps=int(config.selfplay.max_num_steps),
        )
        return data.played_action[0], data.terminated[0]

    actions, terminated = play_one_game(model, jax.random.PRNGKey(seed))
    game_actions = _first_game_actions(actions, terminated)
    return tuple(
        hex_action_to_coordinate(action, config.env.board_size)
        for action in game_actions
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the latest model in a Hex checkpoint directory, play one "
            "self-play game with its stored settings, and print JSON moves."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Orbax checkpoint directory containing numbered step folders",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="self-play RNG seed (default: 0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(json.dumps(moves_from_checkpoint(args.checkpoint, seed=args.seed)))


if __name__ == "__main__":
    main()
