import jax.numpy as jnp
import pytest

from scripts.checkpoint_moves import (
    _first_game_actions,
    hex_action_to_coordinate,
)


@pytest.mark.parametrize(
    ("action", "coordinate"),
    [
        (0, "A1"),
        (4, "E1"),
        (5, "A2"),
        (12, "C3"),
        (24, "E5"),
    ],
)
def test_hex_action_to_coordinate(action: int, coordinate: str) -> None:
    assert hex_action_to_coordinate(action, 5) == coordinate


def test_hex_action_to_coordinate_rejects_out_of_board_action() -> None:
    with pytest.raises(ValueError, match="outside"):
        hex_action_to_coordinate(25, 5)


def test_first_game_actions_stops_at_first_terminal_frame() -> None:
    actions = jnp.array([5, 12, 24, 0], dtype=jnp.int32)
    terminated = jnp.array([False, False, True, False])

    assert _first_game_actions(actions, terminated) == (5, 12, 24)


def test_first_game_actions_requires_a_complete_game() -> None:
    with pytest.raises(RuntimeError, match="did not terminate"):
        _first_game_actions(
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([False, False]),
        )
