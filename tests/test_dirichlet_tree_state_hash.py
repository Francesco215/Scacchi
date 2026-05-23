from typing import NamedTuple

import jax
import jax.numpy as jnp

from scacchi.dirichlet_tree.state_hash import canonical_state_key
from scacchi.envs import make_env


class ToyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    terminated: jax.Array
    rewards: jax.Array
    rule_state: jax.Array


def _state(*, obs=0, player=0, rule=0):
    return ToyState(
        observation=jnp.array([obs], dtype=jnp.int32),
        legal_action_mask=jnp.array([True, True]),
        current_player=jnp.array(player, dtype=jnp.int32),
        terminated=jnp.array(False),
        rewards=jnp.array([0.0, 0.0], dtype=jnp.float32),
        rule_state=jnp.array([rule], dtype=jnp.int32),
    )


def test_state_hash_is_stable_for_same_state():
    key_a = canonical_state_key(_state(obs=3, player=1, rule=2))
    key_b = canonical_state_key(_state(obs=3, player=1, rule=2))

    assert (key_a == key_b).all()


def test_state_hash_changes_for_player_board_and_rule_state():
    base = canonical_state_key(_state(obs=3, player=0, rule=2))

    assert not (base == canonical_state_key(_state(obs=3, player=1, rule=2))).all()
    assert not (base == canonical_state_key(_state(obs=4, player=0, rule=2))).all()
    assert not (base == canonical_state_key(_state(obs=3, player=0, rule=9))).all()


def test_state_hash_gives_same_child_key_for_toy_transposition():
    child_from_path_a = _state(obs=7, player=1, rule=5)
    child_from_path_b = _state(obs=7, player=1, rule=5)

    assert (canonical_state_key(child_from_path_a) == canonical_state_key(child_from_path_b)).all()


def test_hex_state_hash_uses_board_player_and_player_order():
    env = make_env("hex", 5)
    state = env.init(jax.random.PRNGKey(0))
    changed_board = state.replace(_x=state._x._replace(board=state._x.board.at[0].set(1)))
    changed_player = state.replace(current_player=1 - state.current_player)
    changed_order = state.replace(_player_order=state._player_order[::-1])
    base = canonical_state_key(state)

    assert not (base == canonical_state_key(changed_board)).all()
    assert not (base == canonical_state_key(changed_player)).all()
    assert not (base == canonical_state_key(changed_order)).all()
