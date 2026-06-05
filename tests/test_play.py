import jax
import jax.numpy as jnp
import pytest

from scacchi.dirichlet_q_search import posterior_sample_action
from scacchi.play_search import _legalize_played_action, _select_played_action


def test_select_played_action_samples_posterior_target():
    key = jax.random.PRNGKey(0)
    action_weights = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    legal_action_mask = jnp.ones_like(action_weights, dtype=jnp.bool_)
    search_action = jnp.array([0, 0], dtype=jnp.int32)

    played_action = _select_played_action(
        "posterior_sample",
        key,
        action_weights,
        legal_action_mask,
        search_action,
    )

    expected = posterior_sample_action(key, action_weights, legal_action_mask)
    assert jnp.array_equal(played_action, expected)
    assert not jnp.array_equal(played_action, search_action)


def test_select_played_action_can_use_search_action():
    action_weights = jnp.array([[0.0, 1.0, 0.0]])
    legal_action_mask = jnp.ones_like(action_weights, dtype=jnp.bool_)
    search_action = jnp.array([2], dtype=jnp.int32)

    played_action = _select_played_action(
        "search_action",
        jax.random.PRNGKey(0),
        action_weights,
        legal_action_mask,
        search_action,
    )

    assert jnp.array_equal(played_action, search_action)


def test_select_played_action_legalizes_search_action():
    action_weights = jnp.array([[0.0, 1.0, 0.0]])
    legal_action_mask = jnp.array([[False, True, False]])
    search_action = jnp.array([2], dtype=jnp.int32)

    played_action = _select_played_action(
        "search_action",
        jax.random.PRNGKey(0),
        action_weights,
        legal_action_mask,
        search_action,
    )

    assert jnp.array_equal(played_action, jnp.array([1], dtype=jnp.int32))


def test_legalize_played_action_handles_out_of_bounds_and_terminal_rows():
    legal_action_mask = jnp.array(
        [
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ]
    )
    action = jnp.array([-1, 9, 2], dtype=jnp.int32)

    played_action = _legalize_played_action(action, legal_action_mask)

    assert jnp.array_equal(played_action, jnp.array([1, 2, 0], dtype=jnp.int32))


def test_select_played_action_rejects_unknown_source():
    with pytest.raises(ValueError, match="selfplay_action_source"):
        _select_played_action(
            "unknown",
            jax.random.PRNGKey(0),
            jnp.array([[1.0]]),
            jnp.array([[True]]),
            jnp.array([0], dtype=jnp.int32),
        )
