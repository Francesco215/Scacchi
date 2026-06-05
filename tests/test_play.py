from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from scacchi.dirichlet_q_search import posterior_sample_action
from scacchi.distributed import BatchParallel, assert_batch_axis_sharded
from scacchi.play_search import (
    _legalize_played_action,
    _run_scalar_gumbel_search,
    _select_played_action,
)
from scacchi.types import Config, EnvConfig, GumbelSearchConfig, SearchConfig


_COLLECTIVE_HLO_NAMES = (
    "all-gather",
    "all_gather",
    "all-reduce",
    "all_reduce",
    "all-to-all",
    "all_to_all",
    "collective-permute",
    "collective_permute",
    "reduce-scatter",
    "reduce_scatter",
)


class _ToySearchState(NamedTuple):
    legal_action_mask: jax.Array


def _toy_scalar_recurrent_fn(_, rng_key, action, embedding: _ToySearchState):
    del rng_key, action
    prior_logits = embedding.legal_action_mask.astype(jnp.float32) * 0.0
    value = jnp.sum(prior_logits, axis=-1)
    output = mctx.RecurrentFnOutput(
        reward=value,
        discount=value - 1.0,
        prior_logits=prior_logits,
        value=value,
    )
    return output, embedding


def test_scalar_gumbel_search_preserves_batch_sharding_without_collectives():
    device_count = jax.device_count()
    mesh = jax.make_mesh(
        (device_count,),
        ("batch",),
        axis_types=(AxisType.Auto,),
    )
    parallel = BatchParallel(enabled=True, mesh=mesh)
    batch_size = max(device_count * 2, 2)
    num_actions = 3
    matrix_sharding = NamedSharding(mesh, PartitionSpec("batch", None))
    vector_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    env_state = _ToySearchState(
        legal_action_mask=jax.device_put(
            jnp.ones((batch_size, num_actions), dtype=jnp.bool_),
            matrix_sharding,
        )
    )
    model_output = (
        jax.device_put(
            jnp.zeros((batch_size, num_actions), dtype=jnp.float32),
            matrix_sharding,
        ),
        jax.device_put(jnp.zeros((batch_size,), dtype=jnp.float32), vector_sharding),
    )
    config = Config(
        env=EnvConfig(id="toy", num_outcomes=2),
        search=SearchConfig(gumbel=GumbelSearchConfig(num_simulations=2)),
    )

    def run(env_state, model_output, rng_key):
        return _run_scalar_gumbel_search(
            env_state=env_state,
            model_output=model_output,
            recurrent_fn=_toy_scalar_recurrent_fn,
            rng_key=rng_key,
            config=config,
        )

    rng_key = jax.random.PRNGKey(0)
    lowered = jax.jit(run).lower(env_state, model_output, rng_key)
    hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text().lower()
    for collective in _COLLECTIVE_HLO_NAMES:
        assert collective not in hlo_text

    output = jax.jit(run)(env_state, model_output, rng_key)
    output = assert_batch_axis_sharded(output, parallel, batch_axis=0, label="scalar gumbel output")
    assert output.action_weights.shape == (batch_size, num_actions)
    assert output.played_action.shape == (batch_size,)


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
