"""The fused search step builds the same trees as the JAX search, bit for bit.

Reference: ``dirichlet_mctx.search.search`` with ``binned_search.action_selection``
and ``binned_search.update_posterior(kernel="cuda")`` (what
``search_kernel=cuda`` runs). Same positions, same keys, same network; every
tree array must match exactly.
"""

import functools

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi import binned_search
from scacchi.dirichlet_mctx import RootFnOutput
from scacchi.dirichlet_mctx.search import search as tree_search
from scacchi.dirichlet_q_search import make_dirichlet_expand_fn, terminal_outcome_from_reward
from scacchi.envs import make_env
from scacchi.network import AZBinnedNet
from scacchi.play_search import make_evaluator
from scacchi.types import BinConfig

FIELDS = [
    "parents", "children_index", "node_to_play", "node_categorical_outcome", "node_payload",
    "edge_categorical_outcome", "edge_payload", "node_value_priors", "node_value_alpha",
    "edge_alpha", "invalid_actions",
]


def _kernel_available():
    if jax.default_backend() != "gpu":
        return False
    try:
        from scacchi.native_binned._library import find_nvcc

        find_nvcc()
    except RuntimeError:
        return False
    return True


needs_kernel = pytest.mark.skipif(not _kernel_available(), reason="needs a CUDA GPU and nvcc")


def _model(env, seed):
    """A small binned network with random (not zero) output layers, so beliefs differ."""
    model = AZBinnedNet(env.num_actions, env.observation_shape, num_channels=16, num_blocks=1,
                        dtype=jnp.bfloat16, bins=BinConfig(count=9), rngs=nnx.Rngs(seed))
    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    leaves = [leaf + 0.5 * jax.random.normal(key, leaf.shape, leaf.dtype) for leaf, key in zip(leaves, keys)]
    nnx.update(model, jax.tree.unflatten(treedef, leaves))
    return model


def _positions(env, batch, plies, key):
    init_key, key = jax.random.split(key)
    state = jax.vmap(env.init)(jax.random.split(init_key, batch))

    def body(state, step_key):
        logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
        action = jax.random.categorical(step_key, logits).astype(jnp.int32)
        stepped = jax.vmap(env.step)(state, action)
        keep = state.terminated
        return jax.tree.map(lambda old, new: jnp.where(keep.reshape(keep.shape + (1,) * (new.ndim - 1)), old, new), state, stepped), None

    if plies:
        state, _ = jax.lax.scan(body, state, jax.random.split(key, plies))
    return state


def _searches(env, simulations, max_depth, cohorts):
    from scacchi.native_binned.search import fused_search

    def roots(model, state):
        evaluator = make_evaluator(model)
        prediction = evaluator(state.observation)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), state.current_player]
        root = RootFnOutput(
            prior_logits=prediction.logits, value=prediction.alpha_v, action_values=prediction.alpha_q, embedding=state,
            terminal_outcome=jnp.where(state.terminated, terminal_outcome_from_reward(reward, 2), jnp.int8(-1)),
            to_play=state.current_player,
        )
        return evaluator, prediction, root

    @nnx.jit
    def reference(model, state, key):
        evaluator, prediction, root = roots(model, state)
        expand = make_dirichlet_expand_fn(env, evaluator, num_outcomes=2)
        repair = functools.partial(binned_search.update_posterior, points=prediction.support, kernel="cuda")
        return tree_search(
            params=(), rng_key=key, root=root, recurrent_fn=expand, action_selection_fn=binned_search.action_selection,
            posterior_update=repair, num_simulations=simulations, max_depth=max_depth,
            invalid_actions=~state.legal_action_mask, num_outcomes=2,
        )

    @nnx.jit
    def fused(model, state, key):
        evaluator, prediction, root = roots(model, state)
        return fused_search(evaluator, state, root, key, points=prediction.support, num_simulations=simulations,
                            max_depth=max_depth, cohorts=cohorts)

    return reference, fused


@needs_kernel
@pytest.mark.parametrize(
    "size,simulations,max_depth,cohorts",
    # One cohort: with two, cuDNN may round the half batch differently (see search.py).
    [(3, 16, 16, 1), (5, 24, 24, 1), (5, 24, 3, 1), (7, 32, 32, 1), (11, 12, 12, 1)],
)
def test_fused_search_matches_jax_search(size, simulations, max_depth, cohorts):
    env = make_env("hex", size)
    model = _model(env, seed=size)
    reference, fused = _searches(env, simulations, max_depth, cohorts)
    batch = 64
    plies = [0, size * size // 3, (2 * size * size) // 3]
    for index, ply in enumerate(plies):
        states = _positions(env, batch, ply, jax.random.PRNGKey(100 * size + index))
        key = jax.random.PRNGKey(index)
        expected = reference(model, states, key)
        actual = fused(model, states, key)
        for name in FIELDS:
            x = np.asarray(getattr(expected, name))
            y = np.asarray(getattr(actual, name))
            if x.dtype.kind == "f":
                x, y = x.view(np.uint32), y.view(np.uint32)
            np.testing.assert_array_equal(y, x, err_msg=f"{name} differs (hex {size}, ply {ply})")


@needs_kernel
def test_fused_search_handles_finished_roots():
    """Terminal roots run no simulation and keep their root, as in the JAX search."""
    env = make_env("hex", 3)
    model = _model(env, seed=0)
    reference, fused = _searches(env, 8, 8, 1)
    states = _positions(env, 32, 9, jax.random.PRNGKey(5))  # most 3x3 games are over after 9 moves
    assert bool(jnp.any(states.terminated))
    expected = reference(model, states, jax.random.PRNGKey(1))
    actual = fused(model, states, jax.random.PRNGKey(1))
    for name in FIELDS:
        x = np.asarray(getattr(expected, name))
        y = np.asarray(getattr(actual, name))
        if x.dtype.kind == "f":
            x, y = x.view(np.uint32), y.view(np.uint32)
        np.testing.assert_array_equal(y, x, err_msg=name)


@needs_kernel
def test_cohorts_match_with_native_network():
    """With the CUDA network a position's result is batch-independent, so two cohorts give the same trees."""
    from scacchi.native_binned.network import make_native_evaluator
    from scacchi.native_binned.search import fused_search

    env = make_env("hex", 7)
    model = AZBinnedNet(env.num_actions, env.observation_shape, num_channels=128, num_blocks=6,
                        dtype=jnp.bfloat16, bins=BinConfig(count=9), rngs=nnx.Rngs(3))
    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.PRNGKey(3), len(leaves))
    nnx.update(model, jax.tree.unflatten(treedef, [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]))

    def run(cohorts):
        @nnx.jit
        def search(model, states, key):
            evaluator = make_native_evaluator(model)
            prediction = evaluator(states.observation)
            root = RootFnOutput(prior_logits=prediction.logits, value=prediction.alpha_v, action_values=prediction.alpha_q,
                                embedding=states, terminal_outcome=jnp.full(states.terminated.shape, -1, jnp.int8),
                                to_play=states.current_player)
            return fused_search(evaluator, states, root, key, points=prediction.support, num_simulations=16,
                                max_depth=16, cohorts=cohorts)
        return search

    states = _positions(env, 128, 12, jax.random.PRNGKey(9))
    a = run(1)(model, states, jax.random.PRNGKey(4))
    b = run(2)(model, states, jax.random.PRNGKey(4))
    for name in FIELDS:
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.dtype.kind == "f":
            x, y = x.view(np.uint32), y.view(np.uint32)
        np.testing.assert_array_equal(y, x, err_msg=name)
