from typing import Any
from dataclasses import replace

from flax import nnx
import jax
import jax.numpy as jnp

from .distributed import (
    DISABLED_BATCH_PARALLEL,
    BatchParallel,
    assert_batch_axis_sharded,
    constrain_batch_axis,
)
from .play import make_dirichlet_recurrent_fn, make_recurrent_fn
from .play_search import _run_model_search
from .types import EvalBaseline, SearchKind


def _predict(model: Any, obs: jax.Array):
    if isinstance(model, nnx.Module):
        return model(obs, train=False)
    return model(obs)


def _with_eval_num_simulations(config, num_simulations: int | None):
    if num_simulations is None or num_simulations == int(
        config.search.active().num_simulations
    ):
        return config
    search_kind = str(config.search.kind)
    active_search = getattr(config.search, search_kind)
    return replace(
        config,
        search=replace(
            config.search,
            **{search_kind: replace(active_search, num_simulations=num_simulations)},
        ),
    )


def _with_eval_search_kind(config, search_kind: SearchKind):
    if config.search.kind == search_kind:
        return config
    return replace(config, search=replace(config.search, kind=search_kind))


def _baseline_eval_search_config(config):
    if config.eval.baseline != EvalBaseline.pgx:
        return config
    if config.search.kind == SearchKind.gumbel:
        return config

    # PGX baselines expose scalar policy/value heads, not Dirichlet heads.
    num_simulations = max(1, int(config.search.active().num_simulations))
    return _with_eval_num_simulations(
        _with_eval_search_kind(config, SearchKind.gumbel),
        num_simulations,
    )


def _replace_legal_action_mask(env_state, legal_action_mask: jax.Array):
    if hasattr(env_state, "replace"):
        return env_state.replace(legal_action_mask=legal_action_mask)
    if hasattr(env_state, "_replace"):
        return env_state._replace(legal_action_mask=legal_action_mask)
    raise TypeError("env_state must support replacing legal_action_mask")


def _searchable_eval_state(env_state):
    """Give completed eval rows a dummy legal action; their moves are discarded."""

    dummy_legal_action_mask = (
        jnp.zeros_like(env_state.legal_action_mask)
        .at[..., 0]
        .set(True)
    )
    legal_action_mask = jnp.where(
        env_state.terminated[..., None],
        dummy_legal_action_mask,
        env_state.legal_action_mask,
    )
    return _replace_legal_action_mask(env_state, legal_action_mask)


def _step_active_eval_rows(env, env_state, action: jax.Array):
    active = ~env_state.terminated
    action = jnp.asarray(action, dtype=jnp.int32)
    num_actions = env_state.legal_action_mask.shape[-1]
    in_bounds = (0 <= action) & (action < num_actions)
    safe_action = jnp.clip(action, 0, num_actions - 1)
    selected_is_legal = jnp.take_along_axis(
        env_state.legal_action_mask,
        safe_action[..., None],
        axis=-1,
    )[..., 0]
    invalid_action = jnp.any(active & ~(in_bounds & selected_is_legal))

    def step_one(state, row_action, row_active):
        def step_state(state):
            return env.step(state, row_action)

        should_step = row_active & ~invalid_action
        return jax.lax.cond(should_step, step_state, lambda state: state, state)

    return jax.vmap(step_one)(env_state, action, active), active, invalid_action


def _poison_eval_returns(returns: jax.Array, invalid_action: jax.Array) -> jax.Array:
    return jnp.where(invalid_action, jnp.full_like(returns, jnp.nan), returns)


def _make_model_mcts_policy(
    env,
    config,
    model,
    rng_key,
    env_state,
    parallel,
    num_simulations=None,
):
    search_config = _with_eval_num_simulations(config, num_simulations)
    predict = lambda obs: _predict(model, obs)
    search_state = _searchable_eval_state(env_state)
    search_state = assert_batch_axis_sharded(search_state, parallel, batch_axis=0, label="eval search_state")
    model_output = predict(search_state.observation)
    model_output = assert_batch_axis_sharded(model_output, parallel, batch_axis=0, label="eval model_output")
    search_output = _run_model_search(
        env_state=search_state,
        model_output=model_output,
        scalar_recurrent_fn=make_recurrent_fn(env, predict),
        dirichlet_recurrent_fn=make_dirichlet_recurrent_fn(env, predict, search_config),
        rng_key=rng_key,
        config=search_config,
    )
    return assert_batch_axis_sharded(search_output, parallel, batch_axis=0, label="eval search_output")


def _model_eval_action(env, config, model, rng_key, env_state, parallel):
    action = _make_model_mcts_policy(
        env,
        config,
        model,
        rng_key,
        env_state,
        parallel,
    ).played_action
    return action


def make_mcts_evaluate(
    env,
    config,
    baseline_model,
    parallel: BatchParallel | None = None,
):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    eval_batch_size = int(config.eval.batch_size)
    model_eval_config = config
    baseline_eval_config = _baseline_eval_search_config(config)

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: nnx.Module):
        """MCTS evaluation: model search vs pretrained opponent."""
        my_player = 0

        key, init_key = jax.random.split(rng_key)
        init_keys = parallel.split(init_key, eval_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        env_state = constrain_batch_axis(env_state, parallel, batch_axis=0)
        env_state = assert_batch_axis_sharded(env_state, parallel, batch_axis=0, label="eval env_state")
        returns = jnp.zeros_like(env_state.terminated, dtype=jnp.float32)

        def body_fn(val):
            key, env_state, returns = val
            key, my_key, opp_key = jax.random.split(key, 3)

            my_action = _model_eval_action(
                env,
                model_eval_config,
                model,
                my_key,
                env_state,
                parallel,
            )
            opp_action = _model_eval_action(
                env,
                baseline_eval_config,
                baseline_model,
                opp_key,
                env_state,
                parallel,
            )

            is_my_turn = env_state.current_player == my_player
            action = jnp.where(is_my_turn, my_action, opp_action)

            env_state, active, invalid_action = _step_active_eval_rows(
                env,
                env_state,
                action,
            )
            env_state = assert_batch_axis_sharded(env_state, parallel, batch_axis=0, label="eval stepped_env_state")
            reward = env_state.rewards[
                jnp.arange(eval_batch_size),
                my_player,
            ]
            returns = returns + jnp.where(active, reward, 0.0)
            returns = _poison_eval_returns(returns, invalid_action)
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()) & ~(jnp.isnan(x[2]).any()),
            body_fn,
            (key, env_state, returns),
        )
        return assert_batch_axis_sharded(returns, parallel, batch_axis=0, label="eval returns")

    return evaluate
