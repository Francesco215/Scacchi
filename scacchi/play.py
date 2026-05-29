import weakref
from typing import Any, Callable, NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset

from .dirichlet_q_search import (
    NO_PARENT,
    NodeEmbedding,
    flip_outcome,
    outcome_mean,
    outcome_utility,
    terminal_outcome_from_reward,
)
from .dirichlet_tree.types import SearchDiagnostics, TreeTrainingData
from .network import policy_value_from_output
from .play_search import (
    _SearchStepOutput,
    _run_model_search,
    _run_posterior_tree_search_step,
    _select_played_action,
)
from .posterior_tree import is_posterior_tree_policy


BatchedEnvInit = Callable[[jax.Array], Any]
BatchedEnvStep = Callable[[Any, jax.Array, jax.Array], Any]

_CPU_ENV_INIT_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, BatchedEnvInit]] = {}
_CPU_ENV_STEP_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, BatchedEnvStep]] = {}


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    action_weights: chex.Array
    played_action: jax.Array
    legal_action_mask: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    discount: jax.Array
    tree_data: TreeTrainingData | None = None
    search_loss_mask: jax.Array | None = None
    search_diagnostics: SearchDiagnostics | None = None
    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None

    @property
    def q_evidence_mass(self) -> jax.Array:
        return self.q_loss_weight


_STACKED_FRAME_FIELD_NAMES = tuple(
    field
    for field in SelfplayOutput._fields
    if field not in ("tree_data", "search_diagnostics")
)


def _cpu_device() -> jax.Device:
    try:
        return jax.devices("cpu")[0]
    except RuntimeError as exc:
        raise RuntimeError(
            "posterior_tree selfplay requires the JAX CPU platform for PGX env "
            "initialization and stepping. Use JAX_PLATFORMS=cuda,cpu when running "
            "with a GPU."
        ) from exc


def _device_put_cpu(value: Any) -> Any:
    return jax.device_put(value, _cpu_device())


def _env_ref(env: Any) -> weakref.ReferenceType[Any] | None:
    try:
        return weakref.ref(env)
    except TypeError:
        return None


def _cached_cpu_env_init(env: Any) -> BatchedEnvInit:
    cache_key = id(env)
    cached = _CPU_ENV_INIT_CACHE.get(cache_key)
    if cached is not None:
        env_ref, init_fn = cached
        if env_ref is None or env_ref() is env:
            return init_fn
    init_fn = jax.jit(jax.vmap(env.init))
    _CPU_ENV_INIT_CACHE[cache_key] = (_env_ref(env), init_fn)
    return init_fn


def _cached_cpu_env_step(env: Any) -> BatchedEnvStep:
    cache_key = id(env)
    cached = _CPU_ENV_STEP_CACHE.get(cache_key)
    if cached is not None:
        env_ref, step_fn = cached
        if env_ref is None or env_ref() is env:
            return step_fn
    step_fn = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    _CPU_ENV_STEP_CACHE[cache_key] = (_env_ref(env), step_fn)
    return step_fn


def _cached_default_env_init(env: Any) -> BatchedEnvInit:
    return _cached_cpu_env_init(env)


def _cached_default_env_step(env: Any) -> BatchedEnvStep:
    return _cached_cpu_env_step(env)


def make_recurrent_fn(env, predict_fn):
    def recurrent_fn(
        _,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        env_state: pgx.State,
    ):
        del rng_key

        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        logits, value = policy_value_from_output(predict_fn(env_state.observation))
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            env_state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            env_state,
        )

    return recurrent_fn


def make_dirichlet_recurrent_fn(env, predict_fn, config):
    def recurrent_fn(
        _,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        embedding: NodeEmbedding,
    ):
        del rng_key

        current_player = embedding.state.current_player
        env_state = jax.vmap(env.step)(embedding.state, action)
        logits, alpha_v, alpha_q = predict_fn(env_state.observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            env_state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        nonterminal_outcome = outcome_mean(alpha_v)
        terminal_parent_outcome = terminal_outcome_from_reward(
            reward,
            alpha_v.shape[-1],
        )
        terminal_child_outcome = flip_outcome(terminal_parent_outcome)
        outcome_dist = jnp.where(
            env_state.terminated[..., None],
            terminal_child_outcome,
            nonterminal_outcome,
        )
        evidence_weight = jnp.where(
            env_state.terminated,
            jnp.asarray(config.kappa_terminal, dtype=outcome_dist.dtype),
            jnp.asarray(config.kappa_leaf, dtype=outcome_dist.dtype),
        )
        root_action = jnp.where(
            embedding.root_action == NO_PARENT,
            action,
            embedding.root_action,
        )
        depth_parity = 1 - embedding.depth_parity

        value = outcome_utility(outcome_dist)
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        next_embedding = NodeEmbedding(
            state=env_state,
            outcome_dist=outcome_dist,
            alpha_V_prior=alpha_v,
            evidence_weight=evidence_weight,
            root_action=root_action,
            depth_parity=depth_parity,
            alpha_Q_prior=alpha_q,
        )
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            next_embedding,
        )

    return recurrent_fn


def _selfplay_frame(
    *,
    observation: jax.Array,
    legal_action_mask: jax.Array,
    reward: jax.Array,
    terminated: jax.Array,
    discount: jax.Array,
    search_output: _SearchStepOutput,
) -> SelfplayOutput:
    return SelfplayOutput(
        obs=observation,
        reward=reward,
        terminated=terminated,
        legal_action_mask=legal_action_mask,
        discount=discount,
        **search_output._asdict(),
    )


def _stack_optional_tree(values: list[Any]) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *present)


def _stack_selfplay_frames(frames: list[SelfplayOutput]) -> SelfplayOutput:
    def stack_or_none(name: str) -> jax.Array | None:
        values = [getattr(frame, name) for frame in frames]
        if values[0] is None:
            return None
        return jnp.stack(values, axis=0)

    stacked = {name: stack_or_none(name) for name in _STACKED_FRAME_FIELD_NAMES}
    return SelfplayOutput(
        **stacked,
        tree_data=_stack_optional_tree([frame.tree_data for frame in frames]),
        search_diagnostics=_stack_optional_tree(
            [frame.search_diagnostics for frame in frames]
        ),
    )


def _select_posterior_tree_played_action(
    action_source: str,
    rng_key: jax.Array,
    action_weights: jax.Array,
    legal_action_mask: jax.Array,
    search_action: jax.Array,
) -> jax.Array:
    return _select_played_action(
        action_source,
        rng_key,
        action_weights,
        legal_action_mask,
        search_action,
    )


def make_posterior_tree_selfplay(env, config):
    @nnx.jit
    def evaluate_leaves(model: nnx.Module, obs: jax.Array):
        return model(obs, train=False)

    def selfplay(model: nnx.Module, rng_key: jax.Array) -> SelfplayOutput:
        def leaf_evaluator(obs: jax.Array):
            output = evaluate_leaves(model, obs)
            if len(output) != 3:
                raise ValueError(
                    "posterior-tree search requires a Dirichlet model "
                    "returning (logits, alpha_V, alpha_Q)."
                )
            return output

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = _device_put_cpu(
            jax.random.split(init_key, config.selfplay_batch_size)
        )
        env_init = _cached_cpu_env_init(env)
        env_step = _cached_cpu_env_step(env)
        env_state = env_init(init_keys)

        frames = []

        for _ in range(config.max_num_steps):
            rng_key, search_key, reset_key = jax.random.split(rng_key, 3)
            observation = env_state.observation
            legal_action_mask = env_state.legal_action_mask
            actor = env_state.current_player
            search_output = _run_posterior_tree_search_step(
                env=env,
                config=config,
                env_state=env_state,
                leaf_evaluator=leaf_evaluator,
                search_key=search_key,
                device_put_cpu=_device_put_cpu,
            )

            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            reset_keys = _device_put_cpu(reset_keys)
            env_state = env_step(env_state, search_output.played_action, reset_keys)
            reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor]
            discount = -jnp.ones((config.selfplay_batch_size,), dtype=reward.dtype)
            discount = jnp.where(env_state.terminated, 0.0, discount)

            frames.append(
                _selfplay_frame(
                    observation=observation,
                    legal_action_mask=legal_action_mask,
                    reward=reward,
                    terminated=env_state.terminated,
                    discount=discount,
                    search_output=search_output,
                )
            )

        return _stack_selfplay_frames(frames)

    return selfplay


def make_selfplay(env, config):
    if is_posterior_tree_policy(config.search_policy):
        return make_posterior_tree_selfplay(env, config)

    @nnx.jit
    def selfplay(model: nnx.Module, rng_key: jax.Array) -> SelfplayOutput:
        predict_fn = lambda obs: model(obs, train=False)
        recurrent_fn = make_recurrent_fn(env, predict_fn)
        dirichlet_recurrent_fn = make_dirichlet_recurrent_fn(env, predict_fn, config)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step_fn(
            env_state: pgx.State,
            key: jax.Array,
        ) -> tuple[pgx.State, SelfplayOutput]:
            search_key, posterior_key, action_key, reset_key = jax.random.split(key, 4)
            observation = env_state.observation
            legal_action_mask = env_state.legal_action_mask
            model_output = predict_fn(observation)
            search_output = _run_model_search(
                env_state=env_state,
                model_output=model_output,
                scalar_recurrent_fn=recurrent_fn,
                dirichlet_recurrent_fn=dirichlet_recurrent_fn,
                search_key=search_key,
                posterior_key=posterior_key,
                action_key=action_key,
                config=config,
            )

            actor = env_state.current_player
            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            env_state = jax.vmap(auto_reset(env.step, env.init))(
                env_state,
                search_output.played_action,
                reset_keys,
            )
            reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor]
            discount = -jnp.ones_like(reward)
            discount = jnp.where(env_state.terminated, 0.0, discount)
            return env_state, _selfplay_frame(
                observation=observation,
                legal_action_mask=legal_action_mask,
                reward=reward,
                terminated=env_state.terminated,
                discount=discount,
                search_output=search_output,
            )

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        step_keys = jax.random.split(rng_key, config.max_num_steps)
        _, data = step_fn(env_state, step_keys)
        return data

    return selfplay
