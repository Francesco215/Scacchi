import weakref
from typing import Any, Callable, NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset

from .dirichlet_tree.types import SearchDiagnostics, TreeTrainingData
from .dirichlet_tree.native import native_fields_from_beta
from .network import policy_value_from_output
from .posterior_tree import run_posterior_tree_search_state_batch


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


def make_posterior_tree_selfplay(env, selfplay_config, search_config):
    @nnx.jit
    def evaluate_leaves(model: Any, obs: jax.Array):
        return model(obs, train=False)

    def selfplay(model: Any, rng_key: jax.Array) -> SelfplayOutput:
        def leaf_evaluator(obs: jax.Array):
            output = evaluate_leaves(model, obs)
            if len(output) != 3:
                raise ValueError(
                    "posterior-tree search requires a Dirichlet model "
                    "returning (logits, alpha_V, alpha_Q)."
                )
            return output

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, selfplay_config.batch_size)
        env_init = _cached_default_env_init(env)
        env_step = _cached_default_env_step(env)
        env_state = env_init(init_keys)

        obs_seq = []
        reward_seq = []
        terminated_seq = []
        action_weights_seq = []
        played_action_seq = []
        legal_action_mask_seq = []
        beta_q_seq = []
        beta_v_seq = []
        q_loss_weight_seq = []
        q_target_kind_seq = []
        q_target_weight_seq = []
        q_target_outcome_seq = []
        q_target_distance_seq = []
        v_target_kind_seq = []
        v_target_weight_seq = []
        v_target_outcome_seq = []
        v_target_distance_seq = []
        search_loss_mask_seq = []
        discount_seq = []
        tree_data_seq = []
        search_diagnostics_seq = []

        for _ in range(selfplay_config.max_num_steps):
            rng_key, search_key, reset_key = jax.random.split(rng_key, 3)
            observation = env_state.observation
            legal_action_mask = env_state.legal_action_mask
            actor = env_state.current_player
            search_output = run_posterior_tree_search_state_batch(
                env=env,
                root_state_batch=env_state,
                leaf_evaluator=leaf_evaluator,
                rng_key=search_key,
                config=search_config,
            )
            played_action = search_output.action

            reset_keys = jax.random.split(reset_key, selfplay_config.batch_size)
            env_state = env_step(env_state, played_action, reset_keys)
            reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor]
            discount = -jnp.ones((selfplay_config.batch_size,), dtype=reward.dtype)
            discount = jnp.where(env_state.terminated, 0.0, discount)

            obs_seq.append(observation)
            action_weights_seq.append(search_output.action_weights)
            played_action_seq.append(played_action)
            legal_action_mask_seq.append(legal_action_mask)
            beta_q_seq.append(search_output.beta_Q_target)
            beta_v_seq.append(search_output.beta_V_target)
            q_loss_weight_seq.append(search_output.q_loss_weight)
            native_defaults = native_fields_from_beta(
                search_output.beta_Q_target,
                search_output.beta_V_target,
            )
            q_target_kind_seq.append(
                search_output.q_target_kind
                if search_output.q_target_kind is not None
                else native_defaults["q_target_kind"]
            )
            q_target_weight_seq.append(
                search_output.q_target_weight
                if search_output.q_target_weight is not None
                else native_defaults["q_target_weight"]
            )
            q_target_outcome_seq.append(
                search_output.q_target_outcome
                if search_output.q_target_outcome is not None
                else native_defaults["q_target_outcome"]
            )
            q_target_distance_seq.append(
                search_output.q_target_distance
                if search_output.q_target_distance is not None
                else native_defaults["q_target_distance"]
            )
            v_target_kind_seq.append(
                search_output.v_target_kind
                if search_output.v_target_kind is not None
                else native_defaults["v_target_kind"]
            )
            v_target_weight_seq.append(
                search_output.v_target_weight
                if search_output.v_target_weight is not None
                else native_defaults["v_target_weight"]
            )
            v_target_outcome_seq.append(
                search_output.v_target_outcome
                if search_output.v_target_outcome is not None
                else native_defaults["v_target_outcome"]
            )
            v_target_distance_seq.append(
                search_output.v_target_distance
                if search_output.v_target_distance is not None
                else native_defaults["v_target_distance"]
            )
            root_search_mask = search_output.search_loss_mask
            if root_search_mask is None:
                root_search_mask = jnp.sum(search_output.action_weights, axis=-1) > 0
            search_loss_mask_seq.append(root_search_mask)
            if search_output.tree_data is not None:
                tree_data_seq.append(search_output.tree_data)
            diagnostics = getattr(search_output, "diagnostics", None)
            if diagnostics is not None:
                search_diagnostics_seq.append(diagnostics)
            reward_seq.append(reward)
            terminated_seq.append(env_state.terminated)
            discount_seq.append(discount)

        tree_data = None
        if tree_data_seq:
            tree_data = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0),
                *tree_data_seq,
            )
        search_diagnostics = None
        if search_diagnostics_seq:
            search_diagnostics = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0),
                *search_diagnostics_seq,
            )

        return SelfplayOutput(
            obs=jnp.stack(obs_seq, axis=0),
            reward=jnp.stack(reward_seq, axis=0),
            terminated=jnp.stack(terminated_seq, axis=0),
            action_weights=jnp.stack(action_weights_seq, axis=0),
            played_action=jnp.stack(played_action_seq, axis=0),
            legal_action_mask=jnp.stack(legal_action_mask_seq, axis=0),
            beta_Q_target=jnp.stack(beta_q_seq, axis=0),
            beta_V_target=jnp.stack(beta_v_seq, axis=0),
            q_loss_weight=jnp.stack(q_loss_weight_seq, axis=0),
            discount=jnp.stack(discount_seq, axis=0),
            tree_data=tree_data,
            search_loss_mask=jnp.stack(search_loss_mask_seq, axis=0),
            search_diagnostics=search_diagnostics,
            q_target_kind=jnp.stack(q_target_kind_seq, axis=0),
            q_target_weight=jnp.stack(q_target_weight_seq, axis=0),
            q_target_outcome=jnp.stack(q_target_outcome_seq, axis=0),
            q_target_distance=jnp.stack(q_target_distance_seq, axis=0),
            v_target_kind=jnp.stack(v_target_kind_seq, axis=0),
            v_target_weight=jnp.stack(v_target_weight_seq, axis=0),
            v_target_outcome=jnp.stack(v_target_outcome_seq, axis=0),
            v_target_distance=jnp.stack(v_target_distance_seq, axis=0),
        )

    return selfplay


def make_selfplay(env, selfplay_config, search_config):
    return make_posterior_tree_selfplay(env, selfplay_config, search_config)
