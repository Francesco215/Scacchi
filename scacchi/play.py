import chex
from typing import Any, Callable, Literal, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Bool, Float, Int
from pgx.experimental import auto_reset

from .distributed import (
    DISABLED_BATCH_PARALLEL,
    BatchParallel,
    constrain_batch_axis,
    local_shard_map,
)
from .play_search import (
    PlayerOutput,
    PosteriorTargets,
    make_search_player,
)


class TrainingSamples(NamedTuple):
    obs: Float[Array, "batch time ..."]
    reward: Float[Array, "batch time"]
    terminated: Bool[Array, "batch time"]
    discount: Float[Array, "batch time"]
    posterior: PosteriorTargets
    played_action: Int[Array, "batch time"]
    legal_action_mask: Bool[Array, "batch time action"]
    psk_terminated: Bool[Array, "batch time"] | None = None
    # The position each row was searched from; kept only on request.
    state: Any | None = None


class EvalMetrics(NamedTuple):
    returns: Float[Array, "batch"]
    avg_return: Float[Array, ""]
    win_rate: Float[Array, ""]
    draw_rate: Float[Array, ""]
    lose_rate: Float[Array, ""]


Player = Callable[[Any, chex.PRNGKey], PlayerOutput]
PlayMode = Literal["training", "eval"]


def eval_metrics_from_returns(returns: Float[Array, "batch"]) -> EvalMetrics:
    return EvalMetrics(
        returns=returns,
        avg_return=returns.mean(),
        win_rate=(returns > 0).mean(),
        draw_rate=(returns == 0).mean(),
        lose_rate=(returns < 0).mean(),
    )


def play_eval(
    env: Any,
    player_1: Player,
    player_2: Player,
    rng_key: jax.Array,
    *,
    batch_size: int,
    player_1_id: int = 0,
    parallel: BatchParallel | None = None,
) -> EvalMetrics:
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    key, init_key = jax.random.split(rng_key)
    init_keys = parallel.split(init_key, batch_size)
    env_state = jax.vmap(env.init)(init_keys)
    env_state = constrain_batch_axis(env_state, parallel, batch_axis=0)
    returns = jnp.zeros_like(env_state.terminated, dtype=jnp.float32)

    def body_fn(val):
        key, env_state, returns = val
        key, player_1_key, player_2_key = jax.random.split(key, 3)
        player_1_output = player_1(env_state, player_1_key)
        player_2_output = player_2(env_state, player_2_key)
        action = jnp.where(env_state.current_player == player_1_id, player_1_output.action, player_2_output.action)
        env_state = jax.vmap(env.step)(env_state, action)
        reward = env_state.rewards[jnp.arange(batch_size), player_1_id]
        returns = returns + reward
        return key, env_state, returns

    _, _, returns = nnx.while_loop(
        lambda x: ~x[1].terminated.all(),
        body_fn,
        (key, env_state, returns),
    )
    return eval_metrics_from_returns(returns)


def _selfplay_step_frame(
    env: Any,
    player: Player,
    env_state: Any,
    key: jax.Array,
    *,
    batch_size: int,
    extra_states: Any | None = None,
    keep_state: bool = False,
) -> tuple[Any, tuple[TrainingSamples, PosteriorTargets | None]]:
    """One self-play move; ``extra_states`` are searched in the same call but not played."""
    player_key, reset_key = jax.random.split(key, 2)

    # ``pgx.experimental.auto_reset`` returns a freshly initialized position
    # after a game ends, but deliberately carries the previous transition's
    # terminal flags and rewards for one frame.  Those fields belong to the
    # transition we already recorded; passing them into search makes every raw
    # ``env.step`` in a recurrent/expand function a terminal no-op.  Clear the
    # transition marker before asking the player to analyse the fresh board.
    env_state = env_state.replace(
        terminated=jnp.zeros_like(env_state.terminated),
        truncated=jnp.zeros_like(env_state.truncated),
        rewards=jnp.zeros_like(env_state.rewards),
    )
    observation = env_state.observation
    legal_action_mask = env_state.legal_action_mask
    searched_state = env_state
    if extra_states is not None:
        searched_state = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b), axis=0), env_state, extra_states)
    player_output = player(searched_state, player_key)
    if player_output.posterior is None:
        raise ValueError("self-play player must return posterior targets")
    extra_posterior = None
    if extra_states is not None:
        extra_posterior = jax.tree_util.tree_map(lambda x: x[batch_size:], player_output.posterior)
        player_output = PlayerOutput(
            action=player_output.action[:batch_size],
            posterior=jax.tree_util.tree_map(lambda x: x[:batch_size], player_output.posterior),
        )

    searched_from = env_state
    actor = env_state.current_player
    reset_keys = jax.random.split(reset_key, batch_size)
    env_state = jax.vmap(auto_reset(env.step, env.init))(env_state, player_output.action, reset_keys)
    reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor]
    discount = -jnp.ones_like(reward)
    discount = jnp.where(env_state.terminated, 0.0, discount)
    is_psk = getattr(getattr(env_state, "_x", None), "is_psk", None)
    psk_terminated = (
        jnp.zeros_like(env_state.terminated)
        if is_psk is None
        else env_state.terminated & is_psk
    )
    frame = TrainingSamples(
        obs=observation,
        reward=reward,
        terminated=env_state.terminated,
        legal_action_mask=legal_action_mask,
        discount=discount,
        posterior=player_output.posterior,
        played_action=player_output.action,
        psk_terminated=psk_terminated,
        state=searched_from if keep_state else None,
    )
    return env_state, (frame, extra_posterior)


def play_training(
    env: Any,
    player: Player,
    rng_key: jax.Array,
    *,
    batch_size: int,
    max_num_steps: int,
    parallel: BatchParallel | None = None,
    extra_states: Any | None = None,
    keep_states: bool = False,
    initial_states: Any | None = None,
) -> TrainingSamples | tuple[TrainingSamples, PosteriorTargets]:
    """Self-play rollouts. With ``extra_states`` (leaves ``[step, lane, ...]``)
    every search call also searches that step's positions, and their targets
    are returned as a second value with the same leading axes."""
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    rollout_rng, init_key = jax.random.split(rng_key)
    init_keys = jax.random.split(init_key, batch_size)
    # Lanes start from the given positions; a finished game restarts from the opening.
    env_state = jax.vmap(env.init)(init_keys) if initial_states is None else initial_states
    env_state = constrain_batch_axis(env_state, parallel, batch_axis=0)

    def step_fn(env_state: Any, inputs) -> tuple[Any, tuple[TrainingSamples, PosteriorTargets | None]]:
        key, extra = inputs
        return _selfplay_step_frame(
            env,
            player,
            env_state,
            key,
            batch_size=batch_size,
            extra_states=extra,
            keep_state=keep_states,
        )

    step_keys = jax.random.split(rollout_rng, max_num_steps)
    env_state, (data, extra_posteriors) = jax.lax.scan(step_fn, env_state, (step_keys, extra_states))
    data = jax.tree_util.tree_map( #exchanges the batch axis with the step axis
        lambda leaf: leaf.swapaxes(0, 1) if isinstance(leaf, jax.Array) else leaf,
        data,
    )
    if extra_states is None:
        return data
    return data, extra_posteriors


def play(
    env: Any,
    player_1: Player,
    player_2: Player,
    rng_key: jax.Array,
    *,
    mode: PlayMode,
    batch_size: int,
    max_num_steps: int | None = None,
    player_1_id: int = 0,
    parallel: BatchParallel | None = None,
) -> TrainingSamples | EvalMetrics:
    if mode == "training":
        if player_1 is not player_2:
            raise ValueError("training play requires identical self-play players")
        if max_num_steps is None:
            raise ValueError("training play requires max_num_steps")
        return play_training(env, player_1, rng_key, batch_size=batch_size, max_num_steps=max_num_steps, parallel=parallel)
    if mode == "eval":
        return play_eval(env, player_1, player_2, rng_key, batch_size=batch_size, player_1_id=player_1_id, parallel=parallel)
    raise ValueError(f"unknown play mode: {mode!r}")


def make_selfplay(env, config, parallel: BatchParallel | None = None):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel

    def training_player(model: nnx.Module):
        return make_search_player(
            env,
            model,
            config.selfplay.search,
            config.selfplay.action_commitment,
            q_supervision_config=config.training.losses.q_supervision,
        )

    local_batch_size = parallel.local_batch_size(int(config.selfplay.batch_size))
    if local_batch_size == 0:
        raise ValueError(
            "selfplay.batch_size must be at least the batch-parallel device count."
        )

    # Search always receives a device-local batch.  This avoids collectives in
    # MCTX's dynamic gather/scatter loops, and keeps local and multi-device
    # execution on precisely the same JAX path.
    @local_shard_map(
        mesh=parallel.execution_mesh(),
        in_specs=(PartitionSpec(), PartitionSpec(parallel.axis_name, None)),
        out_specs=PartitionSpec(parallel.axis_name),
    )
    def local_selfplay(
        model: nnx.Module,
        device_rng_keys: jax.Array,
    ) -> TrainingSamples:
        rng_key = device_rng_keys[0]
        return play_training(
            env,
            training_player(model),
            rng_key,
            batch_size=local_batch_size,
            max_num_steps=int(config.selfplay.max_num_steps),
            parallel=DISABLED_BATCH_PARALLEL,
        )

    @nnx.jit
    def selfplay(model: nnx.Module, rng_key: jax.Array) -> TrainingSamples:
        device_rng_keys = jax.random.split(rng_key, parallel.device_count)
        return local_selfplay(model, device_rng_keys)

    return selfplay


def make_research_selfplay(env, config, games: int | None = None):
    """Single-device self-play that also searches given positions.

    ``selfplay(model, key, extra_states)`` returns the usual samples, with the
    position of every row, and the search targets of ``extra_states`` (None
    when there are no extra states). ``games`` overrides selfplay.batch_size.
    """
    games = int(config.selfplay.batch_size) if games is None else int(games)

    @nnx.jit
    def selfplay(model: nnx.Module, rng_key: jax.Array, extra_states: Any, initial_states: Any = None):
        player = make_search_player(
            env,
            model,
            config.selfplay.search,
            config.selfplay.action_commitment,
            q_supervision_config=config.training.losses.q_supervision,
        )
        result = play_training(
            env,
            player,
            rng_key,
            batch_size=games,
            max_num_steps=int(config.selfplay.max_num_steps),
            extra_states=extra_states,
            keep_states=True,
            initial_states=initial_states,
        )
        return (result, None) if extra_states is None else result

    return selfplay
