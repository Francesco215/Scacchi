import inspect
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


class EvalMetrics(NamedTuple):
    returns: Float[Array, "batch"]
    avg_return: Float[Array, ""]
    win_rate: Float[Array, ""]
    draw_rate: Float[Array, ""]
    lose_rate: Float[Array, ""]


Player = Callable[[Any, jax.Array], PlayerOutput]
PlayMode = Literal["training", "eval"]


def _local_search_shard_map(**kwargs):
    """Build an nnx.shard_map for local per-device search.

    MCTX's local search body has varying scalar carry values inside while loops.
    The shard-map checker is useful for proving manual axis types, but here it
    rejects a valid local-only search before lowering. Newer JAX names the
    checker toggle `check_vma`; older versions used `check_rep`.
    """

    parameters = inspect.signature(nnx.shard_map).parameters
    if "check_vma" in parameters:
        return nnx.shard_map(**kwargs, check_vma=False)
    if "check_rep" in parameters:
        return nnx.shard_map(**kwargs, check_rep=False)
    return nnx.shard_map(**kwargs)


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
        action = jnp.where(
            env_state.current_player == player_1_id,
            player_1_output.action,
            player_2_output.action,
        )
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
) -> tuple[Any, TrainingSamples]:
    player_key, reset_key = jax.random.split(key, 2)
    observation = env_state.observation
    legal_action_mask = env_state.legal_action_mask
    player_output = player(env_state, player_key)
    if player_output.posterior is None:
        raise ValueError("self-play player must return posterior targets")

    actor = env_state.current_player
    reset_keys = jax.random.split(reset_key, batch_size)
    env_state = jax.vmap(auto_reset(env.step, env.init))(
        env_state,
        player_output.action,
        reset_keys,
    )
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
    )
    return env_state, frame


def play_training(
    env: Any,
    player: Player,
    rng_key: jax.Array,
    *,
    batch_size: int,
    max_num_steps: int,
    parallel: BatchParallel | None = None,
) -> TrainingSamples:
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    rollout_rng, init_key = jax.random.split(rng_key)
    init_keys = jax.random.split(init_key, batch_size)
    env_state = jax.vmap(env.init)(init_keys)
    env_state = constrain_batch_axis(env_state, parallel, batch_axis=0)

    def step_fn(env_state: Any, key: jax.Array) -> tuple[Any, TrainingSamples]:
        return _selfplay_step_frame(
            env,
            player,
            env_state,
            key,
            batch_size=batch_size,
        )

    step_keys = jax.random.split(rollout_rng, max_num_steps)
    env_state, data = jax.lax.scan(step_fn, env_state, step_keys)
    data = jax.tree_util.tree_map( #exchanges the batch axis with the step axis
        lambda leaf: leaf.swapaxes(0, 1) if isinstance(leaf, jax.Array) else leaf,
        data,
    )
    return data


def make_single_device_selfplay(
    env,
    config,
    parallel: BatchParallel | None = None,
):
    """Return the single-device PGX-style self-play function.

    This maps to the body of pgx/examples/alphazero/train.py::selfplay after
    removing pmap's leading device axis.  The returned function still emits
    Scacchi's richer TrainingSamples container with batch/time axes.
    """

    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel

    def training_player(model: nnx.Module):
        return make_search_player(
            env,
            model,
            config.selfplay.search,
            config.selfplay.action_commitment_type,
            q_loss_weight_mode=str(config.training.losses.q_loss_weight_mode),
        )

    @nnx.jit
    def selfplay(model: nnx.Module, rng_key: jax.Array) -> TrainingSamples:
        return play_training(
            env,
            training_player(model),
            rng_key,
            batch_size=int(config.selfplay.batch_size),
            max_num_steps=int(config.selfplay.max_num_steps),
            parallel=parallel,
        )

    return selfplay

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
        return play_training(
            env,
            player_1,
            rng_key,
            batch_size=batch_size,
            max_num_steps=max_num_steps,
            parallel=parallel,
        )
    if mode == "eval":
        return play_eval(env, player_1, player_2, rng_key, batch_size=batch_size, player_1_id=player_1_id, parallel=parallel)
    raise ValueError(f"unknown play mode: {mode!r}")


def make_selfplay(env, config, parallel: BatchParallel | None = None):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel

    if not parallel.enabled or parallel.device_count == 1:
        return make_single_device_selfplay(env, config, parallel)

    def training_player(model: nnx.Module):
        return make_search_player(
            env,
            model,
            config.selfplay.search,
            config.selfplay.action_commitment_type,
            q_loss_weight_mode=str(config.training.losses.q_loss_weight_mode),
        )

    # TODO: this part is super ugly but it works. Let's think on how to remove it safely
    if parallel.mesh is None:
        raise RuntimeError("batch-parallel self-play requires a mesh.")
    if int(config.selfplay.batch_size) % parallel.device_count != 0:
        raise ValueError(
            "selfplay.batch_size must be divisible by the batch-parallel "
            f"device count ({parallel.device_count})."
        )
    local_batch_size = int(config.selfplay.batch_size) // parallel.device_count
    if local_batch_size < 1:
        raise ValueError(
            "selfplay.batch_size must be at least the batch-parallel device count."
        )

    # MCTX uses dynamic gathers/scatters over its leading batch dimension.
    # If it sees a globally sharded [batch, ...] array, XLA inserts
    # cross-device collectives inside the search loop. shard_map gives each
    # device a local [local_batch, ...] search problem and stitches the
    # resulting samples back into a global batch-sharded pytree.
    @_local_search_shard_map(
        mesh=parallel.mesh,
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
