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


def _only_active_player(player: Player, width: int) -> Player:
    """Skip terminated games via prefix compaction when <= 1/4 of games remain active."""
    if width < 64 or width % 4 != 0:
        return player
    quarter = width // 4

    def wrapped(state: Any, key: jax.Array) -> PlayerOutput:
        active = ~state.terminated
        count = jnp.sum(active)

        def none_fn(_):
            return PlayerOutput(action=jnp.zeros((width,), dtype=jnp.int32), posterior=None)

        def quarter_fn(_):
            order = jnp.argsort(~active, stable=True)
            rows = order[:quarter]
            sub = jax.tree.map(lambda x: x[rows], state)
            out = player(sub, key)
            act = jnp.zeros((width,), dtype=out.action.dtype).at[rows].set(out.action)
            return PlayerOutput(action=act, posterior=None)

        def full_fn(_):
            out = player(state, key)
            return PlayerOutput(action=out.action, posterior=None)

        idx = jnp.where(count == 0, 0, jnp.where(count <= quarter, 1, 2))
        return jax.lax.switch(idx, (none_fn, quarter_fn, full_fn), None)

    return wrapped


def play_eval(
    env: Any,
    player_1: Player,
    player_2: Player,
    rng_key: jax.Array,
    *,
    batch_size: int,
    player_1_id: int = 0,
    parallel: BatchParallel | None = None,
    stratified_hex: bool = False,
) -> EvalMetrics:
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    key, init_key = jax.random.split(rng_key)

    if (
        stratified_hex
        and getattr(env, "id", None) == "hex"
        and batch_size >= 2
        and batch_size % 2 == 0
        and parallel.device_count == 1
    ):
        half = batch_size // 2
        init_keys = jax.random.split(init_key, batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        # Stratify colors deterministically: Cohort A [0:half] has player 0 move first (Black),
        # Cohort B [half:B] has player 1 move first (Black). Eliminates 100% of color coin-flip
        # variance and ensures each player searches only half the batch per ply!
        order_a = jnp.broadcast_to(jnp.array([0, 1], dtype=jnp.int32), (half, 2))
        order_b = jnp.broadcast_to(jnp.array([1, 0], dtype=jnp.int32), (half, 2))
        player_order = jnp.concatenate([order_a, order_b], axis=0)
        first_player = jnp.concatenate(
            [jnp.zeros((half,), dtype=jnp.int32), jnp.ones((half,), dtype=jnp.int32)],
            axis=0,
        )
        env_state = env_state.replace(_player_order=player_order, current_player=first_player)
        returns = jnp.zeros_like(env_state.terminated, dtype=jnp.float32)
        p1_fast = _only_active_player(player_1, half)
        p2_fast = _only_active_player(player_2, half)

        def hex_body_fn(val):
            key, env_state, returns, ply = val
            key, p1_key, p2_key = jax.random.split(key, 3)
            state_a = jax.tree.map(lambda x: x[:half], env_state)
            state_b = jax.tree.map(lambda x: x[half:], env_state)

            def even_ply(_):
                # x.color == 0: Cohort A -> player 0, Cohort B -> player 1
                a_out = p1_fast(state_a, p1_key) if player_1_id == 0 else p2_fast(state_a, p2_key)
                b_out = p2_fast(state_b, p2_key) if player_1_id == 0 else p1_fast(state_b, p1_key)
                return jnp.concatenate([a_out.action, b_out.action], axis=0)

            def odd_ply(_):
                # x.color == 1: Cohort A -> player 1, Cohort B -> player 0
                a_out = p2_fast(state_a, p2_key) if player_1_id == 0 else p1_fast(state_a, p1_key)
                b_out = p1_fast(state_b, p1_key) if player_1_id == 0 else p2_fast(state_b, p2_key)
                return jnp.concatenate([a_out.action, b_out.action], axis=0)

            action = jax.lax.cond((ply & 1) == 0, even_ply, odd_ply, None)
            env_state = jax.vmap(env.step)(env_state, action)
            reward = env_state.rewards[jnp.arange(batch_size), player_1_id]
            returns = returns + reward
            return key, env_state, returns, ply + 1

        _, _, returns, _ = nnx.while_loop(
            lambda x: ~x[1].terminated.all(),
            hex_body_fn,
            (key, env_state, returns, jnp.int32(0)),
        )
        return eval_metrics_from_returns(returns)

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
) -> tuple[Any, TrainingSamples]:
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
    player_output = player(env_state, player_key)
    if player_output.posterior is None:
        raise ValueError("self-play player must return posterior targets")

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


def _init_dephased_env_state(env: Any, key: jax.Array, batch_size: int) -> Any:
    """Initialize batch_size games and step each lane a uniform random number of plies."""
    init_key, stop_key, loop_key = jax.random.split(key, 3)
    state = jax.vmap(env.init)(jax.random.split(init_key, batch_size))
    max_plies = max(1, (int(env.num_actions) - 1) // 2)
    stop_ply = jax.random.randint(stop_key, (batch_size,), 0, max_plies + 1)
    step_fn = jax.vmap(auto_reset(env.step, env.init))

    def body(ply, carry):
        st, k = carry
        k, act_k, rst_k = jax.random.split(k, 3)
        st_clean = st.replace(
            terminated=jnp.zeros_like(st.terminated),
            truncated=jnp.zeros_like(st.truncated),
            rewards=jnp.zeros_like(st.rewards),
        )
        logits = jnp.where(st_clean.legal_action_mask, 0.0, -1e9)
        act = jax.random.categorical(act_k, logits).astype(jnp.int32)
        stepped = step_fn(st_clean, act, jax.random.split(rst_k, batch_size))
        active = ply < stop_ply
        nxt = jax.tree.map(
            lambda new, old: jnp.where(active.reshape((batch_size,) + (1,) * (new.ndim - 1)), new, old),
            stepped,
            st,
        )
        return nxt, k

    state, _ = jax.lax.fori_loop(0, max_plies, body, (state, loop_key))
    return state


def _play_training_fast_hex(
    env: Any,
    model: nnx.Module,
    env_state: Any,
    rng_key: jax.Array,
    *,
    batch_size: int,
    max_num_steps: int,
    config: Any,
) -> tuple[Any, TrainingSamples]:
    """Stateful Hex self-play rollout with subtree reuse, GPU position cache, and compact INT8 net."""
    from . import binned
    from .dirichlet_mctx.native_targets import TARGET_BINNED
    from .native_binned.network import empty_position_cache, pack, supports
    from .native_binned.search import empty_fast_tree_carry, fused_fast_search
    from .play_search import make_action_committer, targets_from_summary
    from .types import RootPolicySupport

    if not supports(model):
        raise ValueError("fast_search requires AZBinnedNet with 128 channels and 6 ResNet-v2 blocks")

    bt_cfg = config.selfplay.search.binned_thompson
    q_sup_cfg = config.training.losses.q_supervision
    action_committer = make_action_committer(config.selfplay.action_commitment, None)

    packed = pack(model, "int8")
    points = jnp.asarray(binned.support(model.bins), dtype=jnp.float32)
    atoms = int(model.bins.count)
    num_actions = int(env.num_actions)
    cells = num_actions - 1
    num_sims = int(bt_cfg.num_simulations)
    max_depth = int(bt_cfg.max_depth if bt_cfg.max_depth is not None else num_sims)

    tree_carry0 = empty_fast_tree_carry(batch_size, num_sims, num_actions, atoms, cells)
    cache0 = empty_position_cache(num_actions, atoms, enabled=bt_cfg.position_cache)
    step_env = jax.vmap(auto_reset(env.step, env.init))

    def step_fn(carry, key):
        st, tree_carry, cache = carry
        search_key, cap_key, action_key, reset_key = jax.random.split(key, 4)
        st = st.replace(
            terminated=jnp.zeros_like(st.terminated),
            truncated=jnp.zeros_like(st.truncated),
            rewards=jnp.zeros_like(st.rewards),
        )
        observation = st.observation
        legal_action_mask = st.legal_action_mask

        if bt_cfg.playout_cap_prob < 1.0:
            full_mask = jax.random.bernoulli(cap_key, bt_cfg.playout_cap_prob, (batch_size,))
            lane_caps = jnp.where(full_mask, num_sims + 1, int(bt_cfg.fast_simulations) + 1)
        else:
            full_mask = None
            lane_caps = None

        summary, policy, search_action, tree_carry, cache = fused_fast_search(
            packed,
            st,
            search_key,
            points=points,
            num_simulations=num_sims,
            max_depth=max_depth,
            tree_carry=tree_carry,
            cache=cache,
            lane_caps=lane_caps,
            subtree_reuse=bt_cfg.subtree_reuse,
            use_cache=bt_cfg.position_cache,
            early_stop_prob=bt_cfg.early_stop_prob,
            min_stop_nodes=bt_cfg.min_stop_nodes,
            support_search_evidence=(bt_cfg.root_policy_support == RootPolicySupport.search_evidence),
            policy_target_temperature=bt_cfg.policy_target_temperature,
        )
        posterior = targets_from_summary(
            summary, policy, search_action, legal_action_mask, q_sup_cfg, unresolved_kind=TARGET_BINNED
        )
        if full_mask is not None and posterior.metadata is not None:
            base_mask = posterior.metadata.mask
            new_mask = full_mask if base_mask is None else (base_mask & full_mask)
            posterior = posterior._replace(metadata=posterior.metadata._replace(mask=new_mask))

        played_action = action_committer(posterior, legal_action_mask, action_key)
        actor = st.current_player
        reset_keys = jax.random.split(reset_key, batch_size)
        st_next = step_env(st, played_action, reset_keys)
        reward = st_next.rewards[jnp.arange(batch_size), actor]
        discount = jnp.where(st_next.terminated, 0.0, -jnp.ones_like(reward))
        is_psk = getattr(getattr(st_next, "_x", None), "is_psk", None)
        psk_terminated = jnp.zeros_like(st_next.terminated) if is_psk is None else (st_next.terminated & is_psk)

        tree_carry_next = tree_carry._replace(
            prev_actions=played_action,
            was_reset=st_next.terminated,
        )
        frame = TrainingSamples(
            obs=observation,
            reward=reward,
            terminated=st_next.terminated,
            legal_action_mask=legal_action_mask,
            discount=discount,
            posterior=posterior,
            played_action=played_action,
            psk_terminated=psk_terminated,
        )
        return (st_next, tree_carry_next, cache), frame

    step_keys = jax.random.split(rng_key, max_num_steps)
    (final_state, _, _), data = jax.lax.scan(step_fn, (env_state, tree_carry0, cache0), step_keys)
    data = jax.tree_util.tree_map(
        lambda leaf: leaf.swapaxes(0, 1) if isinstance(leaf, jax.Array) else leaf,
        data,
    )
    return final_state, data


def make_selfplay(env, config, parallel: BatchParallel | None = None):
    from .binned_search import _kernels_available
    from .types import SearchKind

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

    max_num_steps = int(config.selfplay.max_num_steps)
    persistent = bool(getattr(config.selfplay, "persistent_envs", False))
    use_fast_hex = (
        config.selfplay.search.kind == SearchKind.binned_thompson
        and bool(getattr(config.selfplay.search.binned_thompson, "fast_search", False))
        and config.selfplay.search.binned_thompson.network_kernel == "int8"
        and getattr(env, "id", None) == "hex"
        and int(env.num_actions) <= 128
        and _kernels_available()
    )

    if not persistent and not use_fast_hex:
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
                max_num_steps=max_num_steps,
                parallel=DISABLED_BATCH_PARALLEL,
            )

        @nnx.jit
        def selfplay(model: nnx.Module, rng_key: jax.Array) -> TrainingSamples:
            device_rng_keys = jax.random.split(rng_key, parallel.device_count)
            return local_selfplay(model, device_rng_keys)

        return selfplay

    @local_shard_map(
        mesh=parallel.execution_mesh(),
        in_specs=(PartitionSpec(parallel.axis_name, None),),
        out_specs=PartitionSpec(parallel.axis_name),
    )
    def init_persistent_states(device_rng_keys: jax.Array):
        rng_key = device_rng_keys[0]
        if persistent:
            return _init_dephased_env_state(env, rng_key, local_batch_size)
        return jax.vmap(env.init)(jax.random.split(rng_key, local_batch_size))

    @jax.jit
    def jit_init_states(rng_key: jax.Array):
        return init_persistent_states(jax.random.split(rng_key, parallel.device_count))

    @local_shard_map(
        mesh=parallel.execution_mesh(),
        in_specs=(PartitionSpec(), PartitionSpec(parallel.axis_name), PartitionSpec(parallel.axis_name, None)),
        out_specs=(PartitionSpec(parallel.axis_name), PartitionSpec(parallel.axis_name)),
    )
    def local_stateful_selfplay(model: nnx.Module, env_state: Any, device_rng_keys: jax.Array):
        rng_key = device_rng_keys[0]
        if not persistent:
            rng_key, init_key = jax.random.split(rng_key)
            env_state = jax.vmap(env.init)(jax.random.split(init_key, local_batch_size))
        if use_fast_hex:
            return _play_training_fast_hex(
                env,
                model,
                env_state,
                rng_key,
                batch_size=local_batch_size,
                max_num_steps=max_num_steps,
                config=config,
            )
        player = training_player(model)
        step_keys = jax.random.split(rng_key, max_num_steps)
        next_state, data = jax.lax.scan(
            lambda st, k: _selfplay_step_frame(env, player, st, k, batch_size=local_batch_size),
            env_state,
            step_keys,
        )
        data = jax.tree_util.tree_map(
            lambda leaf: leaf.swapaxes(0, 1) if isinstance(leaf, jax.Array) else leaf,
            data,
        )
        return next_state, data

    @nnx.jit
    def run_stateful_selfplay(model: nnx.Module, env_state: Any, rng_key: jax.Array):
        device_rng_keys = jax.random.split(rng_key, parallel.device_count)
        return local_stateful_selfplay(model, env_state, device_rng_keys)

    persistent_box: list[Any] = [None]

    def selfplay(model: nnx.Module, rng_key: jax.Array) -> TrainingSamples:
        if persistent_box[0] is None:
            rng_key, init_key = jax.random.split(rng_key)
            persistent_box[0] = jit_init_states(init_key)
        next_state, data = run_stateful_selfplay(model, persistent_box[0], rng_key)
        if persistent:
            persistent_box[0] = next_state
        return data

    return selfplay
