"""Binned Thompson search on Hex: bitwise-reference fused step and stateful fast megakernel."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..dirichlet_mctx.base import RootFnOutput
from ..dirichlet_mctx.tree import SearchSummary, Tree, instantiate_tree_from_root
from . import _fast_quadrature, _quadrature
from .network import PackedNetwork, PositionCache, apply_compact

PEND = 8
FINISH = 1
DESCEND = 2
F_SUBTREE_INIT = 4
F_ROOT_FINISH = 8
F_READOUT = 16


class FastTreeCarry(NamedTuple):
    """Persistent per-lane tree state carried across consecutive moves of a rollout."""

    parents: jax.Array       # [B, N] i32
    children: jax.Array      # [B, N, A] i32
    to_play: jax.Array       # [B, N] i32
    node_out: jax.Array      # [B, N] i8
    node_pay: jax.Array      # [B, N] i32
    edge_out: jax.Array      # [B, N, A] i8
    edge_pay: jax.Array      # [B, N, A] i32
    vprior: jax.Array        # [B, N, K] f32
    valpha: jax.Array        # [B, N, K] f32
    ealpha: jax.Array        # [B, N, A, K] f32
    invalid: jax.Array       # [B, N, A] bool
    boards: jax.Array        # [B, N, C] i8
    steps: jax.Array         # [B, N] i32
    pending: jax.Array       # [B, 8] i32
    prev_actions: jax.Array  # [B] i32 (-1 before first move)
    was_reset: jax.Array     # [B] bool (True before first move or after game termination)


def empty_fast_tree_carry(batch: int, num_simulations: int, num_actions: int, atoms: int, cells: int) -> FastTreeCarry:
    nodes = num_simulations + 1
    return FastTreeCarry(
        parents=jnp.full((batch, nodes), -1, jnp.int32),
        children=jnp.full((batch, nodes, num_actions), -1, jnp.int32),
        to_play=jnp.zeros((batch, nodes), jnp.int32),
        node_out=jnp.full((batch, nodes), -1, jnp.int8),
        node_pay=jnp.zeros((batch, nodes), jnp.int32),
        edge_out=jnp.full((batch, nodes, num_actions), -1, jnp.int8),
        edge_pay=jnp.zeros((batch, nodes, num_actions), jnp.int32),
        vprior=jnp.zeros((batch, nodes, atoms), jnp.float32),
        valpha=jnp.zeros((batch, nodes, atoms), jnp.float32),
        ealpha=jnp.zeros((batch, nodes, num_actions, atoms), jnp.float32),
        invalid=jnp.ones((batch, nodes, num_actions), jnp.bool_),
        boards=jnp.zeros((batch, nodes, cells), jnp.int8),
        steps=jnp.zeros((batch, nodes), jnp.int32),
        pending=jnp.zeros((batch, PEND), jnp.int32),
        prev_actions=jnp.full((batch,), -1, jnp.int32),
        was_reset=jnp.ones((batch,), jnp.bool_),
    )


def _key_data(keys: jax.Array) -> jax.Array:
    if jnp.issubdtype(keys.dtype, jax.dtypes.prng_key):
        keys = jax.random.key_data(keys)
    return keys.astype(jnp.uint32)


def _step(tree: Tree, boards, steps, pending, net_value, net_q, keys, order, points, sim, *, max_depth: int, flags: int):
    from ._library import SEARCH_TARGET, register_search

    register_search()
    batch, nodes, actions, atoms = tree.edge_alpha.shape
    cells = boards.shape[-1]
    size = int(round(cells ** 0.5))
    quad_nodes, quad_weights = _quadrature(actions)
    state = (
        tree.parents, tree.children_index, tree.node_to_play, tree.node_categorical_outcome, tree.node_payload,
        tree.edge_categorical_outcome, tree.edge_payload, tree.node_value_priors, tree.node_value_alpha,
        tree.edge_alpha, tree.invalid_actions, boards, steps, pending,
    )
    results = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in state]
    results.append(jax.ShapeDtypeStruct((batch, size, size, 4), jnp.bool_))
    call = jax.ffi.ffi_call(SEARCH_TARGET, results, input_output_aliases={i: i for i in range(len(state))})
    out = call(
        *state, net_value.astype(jnp.float32), net_q.astype(jnp.float32), _key_data(keys), order.astype(jnp.int32),
        jnp.asarray(points, jnp.float32), jnp.asarray(quad_nodes), jnp.asarray(quad_weights), jnp.asarray(sim, jnp.int32),
        max_depth=np.int64(max_depth), flags=np.int64(flags),
    )
    tree = tree.replace(
        parents=out[0], children_index=out[1], node_to_play=out[2], node_categorical_outcome=out[3], node_payload=out[4],
        edge_categorical_outcome=out[5], edge_payload=out[6], node_value_priors=out[7], node_value_alpha=out[8],
        edge_alpha=out[9], invalid_actions=out[10],
    )
    return tree, out[11], out[12], out[13], out[14]


def _cohort(tree: Tree, lanes: slice) -> Tree:
    return jax.tree.map(lambda x: x[lanes], tree)


def fused_search(evaluator, state, root: RootFnOutput, rng_key: jax.Array, *, points, num_simulations: int,
                 max_depth: int | None = None, cohorts: int = 1) -> Tree:
    """Bitwise-identical reference fused search step."""
    if max_depth is None:
        max_depth = num_simulations
    tree = instantiate_tree_from_root(root.replace(embedding=()), num_simulations, ~state.legal_action_mask)
    if num_simulations == 0:
        return tree
    batch, nodes, actions, atoms = tree.edge_alpha.shape
    if cohorts < 1 or batch % cohorts:
        cohorts = 1
    board = state._x.board
    boards = jnp.zeros((batch, nodes, board.shape[-1]), jnp.int8).at[:, 0].set(board.astype(jnp.int8))
    steps = jnp.zeros((batch, nodes), jnp.int32).at[:, 0].set(state._x.step_count.astype(jnp.int32))
    pending = jnp.zeros((batch, PEND), jnp.int32)
    order = state._player_order
    width = batch // cohorts
    parts = [slice(c * width, (c + 1) * width) for c in range(cohorts)]
    no_value = jnp.zeros((width, atoms), jnp.float32)
    no_q = jnp.zeros((width, actions, atoms), jnp.float32)
    no_keys = jnp.zeros((width, 2), jnp.uint32)

    def evaluate(observation, carry, after=None):
        active = carry[3][:, 2] != 0
        if after is not None:
            observation, active, _ = jax.lax.optimization_barrier((observation, active, after))

        def run(_):
            prediction = evaluator(observation)
            return prediction.alpha_v.astype(jnp.float32), prediction.alpha_q.astype(jnp.float32)

        return jax.lax.cond(jnp.any(active), run, lambda _: (no_value, no_q), None)

    def step(c, carry, net, keys, sim, flags):
        tree, boards, steps, pending = carry
        tree, boards, steps, pending, observation = _step(
            tree, boards, steps, pending, net[0], net[1], keys, order[parts[c]], points, sim,
            max_depth=max_depth, flags=flags,
        )
        return (tree, boards, steps, pending), observation

    def lane_keys(simulate_key, c):
        return jax.random.split(simulate_key, batch)[parts[c]]

    carries = [(_cohort(tree, part), boards[part], steps[part], pending[part]) for part in parts]
    key, simulate_key, _, _ = jax.random.split(rng_key, 4)
    observations = []
    for c in range(cohorts):
        carries[c], obs = step(c, carries[c], (no_value, no_q), lane_keys(simulate_key, c), 0, DESCEND)
        observations.append(obs)
    net0 = evaluate(observations[0], carries[0])

    def body(index, loop):
        key, carries, observations, net0 = loop
        carries, observations = list(carries), list(observations)
        key, simulate_key, _, _ = jax.random.split(key, 4)
        net = net0
        for c in range(cohorts):
            following = (c + 1) % cohorts
            if following != 0:
                next_net = evaluate(observations[following], carries[following], after=net)
            carries[c], observations[c] = step(c, carries[c], net, lane_keys(simulate_key, c), index, FINISH | DESCEND)
            if following == 0:
                next_net = evaluate(observations[0], carries[0], after=net if cohorts > 1 else None)
            net = next_net
        return key, tuple(carries), tuple(observations), net

    key, carries, observations, net0 = jax.lax.fori_loop(
        1, num_simulations, body, (key, tuple(carries), tuple(observations), net0))
    carries = list(carries)
    net = net0
    for c in range(cohorts):
        following = c + 1
        next_net = evaluate(observations[following], carries[following], after=net) if following < cohorts else None
        carries[c], _ = step(c, carries[c], net, no_keys, num_simulations, FINISH)
        net = next_net
    if cohorts == 1:
        return carries[0][0]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *(carry[0] for carry in carries))


def fused_fast_search(
    packed: PackedNetwork,
    state,
    rng_key: jax.Array,
    *,
    points,
    num_simulations: int,
    max_depth: int | None = None,
    tree_carry: FastTreeCarry | None = None,
    cache: PositionCache,
    lane_caps: jax.Array | None = None,
    subtree_reuse: bool = True,
    use_cache: bool = True,
    early_stop_prob: float = 0.96,
    min_stop_nodes: int = 10,
    support_search_evidence: bool = True,
    policy_target_temperature: float = 1.0,
) -> tuple[SearchSummary, jax.Array, jax.Array, FastTreeCarry, PositionCache]:
    """Full megakernel search step: subtree reuse + dynamic compaction + board-resident INT8 + fused readout."""
    from ..dirichlet_q_search import terminal_outcome_from_reward
    from ._library import FAST_SEARCH_TARGET, register_fast_search

    register_fast_search()
    if max_depth is None:
        max_depth = num_simulations

    batch = state.current_player.shape[0]
    cells = state._x.board.shape[-1]
    size = int(round(cells ** 0.5))
    actions = cells + 1
    atoms = int(points.shape[-1])
    nodes = num_simulations + 1

    if tree_carry is None or not subtree_reuse:
        tree_carry = empty_fast_tree_carry(batch, num_simulations, actions, atoms, cells)

    if lane_caps is None:
        lane_caps = jnp.full((batch,), nodes, jnp.int32)
    else:
        lane_caps = lane_caps.astype(jnp.int32)

    quad_nodes, quad_weights = _fast_quadrature(actions)
    reward = state.rewards[jnp.arange(batch), state.current_player]
    root_term_out = jnp.where(state.terminated, terminal_outcome_from_reward(reward, 2), jnp.int8(-1))
    root_boards = state._x.board.astype(jnp.int8)
    root_steps = state._x.step_count.astype(jnp.int32)
    root_to_play = state.current_player.astype(jnp.int32)
    root_invalid = ~state.legal_action_mask
    root_obs = state.observation.astype(jnp.bool_)
    order = state._player_order.astype(jnp.int32)
    reset_mask = tree_carry.was_reset if subtree_reuse else jnp.ones((batch,), jnp.bool_)
    prev_actions = tree_carry.prev_actions if subtree_reuse else jnp.full((batch,), -1, jnp.int32)

    tree_arrays = (
        tree_carry.parents, tree_carry.children, tree_carry.to_play, tree_carry.node_out,
        tree_carry.node_pay, tree_carry.edge_out, tree_carry.edge_pay, tree_carry.vprior,
        tree_carry.valpha, tree_carry.ealpha, tree_carry.invalid, tree_carry.boards,
        tree_carry.steps, tree_carry.pending,
    )
    tree_specs = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in tree_arrays]
    extra_specs = [
        jax.ShapeDtypeStruct((batch, size, size, 4), jnp.bool_),   # obs_compact
        jax.ShapeDtypeStruct((1,), jnp.int32),                     # d_num_active
        jax.ShapeDtypeStruct((batch, 10), jnp.uint32),             # miss_keys (40B/entry = CacheKey)
        jax.ShapeDtypeStruct((batch, actions), jnp.float32),       # out_policy
        jax.ShapeDtypeStruct((batch, actions, atoms), jnp.float32),# out_qalpha
        jax.ShapeDtypeStruct((batch, atoms), jnp.float32),         # out_valpha
        jax.ShapeDtypeStruct((batch, actions), jnp.float32),       # out_visits
        jax.ShapeDtypeStruct((batch, actions), jnp.int8),          # out_q_cat_out
        jax.ShapeDtypeStruct((batch, actions), jnp.int32),         # out_q_cat_dist
        jax.ShapeDtypeStruct((batch,), jnp.int8),                  # out_v_cat_out
        jax.ShapeDtypeStruct((batch,), jnp.int32),                 # out_v_cat_dist
        jax.ShapeDtypeStruct((batch,), jnp.int32),                 # out_search_action
    ]
    call = jax.ffi.ffi_call(
        FAST_SEARCH_TARGET,
        [*tree_specs, *extra_specs],
        input_output_aliases={i: i for i in range(14)},
    )

    no_v = jnp.zeros((batch, atoms), jnp.float32)
    no_q = jnp.zeros((batch, actions, atoms), jnp.float32)
    no_keys = jnp.zeros((batch, 2), jnp.uint32)
    pts_arr = jnp.asarray(points, jnp.float32)
    qn_arr = jnp.asarray(quad_nodes, jnp.float32)
    qw_arr = jnp.asarray(quad_weights, jnp.float32)

    def invoke(ta, net_v, net_q, keys_u32, cache_obj: PositionCache, flags: int):
        out = call(
            *ta,
            net_v, net_q, keys_u32, order, pts_arr, qn_arr, qw_arr,
            prev_actions, reset_mask, root_boards, root_steps, root_to_play, root_term_out,
            root_invalid, root_obs, lane_caps,
            cache_obj.keys, cache_obj.locks, cache_obj.v, cache_obj.q,
            max_depth=np.int64(max_depth),
            flags=np.int64(flags),
            min_stop_nodes=np.int64(min_stop_nodes),
            use_cache=np.int64(1 if use_cache else 0),
            support_search_evidence=np.int64(1 if support_search_evidence else 0),
            early_stop_prob=float(early_stop_prob),
            policy_temp=float(policy_target_temperature),
        )
        return tuple(out[:14]), out[14], out[15], out[16], out[17:]

    search_key, tie_key = jax.random.split(rng_key)
    tie_keys_u32 = _key_data(jax.random.split(tie_key, batch))

    # 1. Subtree reuse / cold root compaction + compact root network eval (only reset lanes!)
    ta, obs_c, d_active, miss_k, _ = invoke(tree_arrays, no_v, no_q, no_keys, cache, F_SUBTREE_INIT)
    net_v, net_q, cache = apply_compact(
        packed, obs_c, d_active, miss_k, cache, actions, atoms,
        use_cache=use_cache, small_hint=subtree_reuse,
    )

    if num_simulations == 0:
        ta, _, _, _, ro = invoke(ta, net_v, net_q, tie_keys_u32, cache, F_ROOT_FINISH | F_READOUT)
    else:
        key, sim0_key = jax.random.split(search_key)
        sim0_u32 = _key_data(jax.random.split(sim0_key, batch))
        ta, obs_c, d_active, miss_k, _ = invoke(ta, net_v, net_q, sim0_u32, cache, F_ROOT_FINISH | DESCEND)
        net_v, net_q, cache = apply_compact(
            packed, obs_c, d_active, miss_k, cache, actions, atoms,
            use_cache=use_cache, small_hint=False,
        )

        def loop_body(_, carry):
            key, ta, net_v, net_q, cache_c = carry
            key, sim_key = jax.random.split(key)
            sim_u32 = _key_data(jax.random.split(sim_key, batch))
            ta, obs_c, d_active, miss_k, _ = invoke(ta, net_v, net_q, sim_u32, cache_c, FINISH | DESCEND)
            net_v, net_q, cache_c = apply_compact(
                packed, obs_c, d_active, miss_k, cache_c, actions, atoms,
                use_cache=use_cache, small_hint=False,
            )
            return key, ta, net_v, net_q, cache_c

        if num_simulations > 1:
            _, ta, net_v, net_q, cache = jax.lax.fori_loop(1, num_simulations, loop_body, (key, ta, net_v, net_q, cache))

        ta, _, _, _, ro = invoke(ta, net_v, net_q, tie_keys_u32, cache, FINISH | F_READOUT)

    policy, qalpha, valpha, visits, q_cat_out, q_cat_dist, v_cat_out, v_cat_dist, search_action = ro
    summary = SearchSummary(
        visit_counts=visits,
        alpha=qalpha,
        value_alpha=valpha,
        q_categorical_outcome=q_cat_out,
        q_categorical_distance=q_cat_dist,
        v_categorical_outcome=v_cat_out,
        v_categorical_distance=v_cat_dist,
    )
    next_carry = FastTreeCarry(
        *ta,
        prev_actions=tree_carry.prev_actions,
        was_reset=tree_carry.was_reset,
    )
    return summary, policy, search_action, next_carry, cache
