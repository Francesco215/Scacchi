"""Binned Thompson search on Hex with one fused CUDA kernel per simulation.

``fused_search`` returns the same tree as ``dirichlet_mctx.search.search`` with
``binned_search.action_selection`` and ``binned_search.update_posterior``
(``kernel="cuda"``): same keys, same draws, same arithmetic. Between two
kernel calls the network evaluates the leaves (XLA, unchanged); each call
writes the evaluated leaves into the tree, repairs every lane's path to the
root, then descends the next simulation and plays its Hex move, every lane on
its own (``search.cu``).

Hex only (pgx ``hex``; at most 127 cells), binary outcomes, CUDA GPUs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..dirichlet_mctx.base import RootFnOutput
from ..dirichlet_mctx.tree import Tree, instantiate_tree_from_root
from . import _quadrature

PEND = 8
FINISH = 1
DESCEND = 2


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
    """The binned search tree after ``num_simulations`` simulations from ``state``.

    ``state`` is the batched pgx Hex state at the root and ``root`` its network
    evaluation (``embedding`` is not used). ``evaluator(observation)`` must
    return ``alpha_v`` [B, K] and ``alpha_q`` [B, A, K] like
    ``play_search.make_evaluator``.

    With ``cohorts=2`` the lanes are split in two halves that are one phase
    apart: while the network evaluates one half's leaves, the kernel finishes
    and descends the other half, and XLA can run the two at the same time
    (with ``--xla_gpu_command_buffer_scheduling_mode=CONCURRENT``). The tree of
    a lane is then the same as with ``cohorts=1`` as long as the network gives
    a position the same bits at half the batch size. cuDNN picks its
    convolution algorithm per shape: at 4096 lanes this held on Hex 7 and 9
    but not on Hex 8 and 11 (outputs differ by ~1e-6), nor at 512 lanes.
    """
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
            # One network at a time: this cohort's evaluation waits for the
            # previous one, so the step kernel beside it has the other half.
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

    # Same key chain as dirichlet_mctx.search.search: four keys per simulation,
    # the second split into one key per lane for the descent.
    carries = [(_cohort(tree, part), boards[part], steps[part], pending[part]) for part in parts]
    key, simulate_key, _, _ = jax.random.split(rng_key, 4)
    observations = []
    for c in range(cohorts):
        carries[c], obs = step(c, carries[c], (no_value, no_q), lane_keys(simulate_key, c), 0, DESCEND)
        observations.append(obs)
    # Software pipeline: cohort c is finished and descended while the network
    # evaluates cohort c + 1, which is one phase behind.
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
