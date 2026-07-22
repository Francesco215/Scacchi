from dataclasses import replace

import jax
import jax.numpy as jnp

from scacchi import dirichlet_mctx
from scacchi.dirichlet_mctx import action_selection
from scacchi.dirichlet_mctx.outcomes import NO_DISTANCE, NO_OUTCOME
from scacchi.dirichlet_mctx.search import simulate
from scacchi.dirichlet_mctx.utils import _categorize_node_and_publish


def _root(
    num_actions: int,
    *,
    num_outcomes: int = 3,
    prior_logits: tuple[float, ...] | None = None,
) -> dirichlet_mctx.RootFnOutput:
    if prior_logits is None:
        prior_logits = (0.0,) * num_actions
    return dirichlet_mctx.RootFnOutput(
        prior_logits=jnp.asarray([prior_logits], dtype=jnp.float32),
        value=jnp.ones((1, num_outcomes), dtype=jnp.float32),
        action_values=jnp.ones(
            (1, num_actions, num_outcomes),
            dtype=jnp.float32,
        ),
        embedding=jnp.zeros((1,), dtype=jnp.int32),
        terminal_outcome=jnp.full(
            (1,),
            int(NO_OUTCOME),
            dtype=jnp.int8,
        ),
        to_play=jnp.zeros((1,), dtype=jnp.int32),
    )


def _categorize_root(
    edge_outcome: tuple[int, ...],
    edge_distance: tuple[int, ...],
    *,
    invalid_actions: tuple[bool, ...] | None = None,
    rng_key: jax.Array = jax.random.PRNGKey(0),
):
    num_actions = len(edge_outcome)
    if invalid_actions is None:
        invalid_actions = (False,) * num_actions
    invalid = jnp.asarray([invalid_actions], dtype=bool)
    tree = dirichlet_mctx.instantiate_tree_from_root(
        _root(num_actions),
        num_simulations=0,
        root_invalid_actions=invalid,
    )
    tree = replace(
        tree,
        edge_categorical_outcome=tree.edge_categorical_outcome.at[0, 0].set(
            jnp.asarray(edge_outcome, dtype=jnp.int8)
        ),
        edge_payload=tree.edge_payload.at[0, 0].set(
            jnp.asarray(edge_distance, dtype=jnp.int32)
        ),
    )
    tree = _categorize_node_and_publish(
        rng_key,
        tree,
        jnp.asarray([tree.ROOT_INDEX], dtype=jnp.int32),
        jnp.asarray([True]),
    )
    outcome = tree.node_categorical_outcome[:, tree.ROOT_INDEX]
    action = action_selection.categorical_action(
        rng_key,
        outcome,
        tree.edge_categorical_outcome[:, tree.ROOT_INDEX],
        tree.edge_payload[:, tree.ROOT_INDEX],
        invalid,
        num_outcomes=3,
    )
    return tree, action


def test_win_certificate_uses_shortest_certified_edge_despite_unresolved_moves():
    tree, action = _categorize_root(
        (2, int(NO_OUTCOME), 2, 0),
        (5, int(NO_DISTANCE), 2, 9),
    )

    assert tree.node_categorical_outcome[0, 0] == 2
    assert tree.node_payload[0, 0] == 2
    assert action[0] == 2


def test_loss_requires_all_legal_edges_and_uses_longest_defeat():
    unresolved, _ = _categorize_root(
        (0, int(NO_OUTCOME), 0),
        (2, int(NO_DISTANCE), 8),
    )
    solved, action = _categorize_root(
        (0, 0, int(NO_OUTCOME)),
        (2, 8, int(NO_DISTANCE)),
        invalid_actions=(False, False, True),
    )

    assert unresolved.node_categorical_outcome[0, 0] == int(NO_OUTCOME)
    assert unresolved.summary().v_categorical_distance[0] == int(NO_DISTANCE)
    assert solved.node_categorical_outcome[0, 0] == 0
    assert solved.node_payload[0, 0] == 8
    assert action[0] == 1


def test_draw_certificate_samples_uniformly_only_from_draw_edges():
    node_outcome = jnp.asarray([1], dtype=jnp.int8)
    edge_outcome = jnp.asarray([[1, 0, 1, 0]], dtype=jnp.int8)
    edge_distance = jnp.asarray([[2, 50, 7, 100]], dtype=jnp.int32)
    invalid = jnp.zeros((1, 4), dtype=bool)
    keys = jax.random.split(jax.random.PRNGKey(7), 128)

    actions = jax.vmap(
        lambda key: action_selection.categorical_action(
            key,
            node_outcome,
            edge_outcome,
            edge_distance,
            invalid,
            num_outcomes=3,
        )[0]
    )(keys)

    assert set(map(int, actions.tolist())) == {0, 2}
    counts = jnp.bincount(actions, length=4)
    assert abs(int(counts[0]) - int(counts[2])) < 40

    tree, action = _categorize_root(
        (1, 0, 1, 0),
        (2, 50, 7, 100),
        rng_key=jax.random.PRNGKey(7),
    )
    assert tree.node_categorical_outcome[0, 0] == 1
    assert int(action[0]) in {0, 2}
    assert tree.node_payload[0, 0] == tree.edge_payload[0, 0, action[0]]


def test_draw_requires_every_legal_edge_to_be_draw_or_loss():
    tree, _ = _categorize_root(
        (1, 0, int(NO_OUTCOME)),
        (3, 4, int(NO_DISTANCE)),
    )

    assert tree.node_categorical_outcome[0, 0] == int(NO_OUTCOME)
    assert tree.summary().v_categorical_distance[0] == int(NO_DISTANCE)


def test_categorical_certificates_are_absorbing():
    tree, _ = _categorize_root((2, 0), (4, 8))
    tree = replace(
        tree,
        edge_payload=tree.edge_payload.at[0, 0, 0].set(1),
    )
    tree = _categorize_node_and_publish(
        jax.random.PRNGKey(1),
        tree,
        jnp.asarray([tree.ROOT_INDEX], dtype=jnp.int32),
        jnp.asarray([True]),
    )

    assert tree.node_categorical_outcome[0, 0] == 2
    assert tree.node_payload[0, 0] == 4


def test_simulation_masks_categorical_edges_for_custom_selectors():
    invalid = jnp.asarray([[False, False]])
    tree = dirichlet_mctx.instantiate_tree_from_root(
        _root(2),
        num_simulations=0,
        root_invalid_actions=invalid,
    )
    tree = replace(
        tree,
        edge_categorical_outcome=tree.edge_categorical_outcome.at[0, 0, 0].set(
            0
        ),
        edge_payload=tree.edge_payload.at[0, 0, 0].set(1),
    )

    def prefers_first_action(rng_key, candidate_tree, node_index):
        del rng_key
        return action_selection.masked_argmax(
            jnp.asarray([1.0, 0.0]),
            candidate_tree.invalid_actions[node_index],
        )

    simulation = simulate(
        jax.random.split(jax.random.PRNGKey(3), 1),
        tree,
        prefers_first_action,
        1,
    )

    assert simulation.active[0]
    assert simulation.action[0] == 1


def test_terminal_chain_propagates_perspective_and_distance_and_stops_root():
    terminal_depth = 3

    def recurrent_fn(_, rng_key, action, depth):
        del rng_key, action
        child_depth = depth + 1
        batch_size = child_depth.shape[0]
        terminal = child_depth >= terminal_depth
        value = jnp.ones((batch_size, 3), dtype=jnp.float32)
        return (
            dirichlet_mctx.RecurrentFnOutput(
                value=value,
                action_values=jnp.ones(
                    (batch_size, 1, 3),
                    dtype=jnp.float32,
                ),
                invalid_actions=terminal[:, None],
                terminal_outcome=jnp.where(
                    terminal,
                    jnp.asarray(0, dtype=jnp.int8),
                    jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
                ),
                to_play=(child_depth % 2).astype(jnp.int32),
            ),
            child_depth,
        )

    def run(key):
        return dirichlet_mctx.dirichlet_thompson_policy(
            params=(),
            rng_key=key,
            root=_root(1),
            recurrent_fn=recurrent_fn,
            num_simulations=7,
            invalid_actions=jnp.asarray([[False]]),
            max_depth=7,
            policy_samples=4,
        )

    output = jax.jit(run)(jax.random.PRNGKey(11))
    tree = output.search_tree

    assert jnp.array_equal(
        tree.node_categorical_outcome[0, :4],
        jnp.asarray([2, 0, 2, 0], dtype=jnp.int8),
    )
    assert jnp.array_equal(
        tree.node_payload[0, :4],
        jnp.asarray([3, 2, 1, 0], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        tree.edge_categorical_outcome[0, :3, 0],
        jnp.asarray([2, 0, 2], dtype=jnp.int8),
    )
    assert jnp.array_equal(
        tree.edge_payload[0, :3, 0],
        jnp.asarray([3, 2, 1], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        tree.parents[0],
        jnp.asarray([-1, 0, 1, 2, -1, -1, -1, -1], dtype=jnp.int32),
    )
    assert output.action[0] == 0
    assert jnp.array_equal(output.action_weights, jnp.asarray([[1.0]]))
    summary = tree.summary()
    assert summary.v_categorical_outcome[0] == 2
    assert summary.v_categorical_distance[0] == 3
    assert summary.q_categorical_outcome[0, 0] == 2
    assert summary.q_categorical_distance[0, 0] == 3


def test_solved_batch_lane_stays_frozen_while_another_lane_keeps_searching():
    root = _root(1)
    root = replace(
        root,
        prior_logits=jnp.repeat(root.prior_logits, 2, axis=0),
        value=jnp.repeat(root.value, 2, axis=0),
        action_values=jnp.repeat(root.action_values, 2, axis=0),
        embedding=jnp.zeros((2,), dtype=jnp.int32),
        terminal_outcome=jnp.full(
            (2,),
            int(NO_OUTCOME),
            dtype=jnp.int8,
        ),
        to_play=jnp.zeros((2,), dtype=jnp.int32),
    )
    terminal_depth = jnp.asarray([1, 3], dtype=jnp.int32)

    def recurrent_fn(_, rng_key, action, depth):
        del rng_key, action
        child_depth = depth + 1
        terminal = child_depth >= terminal_depth
        value = jnp.ones((2, 3), dtype=jnp.float32)
        return (
            dirichlet_mctx.RecurrentFnOutput(
                value=value,
                action_values=jnp.ones((2, 1, 3), dtype=jnp.float32),
                invalid_actions=terminal[:, None],
                terminal_outcome=jnp.where(
                    terminal,
                    jnp.asarray(0, dtype=jnp.int8),
                    jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
                ),
                to_play=(child_depth % 2).astype(jnp.int32),
            ),
            child_depth,
        )

    output = jax.jit(
        lambda key: dirichlet_mctx.dirichlet_thompson_policy(
            params=(),
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=5,
            invalid_actions=jnp.zeros((2, 1), dtype=bool),
            max_depth=5,
            policy_samples=4,
        )
    )(jax.random.PRNGKey(17))
    tree = output.search_tree

    assert jnp.array_equal(
        tree.node_categorical_outcome[:, tree.ROOT_INDEX],
        jnp.asarray([2, 2], dtype=jnp.int8),
    )
    assert jnp.array_equal(
        tree.node_payload[:, tree.ROOT_INDEX],
        jnp.asarray([1, 3], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        tree.parents[0],
        jnp.asarray([-1, 0, -1, -1, -1, -1], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        tree.parents[1],
        jnp.asarray([-1, 0, 1, 2, -1, -1], dtype=jnp.int32),
    )
    assert jnp.array_equal(output.action_weights, jnp.ones((2, 1)))


def test_mixed_native_thompson_uses_exact_categorical_utility():
    contradictory_alpha = jnp.asarray(
        [[[1000.0, 1.0, 1.0], [1.0, 1.0, 1000.0]]],
        dtype=jnp.float32,
    )
    agreeing_alpha = jnp.asarray(
        [[[1.0, 1.0, 1000.0], [1.0, 1.0, 1000.0]]],
        dtype=jnp.float32,
    )
    categorical_outcome = jnp.asarray([[2, int(NO_OUTCOME)]], dtype=jnp.int8)

    def policy(alpha):
        return action_selection.posterior_best_policy(
            jax.random.PRNGKey(4),
            alpha,
            jnp.asarray([[False, False]]),
            num_samples=64,
            categorical_outcome=categorical_outcome,
        )

    # The unresolved Dirichlet draw has utility strictly below +1, so an exact
    # categorical win must win every comparison regardless of the stale alpha
    # occupying that edge's fixed-shape storage slot.
    expected = jnp.asarray([[1.0, 0.0]])
    assert jnp.array_equal(policy(contradictory_alpha), expected)
    assert jnp.array_equal(policy(agreeing_alpha), expected)
