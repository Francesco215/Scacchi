from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from scacchi import dirichlet_mctx
from scacchi.dirichlet_mctx.action_selection import effective_action_alpha
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME, align_categorical_outcome
from scacchi.dirichlet_mctx.tree import instantiate_tree_from_root


def test_align_categorical_outcome_flips_exact_indices_and_preserves_sentinel():
    outcome = jnp.asarray([int(NO_OUTCOME), 0, 1, 2], dtype=jnp.int8)
    source_player = jnp.zeros((4,), dtype=jnp.int32)
    target_player = jnp.ones((4,), dtype=jnp.int32)

    aligned = align_categorical_outcome(outcome, source_player, target_player, 3)

    assert jnp.array_equal(aligned, jnp.asarray([int(NO_OUTCOME), 2, 1, 0], dtype=jnp.int8))


def test_effective_action_alpha_uses_one_shared_precedence_rule():
    tree = SimpleNamespace(
        children_index=jnp.asarray([[0, 1, -1, 3]], dtype=jnp.int32),
        node_value_priors=jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        node_to_play=jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        edge_alpha=jnp.asarray([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]]),
        edge_payload=jnp.asarray([[1, 0, 0, 7]], dtype=jnp.int32),
        edge_categorical_outcome=jnp.asarray([[int(NO_OUTCOME), int(NO_OUTCOME), int(NO_OUTCOME), 1]], dtype=jnp.int8),
    )

    effective = effective_action_alpha(tree, jnp.asarray(0, dtype=jnp.int32))

    expected = jnp.asarray([[10.0, 11.0], [4.0, 3.0], [30.0, 31.0], [40.0, 41.0]])
    assert jnp.array_equal(effective, expected)


def test_root_action_alpha_is_effective_and_stored_slot_is_explicit():
    root = dirichlet_mctx.RootFnOutput(prior_logits=jnp.zeros((1, 1), dtype=jnp.float32), value=jnp.ones((1, 2), dtype=jnp.float32), action_values=jnp.asarray([[[10.0, 11.0]]]), embedding=jnp.zeros((1,), dtype=jnp.int32), terminal_outcome=jnp.asarray([int(NO_OUTCOME)], dtype=jnp.int8), to_play=jnp.zeros((1,), dtype=jnp.int32))
    tree = instantiate_tree_from_root(root, 1, jnp.asarray([[False]]))
    tree = replace(tree, children_index=tree.children_index.at[0, 0, 0].set(1), node_value_priors=tree.node_value_priors.at[0, 1].set(jnp.asarray([2.0, 3.0])), node_to_play=tree.node_to_play.at[0, 1].set(1))

    assert jnp.array_equal(tree.root_stored_edge_alpha, jnp.asarray([[[10.0, 11.0]]]))
    assert jnp.array_equal(tree.root_action_alpha, jnp.asarray([[[3.0, 2.0]]]))
    assert jnp.array_equal(tree.summary().alpha, tree.root_action_alpha)


def test_zero_simulation_policy_does_not_trace_search_callbacks():
    root = dirichlet_mctx.RootFnOutput(prior_logits=jnp.zeros((1, 2), dtype=jnp.float32), value=jnp.ones((1, 3), dtype=jnp.float32), action_values=jnp.ones((1, 2, 3), dtype=jnp.float32), embedding=jnp.zeros((1,), dtype=jnp.int32), terminal_outcome=jnp.asarray([int(NO_OUTCOME)], dtype=jnp.int8), to_play=jnp.zeros((1,), dtype=jnp.int32))

    def recurrent_fn(*_):
        raise AssertionError("zero-simulation search traced recurrent_fn")

    def loop_fn(*_):
        raise AssertionError("zero-simulation search called loop_fn")

    output = dirichlet_mctx.dirichlet_thompson_policy(params=(), rng_key=jax.random.PRNGKey(0), root=root, recurrent_fn=recurrent_fn, num_simulations=0, invalid_actions=jnp.asarray([[False, False]]), policy_samples=0, loop_fn=loop_fn)

    assert output.search_tree.num_simulations == 0
    assert output.action_weights.shape == (1, 2)
    assert jnp.allclose(jnp.sum(output.action_weights, axis=-1), 1.0)
