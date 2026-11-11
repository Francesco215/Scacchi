"""Thompson tree search on binned beliefs with the weighted-sum backup.

Reuses the Dirichlet tree for topology and exact proof propagation.
"""

import functools

import jax
import jax.numpy as jnp

from . import binned
from .dirichlet_mctx import RootFnOutput
from .dirichlet_mctx.action_selection import categorical_action, categorical_action_population, effective_action_alpha
from .dirichlet_mctx.native_targets import TARGET_BINNED
from .dirichlet_mctx.posterior_updates import _repair_inputs
from .dirichlet_mctx.search import search as tree_search
from .dirichlet_mctx.tree import PosteriorUpdate
from .dirichlet_q_search import make_dirichlet_expand_fn, terminal_outcome_from_reward


def action_selection(key, tree, node):
    invalid = tree.invalid_actions[node] | (tree.edge_categorical_outcome[node] >= 0)
    return binned.sample_action(key, effective_action_alpha(tree, node), invalid)


def only_repaired(compute, repaired, *operands):
    """``compute(*operands)`` on the lanes in ``repaired``; zeros elsewhere.

    A backward step visits one path node in every lane, but a lane whose path
    has already ended keeps its old value and its result would be discarded.
    The lanes that do need a repair are moved to the front and ``compute``
    runs on the smallest of a few fixed prefixes that holds them all. A lane's
    result does not depend on which prefix it was computed in.
    """
    lanes = repaired.shape[0]
    sizes = sorted({max(1, lanes >> halvings) for halvings in range(5)})
    order = jnp.argsort(~repaired, stable=True)
    count = jnp.sum(repaired)
    full = jax.eval_shape(compute, *operands)

    def on_prefix(size):
        def run(_):
            if size == lanes:
                return compute(*operands)
            rows = order[:size]
            return jnp.zeros(full.shape, full.dtype).at[rows].set(compute(*(operand[rows] for operand in operands)))

        return run

    branches = [lambda _: jnp.zeros(full.shape, full.dtype), *(on_prefix(size) for size in sizes)]
    # Branch 0 when no lane is repaired, otherwise the smallest prefix >= count.
    index = jnp.sum(count > jnp.asarray([0, *sizes[:-1]]))
    result = jax.lax.switch(index, branches, None)
    return jnp.where(repaired.reshape((lanes,) + (1,) * (result.ndim - 1)), result, 0)


def update_posterior(key, context, *, points):
    """Node belief = law of ``sum_a pi(a) X_a`` over its current edge beliefs.

    ``pi`` is the probability that each move is best. There are no counts, no
    prior strength and no blend with the node's own value prediction.
    """
    del key
    prepared = _repair_inputs(context)
    edges = binned.categorical_edges(prepared.effective_alpha, context.node.edge_categorical_outcome)
    repaired = context.active & (prepared.n_down > 0)
    value = only_repaired(
        lambda edges, invalid: binned.weighted_sum(edges, binned.best_policy(edges, invalid), points),
        repaired, edges, context.node.invalid_actions,
    )
    value = jnp.where(repaired[..., None], value, context.node.value_alpha)
    return PosteriorUpdate(edge_alpha=prepared.edge_alpha, edge_payload=prepared.edge_payload, value_alpha=value)


def make_binned_search(env, evaluator, config, q_supervision_config):
    from .play_search import _normalize_policy_on_support, _root_policy_support, targets_from_summary

    expand = make_dirichlet_expand_fn(env, evaluator, num_outcomes=2)

    def search(state, key):
        prediction = evaluator(state.observation)
        repair = functools.partial(update_posterior, points=prediction.support)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), state.current_player]
        root = RootFnOutput(
            prior_logits=prediction.logits,
            value=prediction.alpha_v,
            action_values=prediction.alpha_q,
            embedding=state,
            terminal_outcome=jnp.where(state.terminated, terminal_outcome_from_reward(reward, 2), jnp.int8(-1)),
            to_play=state.current_player,
        )
        search_key, tie_key = jax.random.split(key)
        tree = tree_search(
            params=(), rng_key=search_key, root=root, recurrent_fn=expand,
            action_selection_fn=action_selection, posterior_update=repair,
            num_simulations=config.num_simulations, max_depth=config.max_depth,
            invalid_actions=~state.legal_action_mask, num_outcomes=2,
        )
        summary = tree.summary()
        support = _root_policy_support(state.legal_action_mask, summary.visit_counts > 0, summary.q_categorical_outcome, config.root_policy_support)
        probs = binned.categorical_edges(summary.alpha, summary.q_categorical_outcome)
        policy = binned.best_policy(probs, ~support)
        solved_policy = categorical_action_population(
            summary.v_categorical_outcome, summary.q_categorical_outcome,
            summary.q_categorical_distance, ~support, num_outcomes=2,
        )
        solved = summary.v_categorical_outcome >= 0
        policy = jnp.where(solved[:, None], solved_policy, policy)
        policy = _normalize_policy_on_support(policy, support, temperature=config.policy_target_temperature)
        solved_action = categorical_action(
            tie_key, summary.v_categorical_outcome, summary.q_categorical_outcome,
            summary.q_categorical_distance, ~support, num_outcomes=2,
        )
        action = jnp.where(solved, solved_action, jnp.argmax(policy, axis=-1)).astype(jnp.int32)
        return targets_from_summary(summary, policy, action, state.legal_action_mask, q_supervision_config, unresolved_kind=TARGET_BINNED)

    return search
