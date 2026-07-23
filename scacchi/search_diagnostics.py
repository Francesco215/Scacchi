"""Generation-time diagnostics for search-based self-distillation.

The quantities in this module compare the exact network outputs which
initialized a root with the native targets emitted by that same search.  They
therefore measure search displacement, not learner fit and not information
gain in the Bayesian sense.  All logarithms are natural logarithms.

Every field in :class:`SearchDiagnostics` is additive.  A caller may sum it
over devices, games, and time and only then divide a ``*_sum`` by its matching
``*_count``.  This avoids averaging roots with different numbers of legal Q
targets as though they had equal action-level sample sizes.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
from jaxtyping import Array, Bool, Float

from .dirichlet_mctx.native_targets import (
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
)
from .dirichlet_mctx.tree import SearchSummary, Tree


class NativeDisplacement(NamedTuple):
    """Per-target native displacement and the masks defining its populations."""

    semantic_kl: Float[Array, "*target"]
    semantic_mask: Bool[Array, "*target"]
    dirichlet_kl: Float[Array, "*target"]
    dirichlet_mask: Bool[Array, "*target"]
    categorical_surprisal: Float[Array, "*target"]
    categorical_mask: Bool[Array, "*target"]


class SearchDiagnostics(NamedTuple):
    """Additive diagnostics produced for every batch lane at a search root."""

    search_policy_kl_sum: Float[Array, "batch"]
    search_policy_kl_count: Float[Array, "batch"]
    search_v_semantic_kl_sum: Float[Array, "batch"]
    search_v_semantic_kl_count: Float[Array, "batch"]
    search_v_dirichlet_kl_sum: Float[Array, "batch"]
    search_v_dirichlet_kl_count: Float[Array, "batch"]
    search_v_categorical_surprisal_sum: Float[Array, "batch"]
    search_v_categorical_surprisal_count: Float[Array, "batch"]
    search_q_semantic_kl_sum: Float[Array, "batch"]
    search_q_semantic_kl_count: Float[Array, "batch"]
    search_q_policy_semantic_kl_sum: Float[Array, "batch"]
    search_q_policy_semantic_kl_count: Float[Array, "batch"]
    search_q_dirichlet_kl_sum: Float[Array, "batch"]
    search_q_dirichlet_kl_count: Float[Array, "batch"]
    search_q_categorical_surprisal_sum: Float[Array, "batch"]
    search_q_categorical_surprisal_count: Float[Array, "batch"]
    search_root_count: Float[Array, "batch"]
    search_legal_action_count: Float[Array, "batch"]
    search_visited_action_count: Float[Array, "batch"]
    search_repaired_action_count: Float[Array, "batch"]
    search_categorical_action_count: Float[Array, "batch"]
    search_solved_root_count: Float[Array, "batch"]
    search_expanded_node_count: Float[Array, "batch"]
    search_simulation_active_count: Float[Array, "batch"]
    search_executed_simulation_row_count: Float[Array, "batch"]
    search_requested_simulation_count: Float[Array, "batch"]
    search_structural_support_sum: Float[Array, "batch"]
    search_max_depth_sum: Float[Array, "batch"]
    search_policy_support_sum: Float[Array, "batch"]
    search_policy_ess_sum: Float[Array, "batch"]
    search_policy_top1_agreement_count: Float[Array, "batch"]


class DistillationDiscrepancy(NamedTuple):
    """Additive target gaps for one fixed learner probe.

    The unweighted Q fields preserve action-level population means.  The
    ``q_weighted_*`` fields instead carry weighted numerators and total
    positive finite sample weight, so their ratio matches a weighted learner
    reduction without pretending that evidence or policy mass is a count.
    """

    policy_kl_sum: Float[Array, ""]
    policy_kl_count: Float[Array, ""]
    v_semantic_kl_sum: Float[Array, ""]
    v_semantic_kl_count: Float[Array, ""]
    v_dirichlet_kl_sum: Float[Array, ""]
    v_dirichlet_kl_count: Float[Array, ""]
    q_semantic_kl_sum: Float[Array, ""]
    q_semantic_kl_count: Float[Array, ""]
    q_dirichlet_kl_sum: Float[Array, ""]
    q_dirichlet_kl_count: Float[Array, ""]
    q_weighted_semantic_kl_sum: Float[Array, ""]
    q_weighted_semantic_kl_weight: Float[Array, ""]
    q_weighted_dirichlet_kl_sum: Float[Array, ""]
    q_weighted_dirichlet_kl_weight: Float[Array, ""]


def dirichlet_kl(
    target_alpha: Float[Array, "*target outcome"],
    prior_alpha: Float[Array, "*target outcome"],
) -> Float[Array, "*target"]:
    """Return ``KL(Dir(target_alpha) || Dir(prior_alpha))`` in nats."""

    target_alpha = jnp.asarray(target_alpha)
    prior_alpha = jnp.asarray(prior_alpha)
    target_mass = jnp.sum(target_alpha, axis=-1)
    prior_mass = jnp.sum(prior_alpha, axis=-1)
    log_normalizer_ratio = (
        gammaln(target_mass)
        - jnp.sum(gammaln(target_alpha), axis=-1)
        - gammaln(prior_mass)
        + jnp.sum(gammaln(prior_alpha), axis=-1)
    )
    shape_term = jnp.sum(
        (target_alpha - prior_alpha)
        * (digamma(target_alpha) - digamma(target_mass)[..., None]),
        axis=-1,
    )
    return log_normalizer_ratio + shape_term


def _categorical_kl(
    target: Float[Array, "*target outcome"],
    prior: Float[Array, "*target outcome"],
) -> Float[Array, "*target"]:
    positive = target > 0
    safe_target = jnp.where(positive, target, jnp.ones_like(target))
    terms = jnp.where(
        positive,
        target * (jnp.log(safe_target) - jnp.log(prior)),
        jnp.zeros_like(target),
    )
    return jnp.sum(terms, axis=-1)


def policy_displacement(
    target_policy: Float[Array, "batch action"],
    prior_logits: Float[Array, "batch action"],
    legal_action_mask: Bool[Array, "batch action"],
) -> tuple[Float[Array, "batch"], Bool[Array, "batch"]]:
    """Return ``KL(target_policy || masked_softmax(prior_logits))`` per root."""

    target_policy = jnp.where(
        legal_action_mask,
        target_policy,
        jnp.zeros_like(target_policy),
    )
    target_mass = jnp.sum(target_policy, axis=-1, keepdims=True)
    valid = jnp.any(legal_action_mask, axis=-1) & (target_mass[..., 0] > 0)
    target_policy = target_policy / jnp.where(
        target_mass > 0,
        target_mass,
        jnp.ones_like(target_mass),
    )
    masked_logits = jnp.where(
        legal_action_mask,
        prior_logits,
        jnp.asarray(jnp.finfo(prior_logits.dtype).min, prior_logits.dtype),
    )
    log_prior = jax.nn.log_softmax(masked_logits, axis=-1)
    positive = target_policy > 0
    safe_target = jnp.where(
        positive,
        target_policy,
        jnp.ones_like(target_policy),
    )
    kl = jnp.sum(
        jnp.where(
            positive,
            target_policy * (jnp.log(safe_target) - log_prior),
            jnp.zeros_like(target_policy),
        ),
        axis=-1,
    )
    return kl, valid


def native_displacement(
    prior_alpha: Float[Array, "*target outcome"],
    target_alpha: Float[Array, "*target outcome"],
    target_kind: Array,
    target_outcome: Array,
) -> NativeDisplacement:
    """Compare native targets with their root priors in common outcome space.

    Unresolved targets contribute both a categorical KL between Dirichlet
    means and a full Dirichlet KL.  Categorical certificates contribute the
    KL from a point mass to the prior mean, which is exactly prior surprisal.
    """

    prior_alpha = jnp.asarray(prior_alpha)
    target_alpha = jnp.asarray(target_alpha)
    target_kind = jnp.asarray(target_kind)
    target_outcome = jnp.asarray(target_outcome)

    num_outcomes = prior_alpha.shape[-1]
    prior_valid = jnp.all(
        jnp.isfinite(prior_alpha) & (prior_alpha > 0),
        axis=-1,
    )
    target_dirichlet_valid = jnp.all(
        jnp.isfinite(target_alpha) & (target_alpha > 0),
        axis=-1,
    )
    dirichlet_mask = (
        (target_kind == int(TARGET_DIRICHLET))
        & prior_valid
        & target_dirichlet_valid
    )
    categorical_kind = target_kind == int(TARGET_CATEGORICAL)
    valid_outcome = (target_outcome >= 0) & (target_outcome < num_outcomes)
    categorical_mask = categorical_kind & valid_outcome & prior_valid
    semantic_mask = dirichlet_mask | categorical_mask

    prior_mean = prior_alpha / jnp.sum(prior_alpha, axis=-1, keepdims=True)
    # Categorical and padded rows do not semantically contain a Dirichlet
    # target.  Their alpha storage is allowed to be zero/sentinel data, so
    # substitute the valid prior before evaluating either normalization or the
    # closed-form full KL.  The categorical mean is replaced by its point mass
    # immediately below.
    safe_target_alpha = jnp.where(
        dirichlet_mask[..., None],
        target_alpha,
        prior_alpha,
    )
    target_mean = safe_target_alpha / jnp.sum(
        safe_target_alpha,
        axis=-1,
        keepdims=True,
    )
    categorical_mean = jax.nn.one_hot(
        target_outcome,
        num_outcomes,
        dtype=prior_mean.dtype,
    )
    target_mean = jnp.where(
        categorical_mask[..., None],
        categorical_mean,
        target_mean,
    )
    semantic_kl = _categorical_kl(target_mean, prior_mean)
    full_kl = dirichlet_kl(safe_target_alpha, prior_alpha)

    safe_outcome = jnp.clip(target_outcome, 0, num_outcomes - 1)
    categorical_prior_probability = jnp.take_along_axis(
        prior_mean,
        safe_outcome[..., None],
        axis=-1,
    )[..., 0]
    categorical_surprisal = -jnp.log(categorical_prior_probability)
    return NativeDisplacement(
        semantic_kl=semantic_kl,
        semantic_mask=semantic_mask,
        dirichlet_kl=full_kl,
        dirichlet_mask=dirichlet_mask,
        categorical_surprisal=categorical_surprisal,
        categorical_mask=categorical_mask,
    )


def distillation_discrepancy(
    *,
    prior_logits: Float[Array, "*batch action"],
    target_policy: Float[Array, "*batch action"],
    legal_action_mask: Bool[Array, "*batch action"],
    policy_row_mask: Bool[Array, "*batch"],
    prior_alpha_v: Float[Array, "*batch outcome"] | None = None,
    prior_alpha_q: Float[Array, "*batch action outcome"] | None = None,
    target_alpha_v: Float[Array, "*batch outcome"] | None = None,
    target_alpha_q: Float[Array, "*batch action outcome"] | None = None,
    v_target_kind: Array | None = None,
    v_target_outcome: Array | None = None,
    q_target_kind: Array | None = None,
    q_target_outcome: Array | None = None,
    v_mask: Bool[Array, "*batch"] | None = None,
    q_mask: Bool[Array, "*batch action"] | None = None,
    q_sample_weight: Float[Array, "*batch action"] | None = None,
) -> DistillationDiscrepancy:
    """Measure a network-to-fixed-target gap without mutating either input.

    The full-Dirichlet populations exclude categorical targets and non-finite
    closed-form values.  Raw sums and counts (or total sample weight) are
    returned so callers can pool probes before forming a mean or capture
    fraction.
    """

    policy_kl, valid_policy = policy_displacement(
        target_policy,
        prior_logits,
        legal_action_mask,
    )
    policy_mask = (
        valid_policy
        & policy_row_mask
        & jnp.isfinite(policy_kl)
    )
    dtype = prior_logits.dtype

    def masked_sum(value: jax.Array, mask: jax.Array) -> jax.Array:
        return jnp.sum(
            jnp.where(mask, value, jnp.zeros_like(value))
        )

    def mask_count(mask: jax.Array) -> jax.Array:
        return jnp.sum(mask.astype(dtype))

    def weighted_sum_and_weight(
        value: jax.Array,
        mask: jax.Array,
        sample_weight: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Pool a population under optional finite, positive sample weights."""

        weight = (
            jnp.ones_like(value)
            if sample_weight is None
            else jnp.broadcast_to(
                jnp.asarray(sample_weight, dtype=value.dtype),
                value.shape,
            )
        )
        valid = (
            mask
            & jnp.isfinite(value)
            & jnp.isfinite(weight)
            & (weight > 0)
        )
        safe_value = jnp.where(valid, value, jnp.zeros_like(value))
        safe_weight = jnp.where(valid, weight, jnp.zeros_like(weight))
        return jnp.sum(safe_weight * safe_value), jnp.sum(safe_weight)

    zero = jnp.zeros((), dtype=dtype)
    has_native = all(
        value is not None
        for value in (
            prior_alpha_v,
            prior_alpha_q,
            target_alpha_v,
            target_alpha_q,
            v_target_kind,
            v_target_outcome,
            q_target_kind,
            q_target_outcome,
            v_mask,
            q_mask,
        )
    )
    if not has_native:
        return DistillationDiscrepancy(
            policy_kl_sum=masked_sum(policy_kl, policy_mask),
            policy_kl_count=mask_count(policy_mask),
            v_semantic_kl_sum=zero,
            v_semantic_kl_count=zero,
            v_dirichlet_kl_sum=zero,
            v_dirichlet_kl_count=zero,
            q_semantic_kl_sum=zero,
            q_semantic_kl_count=zero,
            q_dirichlet_kl_sum=zero,
            q_dirichlet_kl_count=zero,
            q_weighted_semantic_kl_sum=zero,
            q_weighted_semantic_kl_weight=zero,
            q_weighted_dirichlet_kl_sum=zero,
            q_weighted_dirichlet_kl_weight=zero,
        )

    assert prior_alpha_v is not None
    assert prior_alpha_q is not None
    assert target_alpha_v is not None
    assert target_alpha_q is not None
    assert v_target_kind is not None
    assert v_target_outcome is not None
    assert q_target_kind is not None
    assert q_target_outcome is not None
    assert v_mask is not None
    assert q_mask is not None
    v = native_displacement(
        prior_alpha_v,
        target_alpha_v,
        v_target_kind,
        v_target_outcome,
    )
    q = native_displacement(
        prior_alpha_q,
        target_alpha_q,
        q_target_kind,
        q_target_outcome,
    )
    v_semantic_mask = (
        v.semantic_mask & v_mask & jnp.isfinite(v.semantic_kl)
    )
    v_dirichlet_mask = (
        v.dirichlet_mask & v_mask & jnp.isfinite(v.dirichlet_kl)
    )
    q_semantic_mask = (
        q.semantic_mask & q_mask & jnp.isfinite(q.semantic_kl)
    )
    q_dirichlet_mask = (
        q.dirichlet_mask & q_mask & jnp.isfinite(q.dirichlet_kl)
    )
    q_weighted_semantic_sum, q_weighted_semantic_weight = (
        weighted_sum_and_weight(
            q.semantic_kl,
            q_semantic_mask,
            q_sample_weight,
        )
    )
    q_weighted_dirichlet_sum, q_weighted_dirichlet_weight = (
        weighted_sum_and_weight(
            q.dirichlet_kl,
            q_dirichlet_mask,
            q_sample_weight,
        )
    )
    return DistillationDiscrepancy(
        policy_kl_sum=masked_sum(policy_kl, policy_mask),
        policy_kl_count=mask_count(policy_mask),
        v_semantic_kl_sum=masked_sum(v.semantic_kl, v_semantic_mask),
        v_semantic_kl_count=mask_count(v_semantic_mask),
        v_dirichlet_kl_sum=masked_sum(v.dirichlet_kl, v_dirichlet_mask),
        v_dirichlet_kl_count=mask_count(v_dirichlet_mask),
        q_semantic_kl_sum=masked_sum(q.semantic_kl, q_semantic_mask),
        q_semantic_kl_count=mask_count(q_semantic_mask),
        q_dirichlet_kl_sum=masked_sum(q.dirichlet_kl, q_dirichlet_mask),
        q_dirichlet_kl_count=mask_count(q_dirichlet_mask),
        q_weighted_semantic_kl_sum=q_weighted_semantic_sum,
        q_weighted_semantic_kl_weight=q_weighted_semantic_weight,
        q_weighted_dirichlet_kl_sum=q_weighted_dirichlet_sum,
        q_weighted_dirichlet_kl_weight=q_weighted_dirichlet_weight,
    )


def _tree_max_depth(tree: Tree) -> Float[Array, "batch"]:
    """Recover each lane's maximum expanded depth from insertion-order parents."""

    parents = tree.parents
    batch = jnp.arange(parents.shape[0])
    depths = jnp.zeros_like(parents)

    def set_depth(node: int, current_depths: jax.Array) -> jax.Array:
        parent = parents[:, node]
        expanded = parent != int(Tree.NO_PARENT)
        safe_parent = jnp.where(expanded, parent, int(Tree.ROOT_INDEX))
        depth = current_depths[batch, safe_parent] + 1
        return current_depths.at[:, node].set(
            jnp.where(expanded, depth, jnp.zeros_like(depth))
        )

    depths = jax.lax.fori_loop(1, parents.shape[1], set_depth, depths)
    return jnp.max(depths, axis=-1).astype(tree.edge_alpha.dtype)


def root_search_diagnostics(
    *,
    prior_logits: Float[Array, "batch action"],
    prior_alpha_v: Float[Array, "batch outcome"],
    prior_alpha_q: Float[Array, "batch action outcome"],
    target_policy: Float[Array, "batch action"],
    target_alpha_v: Float[Array, "batch outcome"],
    target_alpha_q: Float[Array, "batch action outcome"],
    q_target_kind: Array,
    q_target_outcome: Array,
    v_target_kind: Array,
    v_target_outcome: Array,
    legal_action_mask: Bool[Array, "batch action"],
    tree: Tree,
    summary: SearchSummary,
) -> SearchDiagnostics:
    """Build additive root diagnostics from one completed DT search."""

    policy_kl, policy_mask = policy_displacement(
        target_policy,
        prior_logits,
        legal_action_mask,
    )
    v = native_displacement(
        prior_alpha_v,
        target_alpha_v,
        v_target_kind,
        v_target_outcome,
    )
    q = native_displacement(
        prior_alpha_q,
        target_alpha_q,
        q_target_kind,
        q_target_outcome,
    )

    dtype = target_alpha_v.dtype

    def masked_value(value: jax.Array, mask: jax.Array) -> jax.Array:
        return jnp.where(mask, value, jnp.zeros_like(value))

    def count(mask: jax.Array) -> jax.Array:
        return mask.astype(dtype)

    policy_metric_mask = policy_mask & jnp.isfinite(policy_kl)
    v_semantic_mask = v.semantic_mask & jnp.isfinite(v.semantic_kl)
    v_dirichlet_mask = v.dirichlet_mask & jnp.isfinite(v.dirichlet_kl)
    v_categorical_mask = (
        v.categorical_mask & jnp.isfinite(v.categorical_surprisal)
    )
    q_semantic_mask = q.semantic_mask & jnp.isfinite(q.semantic_kl)
    q_dirichlet_mask = q.dirichlet_mask & jnp.isfinite(q.dirichlet_kl)
    q_categorical_mask = (
        q.categorical_mask & jnp.isfinite(q.categorical_surprisal)
    )

    q_semantic_sum = jnp.sum(
        masked_value(q.semantic_kl, q_semantic_mask),
        axis=-1,
    )
    normalized_target_policy = jnp.where(
        legal_action_mask,
        target_policy,
        jnp.zeros_like(target_policy),
    )
    target_policy_mass = jnp.sum(
        normalized_target_policy,
        axis=-1,
        keepdims=True,
    )
    normalized_target_policy = normalized_target_policy / jnp.where(
        target_policy_mass > 0,
        target_policy_mass,
        jnp.ones_like(target_policy_mass),
    )
    q_policy_weight = jnp.where(
        q_semantic_mask,
        normalized_target_policy,
        jnp.zeros_like(normalized_target_policy),
    )
    q_policy_semantic_sum = jnp.sum(
        q_policy_weight * masked_value(q.semantic_kl, q_semantic_mask),
        axis=-1,
    )
    q_policy_semantic_count = jnp.sum(q_policy_weight, axis=-1)
    q_dirichlet_sum = jnp.sum(
        masked_value(q.dirichlet_kl, q_dirichlet_mask),
        axis=-1,
    )
    q_categorical_sum = jnp.sum(
        masked_value(q.categorical_surprisal, q_categorical_mask),
        axis=-1,
    )

    root = int(Tree.ROOT_INDEX)
    visited = legal_action_mask & (
        tree.children_index[:, root] != int(Tree.UNVISITED)
    )
    categorical = legal_action_mask & q.categorical_mask
    repaired = legal_action_mask & (
        categorical | (summary.visit_counts > 0)
    )
    expanded = tree.parents[:, 1:] != int(Tree.NO_PARENT)
    root_count = jnp.ones((prior_logits.shape[0],), dtype=dtype)
    valid_policy = policy_mask
    policy_support = jnp.sum(
        (
            legal_action_mask
            & (normalized_target_policy > 0)
        ).astype(dtype),
        axis=-1,
    )
    policy_ess_denominator = jnp.sum(
        jnp.square(normalized_target_policy),
        axis=-1,
    )
    policy_ess = jnp.where(
        valid_policy,
        1.0 / jnp.maximum(
            policy_ess_denominator,
            jnp.asarray(jnp.finfo(dtype).tiny, dtype),
        ),
        jnp.zeros_like(policy_ess_denominator),
    )
    masked_prior_logits = jnp.where(
        legal_action_mask,
        prior_logits,
        jnp.asarray(jnp.finfo(prior_logits.dtype).min, prior_logits.dtype),
    )
    policy_top1_agreement = valid_policy & (
        jnp.argmax(masked_prior_logits, axis=-1)
        == jnp.argmax(normalized_target_policy, axis=-1)
    )

    return SearchDiagnostics(
        search_policy_kl_sum=masked_value(policy_kl, policy_metric_mask),
        search_policy_kl_count=count(policy_metric_mask),
        search_v_semantic_kl_sum=masked_value(
            v.semantic_kl,
            v_semantic_mask,
        ),
        search_v_semantic_kl_count=count(v_semantic_mask),
        search_v_dirichlet_kl_sum=masked_value(
            v.dirichlet_kl,
            v_dirichlet_mask,
        ),
        search_v_dirichlet_kl_count=count(v_dirichlet_mask),
        search_v_categorical_surprisal_sum=masked_value(
            v.categorical_surprisal,
            v_categorical_mask,
        ),
        search_v_categorical_surprisal_count=count(v_categorical_mask),
        search_q_semantic_kl_sum=q_semantic_sum,
        search_q_semantic_kl_count=jnp.sum(
            count(q_semantic_mask),
            axis=-1,
        ),
        search_q_policy_semantic_kl_sum=q_policy_semantic_sum,
        search_q_policy_semantic_kl_count=q_policy_semantic_count,
        search_q_dirichlet_kl_sum=q_dirichlet_sum,
        search_q_dirichlet_kl_count=jnp.sum(
            count(q_dirichlet_mask),
            axis=-1,
        ),
        search_q_categorical_surprisal_sum=q_categorical_sum,
        search_q_categorical_surprisal_count=jnp.sum(
            count(q_categorical_mask),
            axis=-1,
        ),
        search_root_count=root_count,
        search_legal_action_count=jnp.sum(
            legal_action_mask.astype(dtype),
            axis=-1,
        ),
        search_visited_action_count=jnp.sum(visited.astype(dtype), axis=-1),
        search_repaired_action_count=jnp.sum(repaired.astype(dtype), axis=-1),
        search_categorical_action_count=jnp.sum(
            categorical.astype(dtype),
            axis=-1,
        ),
        search_solved_root_count=count(v.categorical_mask),
        search_expanded_node_count=jnp.sum(
            expanded.astype(dtype),
            axis=-1,
        ),
        search_simulation_active_count=tree.simulation_active_count.astype(
            dtype
        ),
        search_executed_simulation_row_count=(
            tree.executed_simulation_call_count.astype(dtype)
        ),
        search_requested_simulation_count=jnp.full_like(
            root_count,
            float(tree.num_simulations),
        ),
        search_structural_support_sum=jnp.sum(
            jnp.where(
                legal_action_mask,
                summary.visit_counts,
                jnp.zeros_like(summary.visit_counts),
            ),
            axis=-1,
        ),
        search_max_depth_sum=_tree_max_depth(tree),
        search_policy_support_sum=masked_value(
            policy_support,
            policy_metric_mask,
        ),
        search_policy_ess_sum=masked_value(
            policy_ess,
            policy_metric_mask,
        ),
        search_policy_top1_agreement_count=count(
            policy_top1_agreement & policy_metric_mask
        ),
    )


__all__ = [
    "DistillationDiscrepancy",
    "NativeDisplacement",
    "SearchDiagnostics",
    "dirichlet_kl",
    "distillation_discrepancy",
    "native_displacement",
    "policy_displacement",
    "root_search_diagnostics",
]
