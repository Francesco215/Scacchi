"""A small MCTX-shaped backend for Dirichlet Thompson tree search."""

from .action_selection import (
    align_outcome,
    effective_action_alpha,
    flip_outcome,
    masked_argmax,
    outcome_mean,
    outcome_utility,
    root_action_alpha,
    sample_dirichlet,
    thompson_action_selection,
    thompson_policy,
    thompson_sample,
)
from .base import PolicyOutput, RecurrentFnOutput, RootFnOutput
from .policies import dirichlet_thompson_policy, posterior_best_policy_target
from .posterior_updates import (
    DEFAULT_POLICY_SAMPLE_CHUNK_SIZE,
    DEFAULT_POLICY_SAMPLES,
    mix_value_prior,
    update_posterior,
)
from .search import instantiate_tree_from_root, search
from .tree import (
    ChildrenView,
    LeafView,
    NodePosterior,
    NodeView,
    PosteriorUpdateContext,
    SearchSummary,
    Tree,
)

__all__ = [
    "PolicyOutput",
    "ChildrenView",
    "DEFAULT_POLICY_SAMPLE_CHUNK_SIZE",
    "DEFAULT_POLICY_SAMPLES",
    "LeafView",
    "NodePosterior",
    "NodeView",
    "PosteriorUpdateContext",
    "RecurrentFnOutput",
    "RootFnOutput",
    "SearchSummary",
    "Tree",
    "align_outcome",
    "dirichlet_thompson_policy",
    "effective_action_alpha",
    "flip_outcome",
    "instantiate_tree_from_root",
    "masked_argmax",
    "mix_value_prior",
    "outcome_mean",
    "outcome_utility",
    "posterior_best_policy_target",
    "root_action_alpha",
    "sample_dirichlet",
    "search",
    "thompson_action_selection",
    "thompson_policy",
    "thompson_sample",
    "update_posterior",
]
