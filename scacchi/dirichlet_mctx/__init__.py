"""A small MCTX-shaped backend for Dirichlet Thompson tree search."""

from .action_selection import (
    align_outcome,
    flip_outcome,
    masked_argmax,
    outcome_mean,
    outcome_utility,
    policy_prior_interior_action_selection,
    thompson_root_action_selection,
)
from .base import PolicyOutput, RecurrentFnOutput, RootFnOutput
from .policies import dirichlet_thompson_policy, posterior_best_policy_target
from .posterior_updates import update_posterior
from .search import instantiate_tree_from_root, search
from .tree import Posterior, SearchSummary, Tree

__all__ = [
    "PolicyOutput",
    "Posterior",
    "RecurrentFnOutput",
    "RootFnOutput",
    "SearchSummary",
    "Tree",
    "align_outcome",
    "dirichlet_thompson_policy",
    "flip_outcome",
    "instantiate_tree_from_root",
    "masked_argmax",
    "outcome_mean",
    "outcome_utility",
    "policy_prior_interior_action_selection",
    "posterior_best_policy_target",
    "search",
    "thompson_root_action_selection",
    "update_posterior",
]
