"""A small MCTX-shaped backend for Dirichlet Thompson tree search."""

from .action_selection import (
    categorical_action,
    effective_action_alpha,
    masked_argmax,
    posterior_best_policy,
    sample_dirichlet,
    thompson_action_selection,
    thompson_sample,
)
from .base import PolicyOutput, RecurrentFnOutput, RootFnOutput
from .native_targets import (
    NativeTargetFields,
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    TARGET_PAD,
    categorical_point,
    dirichlet_nll_at_categorical,
    native_fields_from_beta,
)
from .outcomes import NO_DISTANCE, NO_OUTCOME, align_categorical_outcome, align_outcome, categorical_utility, flip_outcome, outcome_mean, outcome_utility
from .policies import dirichlet_thompson_policy
from .posterior_updates import (
    DEFAULT_KAPPA,
    DEFAULT_POLICY_SAMPLE_CHUNK_SIZE,
    DEFAULT_POLICY_SAMPLES,
    mix_value_prior,
    update_posterior,
)
from .search import search
from .tree import instantiate_tree_from_root
from .tree import (
    ChildrenView,
    LeafView,
    NodeView,
    PosteriorUpdate,
    PosteriorUpdateContext,
    SearchSummary,
    Tree,
    UnbatchedTree,
)

__all__ = [
    "PolicyOutput",
    "ChildrenView",
    "DEFAULT_KAPPA",
    "DEFAULT_POLICY_SAMPLE_CHUNK_SIZE",
    "DEFAULT_POLICY_SAMPLES",
    "LeafView",
    "NodeView",
    "NO_DISTANCE",
    "NO_OUTCOME",
    "NativeTargetFields",
    "PosteriorUpdateContext",
    "PosteriorUpdate",
    "RecurrentFnOutput",
    "RootFnOutput",
    "SearchSummary",
    "TARGET_CATEGORICAL",
    "TARGET_DIRICHLET",
    "TARGET_PAD",
    "Tree",
    "UnbatchedTree",
    "align_categorical_outcome",
    "align_outcome",
    "categorical_action",
    "categorical_point",
    "categorical_utility",
    "dirichlet_thompson_policy",
    "dirichlet_nll_at_categorical",
    "effective_action_alpha",
    "flip_outcome",
    "instantiate_tree_from_root",
    "masked_argmax",
    "mix_value_prior",
    "native_fields_from_beta",
    "outcome_mean",
    "outcome_utility",
    "posterior_best_policy",
    "sample_dirichlet",
    "search",
    "thompson_action_selection",
    "thompson_sample",
    "update_posterior",
]
