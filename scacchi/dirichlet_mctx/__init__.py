"""Public facade for Dirichlet Thompson tree search."""

from .base import PolicyOutput, RecurrentFnOutput, RootFnOutput
from .policies import dirichlet_thompson_policy
from .posterior_updates import update_posterior
from .tree import PosteriorUpdate, PosteriorUpdateContext

__all__ = [
    "PolicyOutput",
    "PosteriorUpdate",
    "PosteriorUpdateContext",
    "RecurrentFnOutput",
    "RootFnOutput",
    "dirichlet_thompson_policy",
    "update_posterior",
]
