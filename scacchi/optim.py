"""Optimizer factories."""

from __future__ import annotations

import optax
from omegaconf import DictConfig

from scacchi.config import OptimizerConfig


def make_optimizer(cfg: DictConfig | OptimizerConfig) -> optax.GradientTransformation:
    """Build an Optax optimizer from config."""

    name = str(cfg.name).lower()
    learning_rate = float(cfg.learning_rate)
    weight_decay = float(getattr(cfg, "weight_decay", 0.0))
    if name == "adam":
        return optax.adam(learning_rate=learning_rate)
    if name == "adamw":
        return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "muon":
        return optax.contrib.muon(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_weight_decay=float(getattr(cfg, "adam_weight_decay", 0.0)),
        )
    msg = f"Unknown optimizer '{cfg.name}'."
    raise ValueError(msg)
