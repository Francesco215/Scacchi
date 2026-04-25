"""Small pickle checkpoints for local experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax

from scacchi.types import TrainState


def save_checkpoint(
    path: str | Path,
    *,
    cfg: Any,
    train_state: TrainState,
    rng_key: Any,
    step: int,
) -> None:
    """Save model parameters, optimizer state, RNG, and config metadata."""

    payload = {
        "cfg": cfg,
        "train_state": jax.device_get(train_state),
        "rng_key": jax.device_get(rng_key),
        "step": int(step),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
