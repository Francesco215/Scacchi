"""Orbax checkpoint manager for Scacchi — save/restore/from_pretrained."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import orbax.checkpoint as ocp
import pgx
from flax import nnx
from pgx._src.baseline import BaselineModelId

from .network import build_model

if TYPE_CHECKING:
    from .train import Config


class NoOpCheckpointManager(ocp.CheckpointManager):
    """Drops all saves — returned when max_to_keep == 0."""

    def should_save(self, step: int) -> bool:
        return False


def build_checkpoint_manager(
    config: Config,
    ckpt_dir: Path,
) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.ckpt_max_to_keep,
        save_interval_steps=config.ckpt_save_interval_steps,
        save_on_steps=[config.max_num_iters - 1],
        enable_async_checkpointing=True,
    )
    item_names = ("model", "optimizer", "rngs", "meta")
    if config.ckpt_max_to_keep == 0:
        return NoOpCheckpointManager(ckpt_dir, options=options, item_names=item_names)
    return ocp.CheckpointManager(ckpt_dir, options=options, item_names=item_names)


def maybe_save(
    manager: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    rng_key: jax.Array,
    config: Config,
    hours: float,
    frames: int,
) -> None:
    if not manager.should_save(step):
        return
    meta: dict[str, Any] = {
        "config": config.model_dump(),
        "step": step,
        "hours": hours,
        "frames": frames,
    }
    save_args = ocp.args.Composite(
        model=ocp.args.StandardSave(nnx.state(model)),
        optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
        rngs=ocp.args.StandardSave({"key": rng_key}),
        meta=ocp.args.JsonSave(meta),
    )
    manager.save(step, args=save_args)


def restore(
    manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    rng_key: jax.Array,
) -> tuple[int, jax.Array, float, int]:
    """Restore in-place; returns (start_iter, rng_key, hours, frames)."""
    step = manager.latest_step()
    if step is None:
        print("No checkpoint found, starting from scratch.")
        return 0, rng_key, 0.0, 0

    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(nnx.state(model)),
            optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
            rngs=ocp.args.StandardRestore({"key": rng_key}),
            meta=ocp.args.JsonRestore(),
        ),
    )
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    rng_key = restored["rngs"]["key"]
    meta = restored["meta"]
    print(f"Restored checkpoint from step {step}.")
    return step + 1, rng_key, float(meta["hours"]), int(meta["frames"])


def from_pretrained(
    checkpoint_path: str,
    env: pgx.Env,
    rngs: nnx.Rngs | None = None,
) -> nnx.Module:
    """Load a model from a checkpoint directory (no optimizer required)."""
    if rngs is None:
        rngs = nnx.Rngs(0)
    checkpoint_path = str(Path(checkpoint_path).resolve())

    with ocp.CheckpointManager(checkpoint_path) as manager:
        step = manager.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

        meta_restored = manager.restore(
            step, args=ocp.args.Composite(meta=ocp.args.JsonRestore())
        )
        meta = meta_restored["meta"]

    from .train import Config  # local import to avoid circular at module level

    config = Config(**meta["config"])
    h, w, _ = env.observation_shape
    observation_shape = (h, w, config.num_history_steps * 14 + 3)
    model = build_model(
        config,
        num_actions=env.num_actions,
        observation_shape=observation_shape,
        rngs=rngs,
    )

    with ocp.CheckpointManager(checkpoint_path) as manager:
        step = manager.latest_step()
        restored = manager.restore(
            step,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
        )
    nnx.update(model, restored["model"])
    return model
