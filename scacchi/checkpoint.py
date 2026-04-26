"""Orbax checkpoints for training state, RNG, and metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax
import orbax.checkpoint as ocp
from omegaconf import DictConfig, OmegaConf

from scacchi.types import TrainState

ITEM_NAMES = ("model_state", "optimizer_state", "rngs", "meta")


class NoOpCheckpointManager(ocp.CheckpointManager):
    """Checkpoint manager that never saves, used when max_to_keep is zero."""

    def should_save(self, step: int) -> bool:
        del step
        return False


class RestoreResult(NamedTuple):
    start_step: int
    train_state: TrainState
    rng_key: jax.Array
    meta: dict[str, Any]


def build_checkpoint_manager(
    cfg: DictConfig,
    ckpt_dir: str | Path,
    *,
    max_steps: int,
    item_names: tuple[str, ...] = ITEM_NAMES,
) -> ocp.CheckpointManager:
    """Build an Orbax checkpoint manager."""

    max_to_keep = int(cfg.max_to_keep)
    save_on_steps = [max_steps - 1] if max_steps > 0 else []
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=int(cfg.save_interval_steps),
        save_on_steps=save_on_steps,
        single_host_load_and_broadcast=jax.process_count() > 1,
        enable_async_checkpointing=True,
        multiprocessing_options=ocp.options.MultiprocessingOptions(primary_host=0),
    )
    manager_cls = NoOpCheckpointManager if max_to_keep == 0 else ocp.CheckpointManager
    return manager_cls(Path(ckpt_dir).resolve(), options=options, item_names=item_names)


def _json_meta(
    *,
    cfg: Any,
    step: int,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg_json = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    return {
        "cfg": cfg_json,
        "step": int(step),
        "metadata": metadata or {},
    }


def maybe_save_checkpoint(
    manager: ocp.CheckpointManager,
    step: int,
    *,
    cfg: Any,
    train_state: TrainState,
    rng_key: jax.Array,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Save a structured Orbax checkpoint if configured for this step."""

    if not manager.should_save(step):
        return False

    save_args = ocp.args.Composite(
        model_state=ocp.args.StandardSave(train_state.params),
        optimizer_state=ocp.args.StandardSave(train_state.opt_state),
        rngs=ocp.args.StandardSave({"key": rng_key}),
        meta=ocp.args.JsonSave(_json_meta(cfg=cfg, step=step, metadata=metadata)),
    )
    return manager.save(step, args=save_args)


def restore_checkpoint(
    manager: ocp.CheckpointManager,
    *,
    train_state: TrainState,
    rng_key: jax.Array,
) -> RestoreResult:
    """Restore latest checkpoint, or return provided state if none exists."""

    step = manager.latest_step()
    if step is None:
        return RestoreResult(start_step=0, train_state=train_state, rng_key=rng_key, meta={})

    restore_args = ocp.args.Composite(
        model_state=ocp.args.StandardRestore(train_state.params),
        optimizer_state=ocp.args.StandardRestore(train_state.opt_state),
        rngs=ocp.args.StandardRestore({"key": rng_key}),
        meta=ocp.args.JsonRestore(),
    )
    restored = manager.restore(step, args=restore_args)
    restored_state = TrainState(
        params=restored["model_state"],
        opt_state=restored["optimizer_state"],
    )
    return RestoreResult(
        start_step=int(step) + 1,
        train_state=restored_state,
        rng_key=restored["rngs"]["key"],
        meta=restored["meta"],
    )
