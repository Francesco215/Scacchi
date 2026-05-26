"""Orbax checkpoint manager for Scacchi — save/restore/from_pretrained."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import orbax.checkpoint as ocp
import pgx
from flax import nnx

from .network import AZNet, BoardlawDirichletNet, BoardlawNet

if TYPE_CHECKING:
    from .config import CheckpointingConfig, Config, RunConfig


def _suppress_orbax_logs() -> None:
    """Keep Orbax checkpoint internals from spamming INFO logs."""
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("orbax").setLevel(logging.WARNING)

    try:
        from absl import logging as absl_logging
    except ImportError:
        return

    absl_logging.set_verbosity(absl_logging.WARNING)


class NoOpCheckpointManager(ocp.CheckpointManager):
    """Drops all saves — returned when max_to_keep == 0."""

    def should_save(self, step: int) -> bool:
        return False


def build_checkpoint_manager(
    checkpointing: CheckpointingConfig,
    run: RunConfig,
    ckpt_dir: Path,
) -> ocp.CheckpointManager:
    _suppress_orbax_logs()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=checkpointing.max_to_keep,
        save_interval_steps=checkpointing.save_interval_steps,
        save_on_steps=[run.max_num_iters - 1],
        enable_async_checkpointing=True,
    )
    item_names = ("model", "optimizer", "rngs", "meta")
    if checkpointing.max_to_keep == 0:
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
        "config": config.to_dict(),
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
    _suppress_orbax_logs()
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
        model = _build_checkpoint_model(meta_restored["meta"].get("config", {}), env, rngs)
        restored = manager.restore(
            step,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
        )

    nnx.update(model, restored["model"])
    return model


def _build_checkpoint_model(config: Any, env: pgx.Env, rngs: nnx.Rngs) -> nnx.Module:
    root = config if isinstance(config, dict) else {}
    model = root.get("model", root)
    env_config = root.get("env", root)
    regularization = root.get("training", {}).get("regularization", root)
    network = str(model.get("network", "boardlaw_dirichlet"))
    width = int(model.get("num_channels", 128))
    depth = int(model.get("num_layers", 6))
    if network == "aznet":
        return AZNet(
            num_actions=env.num_actions,
            observation_shape=env.observation_shape,
            num_channels=width,
            num_blocks=depth,
            resnet_v2=bool(model.get("resnet_v2", True)),
            rngs=rngs,
        )
    if network == "boardlaw":
        return BoardlawNet(
            num_actions=env.num_actions,
            observation_shape=env.observation_shape,
            width=width,
            depth=depth,
            rngs=rngs,
        )
    if network == "boardlaw_dirichlet":
        return BoardlawDirichletNet(
            num_actions=env.num_actions,
            observation_shape=env.observation_shape,
            num_outcomes=int(env_config.get("num_outcomes") or 3),
            width=width,
            depth=depth,
            dirichlet_concentration_clip=regularization.get(
                "dirichlet_concentration_clip",
                8.0,
            ),
            rngs=rngs,
        )
    raise ValueError(f"unknown checkpoint network: {network!r}")
