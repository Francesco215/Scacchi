"""Orbax checkpoint manager for Scacchi — save/restore/from_pretrained."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
import pgx
from flax import nnx

from .network import build_model
from .types import Config, config_to_dict, load_config

if TYPE_CHECKING:
    from .types import Config


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

    def __init__(self, directory: Path, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._directory: Path = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def __enter__(self) -> NoOpCheckpointManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback

    def latest_step(self) -> int | None:
        return None

    def should_save(self, step: int) -> bool:
        del step
        return False

    def save(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return False


def _checkpoint_manager_options(
    *,
    max_to_keep: int | None = None,
    save_interval_steps: int = 1,
    save_on_steps: tuple[int, ...] | None = None,
    read_only: bool = False,
    primary_only: bool = False,
) -> ocp.CheckpointManagerOptions:
    is_multihost = jax.process_count() > 1
    if primary_only and is_multihost:
        # Pod workers have private local disks, so cross-host barriers can
        # never complete: process 0 saves alone, with no collective calls.
        # The caller must pre-create the directory and give every other
        # process a NoOpCheckpointManager.
        return ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
            save_on_steps=save_on_steps,
            single_host_load_and_broadcast=False,
            enable_async_checkpointing=True,
            multiprocessing_options=ocp.options.MultiprocessingOptions(
                primary_host=0,
                active_processes={0},
            ),
            create=False,
            read_only=read_only,
        )
    return ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=save_interval_steps,
        save_on_steps=save_on_steps,
        single_host_load_and_broadcast=is_multihost,
        enable_async_checkpointing=True,
        multiprocessing_options=ocp.options.MultiprocessingOptions(primary_host=0),
        read_only=read_only,
    )


def build_checkpoint_manager(
    config: Config,
    ckpt_dir: Path,
) -> ocp.CheckpointManager:
    _suppress_orbax_logs()
    options = _checkpoint_manager_options(
        max_to_keep=config.checkpointing.max_to_keep,
        save_interval_steps=config.checkpointing.save_interval_steps,
        save_on_steps=(config.run.max_num_iters - 1,),
        primary_only=True,
    )
    item_names = ("model", "optimizer", "rngs", "meta")
    if config.checkpointing.max_to_keep == 0:
        return NoOpCheckpointManager(ckpt_dir)
    if jax.process_count() > 1 and jax.process_index() != 0:
        # Pod workers have private local disks: only process 0 saves (its
        # manager uses active_processes={0}); the rest must not construct a
        # real manager or orbax raises on the subset barrier. Note multihost
        # RESUME from these checkpoints requires shared storage.
        return NoOpCheckpointManager(ckpt_dir)
    return ocp.CheckpointManager(ckpt_dir, options=options, item_names=item_names)


def _rng_key_to_checkpoint_value(rng_key: jax.Array) -> np.ndarray:
    key_data = rng_key
    if jax.dtypes.issubdtype(rng_key.dtype, jax.dtypes.prng_key):
        key_data = jax.random.key_data(rng_key)
    return np.asarray(jax.device_get(key_data))


def _rng_key_from_checkpoint_value(value: Any, template: jax.Array) -> jax.Array:
    if jax.dtypes.issubdtype(template.dtype, jax.dtypes.prng_key):
        key_data = jnp.asarray(value, dtype=jnp.uint32)
        return jax.random.wrap_key_data(
            key_data,
            impl=jax.random.key_impl(template),
        )
    return jnp.asarray(value, dtype=template.dtype)


def _load_checkpoint_config(raw: dict[str, Any]) -> Config:
    """Load current or pre-nested-schema checkpoint metadata.

    The solved Hex baselines were produced before runtime configuration was
    split into nested sections.  Keep that compatibility local to checkpoint
    loading so flat user configs remain an error.
    """
    if "model" in raw or "env" in raw or "run" in raw:
        # Checkpoints written by the reset-per-block API stored a per-block
        # simulation count and a block count.  The replacement search owns one
        # persistent tree, so migrate that historical product at this I/O
        # boundary instead of retaining ``num_blocks`` in the public config.
        migrated = copy.deepcopy(raw)

        def migrate_search(search: Any) -> None:
            if not isinstance(search, dict):
                return
            settings = search.get("dirichlet_thompson")
            if isinstance(settings, dict):
                blocks = settings.pop("num_blocks", None)
                if blocks is not None:
                    settings["num_simulations"] = int(
                        settings.get("num_simulations", 32)
                    ) * int(blocks)
                # Native categorical search has one prior-mixing kappa.  Old
                # leaf/terminal constants never represented this quantity;
                # the implementation used the app.js default of four.
                settings.pop("constants", None)
                settings.setdefault("kappa", 4.0)
            gumbel = search.get("gumbel")
            if isinstance(gumbel, dict):
                for stale in (
                    "constants",
                    "policy_sample_chunk_size",
                    "completed_q_value_scale",
                    "completed_q_rescale_values",
                ):
                    gumbel.pop(stale, None)

        migrate_search(migrated.get("search"))
        selfplay = migrated.get("selfplay")
        if isinstance(selfplay, dict):
            migrate_search(selfplay.get("search"))
        eval_config = migrated.get("eval")
        if isinstance(eval_config, dict):
            migrate_search(eval_config.get("player_search"))
            migrate_search(eval_config.get("baseline_search"))
        return load_config(OmegaConf.create(migrated))

    legacy_sections: dict[str, dict[str, Any]] = {
        "run": {
            "seed": raw.get("seed", 0),
            "max_num_iters": raw.get("max_num_iters", 400),
        },
        "env": {
            "id": raw.get("env_id", "go_9x9"),
            "board_size": raw.get("board_size"),
        },
        "model": {
            "network": raw.get("network", "aznet"),
            "num_channels": raw.get("num_channels", 128),
            "num_layers": raw.get("num_layers", 6),
            "resnet_v2": raw.get("resnet_v2", True),
        },
        "selfplay": {
            "batch_size": raw.get("selfplay_batch_size", 1024),
            "max_num_steps": raw.get("max_num_steps", 256),
            "search": {
                "kind": "gumbel",
                "gumbel": {
                    "num_simulations": raw.get("num_simulations", 32),
                },
            },
        },
        "training": {
            "batch_size": raw.get("training_batch_size", 4096),
            "learning_rate": raw.get("learning_rate", 0.001),
        },
        "eval": {
            "interval": raw.get("eval_interval", 5),
            "batch_size": raw.get("eval_batch_size", 16),
        },
        "logging": {
            "interval": raw.get("log_interval", 1),
            "wandb": {
                "enabled": raw.get("wandb_enabled", True),
                "project": raw.get("wandb_project", "scacchi-az"),
            },
        },
        "checkpointing": {
            "max_to_keep": raw.get("ckpt_max_to_keep", 3),
            "save_interval_steps": raw.get("ckpt_save_interval_steps", 50),
        },
    }
    return load_config(OmegaConf.create(legacy_sections))


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
        "config": config_to_dict(config),
        "step": step,
        "hours": hours,
        "frames": frames,
    }
    save_args = ocp.args.Composite(
        model=ocp.args.StandardSave(nnx.state(model)),
        optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
        rngs=ocp.args.StandardSave({"key": _rng_key_to_checkpoint_value(rng_key)}),
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
            rngs=ocp.args.StandardRestore(
                {"key": _rng_key_to_checkpoint_value(rng_key)}
            ),
            meta=ocp.args.JsonRestore(),
        ),
    )
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])
    rng_key = _rng_key_from_checkpoint_value(restored["rngs"]["key"], rng_key)
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

    options = _checkpoint_manager_options(read_only=True)
    with ocp.CheckpointManager(checkpoint_path, options=options) as manager:
        step = manager.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

        meta_restored = manager.restore(
            step, args=ocp.args.Composite(meta=ocp.args.JsonRestore())
        )
        config = _load_checkpoint_config(meta_restored["meta"]["config"])
        model = build_model(
            config,
            num_actions=env.num_actions,
            observation_shape=env.observation_shape,
            rngs=rngs,
        )
        restored = manager.restore(
            step,
            args=ocp.args.Composite(model=ocp.args.StandardRestore(nnx.state(model))),
        )

    nnx.update(model, restored["model"])
    return model
