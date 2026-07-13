from pathlib import Path
from typing import Any

import jax
import numpy as np

from scacchi import checkpoint
from scacchi.types import CheckpointingConfig, Config, RunConfig


def test_legacy_flat_checkpoint_config_loads_for_solved_baselines() -> None:
    config = checkpoint._load_checkpoint_config(
        {
            "env_id": "hex",
            "board_size": 5,
            "seed": 0,
            "max_num_iters": 300,
            "network": "boardlaw",
            "num_channels": 512,
            "num_layers": 8,
            "resnet_v2": True,
            "selfplay_batch_size": 1024,
            "num_simulations": 32,
            "max_num_steps": 256,
            "training_batch_size": 4096,
            "learning_rate": 1e-3,
            "log_interval": 1,
            "eval_interval": 0,
            "eval_batch_size": 64,
            "wandb_enabled": True,
            "wandb_project": "scacchi-az",
            "ckpt_max_to_keep": 3,
            "ckpt_save_interval_steps": 50,
            # Removed MoHex fields in the real baseline metadata are ignored.
            "mohex_max_memory": 207108864,
        }
    )

    assert config.env.id == "hex"
    assert config.env.board_size == 5
    assert config.model.network == "boardlaw"
    assert config.model.num_channels == 512
    assert config.model.num_layers == 8
    assert config.selfplay.search.kind == "gumbel"
    assert config.selfplay.search.gumbel.num_simulations == 32
    assert config.eval.interval == 0
    assert config.eval.batch_size == 64


def test_rng_key_checkpoint_value_is_host_numpy_array() -> None:
    rng_key = jax.random.PRNGKey(7)

    value = checkpoint._rng_key_to_checkpoint_value(rng_key)

    assert isinstance(value, np.ndarray)
    assert value.dtype == np.uint32
    np.testing.assert_array_equal(value, np.asarray([0, 7], dtype=np.uint32))


def test_rng_key_restore_returns_jax_key_with_template_dtype() -> None:
    template = jax.random.PRNGKey(0)
    value = np.asarray([123, 456], dtype=np.uint32)

    restored = checkpoint._rng_key_from_checkpoint_value(value, template)

    assert isinstance(restored, jax.Array)
    assert restored.dtype == template.dtype
    np.testing.assert_array_equal(np.asarray(jax.device_get(restored)), value)


def test_typed_rng_key_checkpoint_round_trip() -> None:
    rng_key = jax.random.key(7)

    value = checkpoint._rng_key_to_checkpoint_value(rng_key)
    restored = checkpoint._rng_key_from_checkpoint_value(value, rng_key)

    assert isinstance(value, np.ndarray)
    assert restored.dtype == rng_key.dtype
    assert jax.random.key_impl(restored) == jax.random.key_impl(rng_key)
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(restored)),
        np.asarray(jax.random.key_data(rng_key)),
    )


def test_build_checkpoint_manager_uses_multihost_orbax_options(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeCheckpointManager:
        def __init__(
            self,
            directory: Path,
            *,
            options: Any,
            item_names: tuple[str, ...],
        ) -> None:
            captured["directory"] = directory
            captured["options"] = options
            captured["item_names"] = item_names

    config = Config(
        run=RunConfig(max_num_iters=11),
        checkpointing=CheckpointingConfig(max_to_keep=3, save_interval_steps=7),
    )
    monkeypatch.setattr(checkpoint.jax, "process_count", lambda: 2)
    monkeypatch.setattr(checkpoint.ocp, "CheckpointManager", FakeCheckpointManager)

    manager = checkpoint.build_checkpoint_manager(config, tmp_path)

    assert isinstance(manager, FakeCheckpointManager)
    assert captured["directory"] == tmp_path
    assert captured["item_names"] == ("model", "optimizer", "rngs", "meta")

    options = captured["options"]
    assert options.max_to_keep == 3
    assert options.save_interval_steps == 7
    assert options.save_on_steps == frozenset({10})
    # Pod workers have private disks: process 0 saves alone, no collectives.
    assert options.single_host_load_and_broadcast is False
    assert options.enable_async_checkpointing is True
    assert options.multiprocessing_options.primary_host == 0
    assert options.multiprocessing_options.active_processes == {0}
    assert options.create is False


def test_build_checkpoint_manager_multihost_nonprimary_is_noop(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config = Config(
        run=RunConfig(max_num_iters=11),
        checkpointing=CheckpointingConfig(max_to_keep=3, save_interval_steps=7),
    )
    monkeypatch.setattr(checkpoint.jax, "process_count", lambda: 2)
    monkeypatch.setattr(checkpoint.jax, "process_index", lambda: 1)

    def fail_if_constructed(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("non-primary processes must not build orbax managers")

    monkeypatch.setattr(checkpoint.ocp, "CheckpointManager", fail_if_constructed)

    manager = checkpoint.build_checkpoint_manager(config, tmp_path)
    assert isinstance(manager, checkpoint.NoOpCheckpointManager)


def test_disabled_checkpoint_manager_does_not_construct_orbax(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config = Config(
        run=RunConfig(max_num_iters=11),
        checkpointing=CheckpointingConfig(max_to_keep=0, save_interval_steps=7),
    )

    def fail_if_constructed(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Orbax CheckpointManager should not be constructed")

    monkeypatch.setattr(checkpoint.ocp, "CheckpointManager", fail_if_constructed)

    manager = checkpoint.build_checkpoint_manager(config, tmp_path)

    assert isinstance(manager, checkpoint.NoOpCheckpointManager)
    assert manager.directory == tmp_path
    assert manager.latest_step() is None
    assert manager.should_save(10) is False
    assert manager.save(10) is False
    with manager as entered:
        assert entered is manager
