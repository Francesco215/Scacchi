from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scacchi import logger as logger_module
from scacchi.logger import (
    CompositeLogger,
    LocalJsonlLogger,
    Logger,
    PrecomputedHistogram,
    WandbLogger,
    build_logger,
)
from scacchi.types import TrainConfig


class _CapturingLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: tuple[int, dict[str, Any], str] | None = None

    def log_metrics(
        self,
        step: int,
        metrics: Mapping[str, Any],
        prefix: str,
    ) -> None:
        self.logged = (step, dict(metrics), prefix)


class _CapturingPbar:
    def __init__(self) -> None:
        self.postfix: dict[str, str] | None = None
        self.update_count = 0

    def set_postfix(self, **postfix: str) -> None:
        self.postfix = postfix
        self.update_count += 1


def test_precomputed_histogram_canonicalizes_valid_arrays() -> None:
    histogram = PrecomputedHistogram(
        counts=np.array([0.0, 2.0, 3.0], dtype=np.float32),
        bin_edges=np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(histogram.counts, np.array([0, 2, 3]))
    np.testing.assert_array_equal(
        histogram.bin_edges,
        np.array([0.0, 1.0, 2.0, 4.0]),
    )
    assert histogram.counts.dtype == np.int64
    assert not histogram.counts.flags.writeable
    assert not histogram.bin_edges.flags.writeable


@pytest.mark.parametrize(
    ("counts", "bin_edges", "error"),
    [
        ([[1, 2]], [0, 1, 2], "one-dimensional"),
        ([1, 2], [[0, 1, 2]], "one-dimensional"),
        ([], [0], "at least one bucket"),
        ([1, 2], [0, 1], r"len\(counts\) \+ 1"),
        ([1, np.inf], [0, 1, 2], "finite"),
        ([1, -1], [0, 1, 2], "nonnegative"),
        ([1, 1.5], [0, 1, 2], "integer-like"),
        ([1, 2], [0, np.nan, 2], "finite"),
        ([1, 2], [0, 1, 1], "strictly increasing"),
        ([1, 2], [0, 2, 1], "strictly increasing"),
    ],
)
def test_precomputed_histogram_rejects_invalid_inputs(
    counts: Any,
    bin_edges: Any,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        PrecomputedHistogram(counts, bin_edges)


def test_precomputed_histogram_rejects_more_than_wandb_bucket_limit() -> None:
    with pytest.raises(ValueError, match="at most 512"):
        PrecomputedHistogram(np.zeros(513), np.arange(514))


def test_logger_preserves_histogram_for_backend_and_skips_it_in_pbar() -> None:
    histogram = PrecomputedHistogram([2, 3], [0.0, 1.0, 2.0])
    logger = _CapturingLogger()
    pbar = _CapturingPbar()

    logger.log(
        step=4,
        metrics={"loss": np.float32(1.25), "concentration": histogram},
        pbar=pbar,  # type: ignore[arg-type]
        prefix="train/",
    )

    assert logger.logged is not None
    step, metrics, prefix = logger.logged
    assert step == 4
    assert prefix == "train/"
    assert metrics == {"loss": 1.25, "concentration": histogram}
    assert pbar.postfix == {"loss": "1.2500"}
    assert pbar.update_count == 1


def test_wandb_logger_converts_histogram_in_same_log_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wandb

    calls: list[tuple[dict[str, Any], int | None]] = []

    def fake_log(payload: dict[str, Any], step: int | None = None) -> None:
        calls.append((payload, step))

    monkeypatch.setattr(wandb, "log", fake_log)
    logger = WandbLogger(project="test")
    histogram = PrecomputedHistogram([4, 1], [0.0, 2.0, 5.0])

    logger.log(
        step=7,
        metrics={"loss": 0.5, "concentration": histogram},
        prefix="train/",
    )

    assert len(calls) == 1
    payload, step = calls[0]
    assert step == 7
    assert payload["train/loss"] == 0.5
    assert payload["train/concentration"].to_json() == {
        "_type": "histogram",
        "values": [4, 1],
        "bins": [0.0, 2.0, 5.0],
    }


def test_local_jsonl_logger_writes_durable_metadata_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "nested" / "run.jsonl"
    real_fsync = logger_module.os.fsync
    fsync_calls: list[int] = []

    def recording_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    monkeypatch.setattr(logger_module.os, "fsync", recording_fsync)
    logger = LocalJsonlLogger(
        path,
        config={"run": {"seed": 7}},
    )
    histogram = PrecomputedHistogram([4, 1], [0.0, 2.0, 5.0])

    with logger:
        logger.log(
            step=3,
            metrics={
                "loss": np.float32(0.5),
                "count": 9,
                "nan": float("nan"),
                "positive_inf": float("inf"),
                "histogram": histogram,
                "ignored": "not a metric",
            },
            prefix="train/",
        )

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    metadata, metrics = records
    assert metadata["schema_version"] == 1
    assert metadata["record_type"] == "run_start"
    assert metadata["run_name"] == logger.run_name
    assert metadata["config"] == {"run": {"seed": 7}}
    assert isinstance(metadata["started_at_unix"], float)

    assert metrics["record_type"] == "metrics"
    assert metrics["step"] == 3
    assert metrics["run_name"] == logger.run_name
    assert metrics["metrics"] == {
        "train/count": 9,
        "train/histogram": {
            "_type": "histogram",
            "bin_edges": [0.0, 2.0, 5.0],
            "counts": [4, 1],
        },
        "train/loss": 0.5,
        "train/nan": {
            "_type": "nonfinite_float",
            "value": "nan",
        },
        "train/positive_inf": {
            "_type": "nonfinite_float",
            "value": "infinity",
        },
    }
    # One sync for metadata, one for metrics, and one before close.
    assert len(fsync_calls) >= 3


def test_local_jsonl_relative_path_uses_original_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        logger_module,
        "_original_working_directory",
        lambda: tmp_path,
    )

    logger = LocalJsonlLogger("logs/train.jsonl")

    assert logger.path == (tmp_path / "logs" / "train.jsonl").resolve()


def test_build_logger_selects_local_only_when_wandb_is_disabled(
    tmp_path: Path,
) -> None:
    config = TrainConfig()
    config.logging.wandb.enabled = False
    config.logging.jsonl_path = str(tmp_path / "run.jsonl")

    local = build_logger(config)
    assert isinstance(local, LocalJsonlLogger)
    assert local.log_every == config.logging.interval
    assert local.max_steps == config.run.max_num_iters

    config.logging.jsonl_path = None
    assert type(build_logger(config)) is Logger

    config.logging.jsonl_path = None
    config.logging.wandb.enabled = True
    assert isinstance(build_logger(config), WandbLogger)


def test_build_logger_selects_composite_when_both_backends_are_enabled(
    tmp_path: Path,
) -> None:
    config = TrainConfig()
    config.logging.wandb.enabled = True
    config.logging.jsonl_path = str(tmp_path / "run.jsonl")

    logger = build_logger(config)

    assert isinstance(logger, CompositeLogger)
    assert logger.log_every == config.logging.interval
    assert logger.max_steps == config.run.max_num_iters
    assert logger.wandb_logger.project == config.logging.wandb.project
    assert logger.local_logger.path == (tmp_path / "run.jsonl").resolve()


def test_composite_uses_wandb_name_and_logs_each_metric_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wandb

    init_calls: list[dict[str, Any]] = []
    log_calls: list[tuple[dict[str, Any], int | None]] = []
    finish_calls = 0

    def fake_init(**kwargs: Any) -> Any:
        init_calls.append(kwargs)
        return SimpleNamespace(name="wandb-authoritative-name")

    def fake_log(payload: dict[str, Any], step: int | None = None) -> None:
        log_calls.append((payload, step))

    def fake_finish() -> None:
        nonlocal finish_calls
        finish_calls += 1

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "log", fake_log)
    monkeypatch.setattr(wandb, "finish", fake_finish)

    path = tmp_path / "run.jsonl"
    logger = CompositeLogger(
        WandbLogger(project="test", log_every=2, max_steps=10),
        LocalJsonlLogger(path, log_every=2, max_steps=10),
    )
    pbar = _CapturingPbar()

    with logger:
        logger.log(
            step=2,
            metrics={"loss": np.float32(0.25)},
            pbar=pbar,  # type: ignore[arg-type]
        )

    assert init_calls == [
        {
            "project": "test",
            "config": None,
            "dir": None,
            "save_code": False,
        }
    ]
    assert finish_calls == 1
    assert logger.run_name == "wandb-authoritative-name"
    assert logger.wandb_logger.run_name == logger.run_name
    assert logger.local_logger.run_name == logger.run_name
    assert log_calls == [({"train/loss": 0.25}, 2)]
    assert pbar.postfix == {"loss": "0.2500"}
    assert pbar.update_count == 1

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record_type"] for record in records] == [
        "run_start",
        "metrics",
    ]
    assert all(
        record["run_name"] == "wandb-authoritative-name"
        for record in records
    )
    assert records[1]["metrics"] == {"train/loss": 0.25}


def test_composite_closes_both_backends_on_training_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wandb

    finish_calls = 0

    monkeypatch.setattr(
        wandb,
        "init",
        lambda **_: SimpleNamespace(name="failed-training-run"),
    )

    def fake_finish() -> None:
        nonlocal finish_calls
        finish_calls += 1

    monkeypatch.setattr(wandb, "finish", fake_finish)

    local = LocalJsonlLogger(tmp_path / "failed.jsonl")
    logger = CompositeLogger(WandbLogger(project="test"), local)

    with pytest.raises(RuntimeError, match="training failed"):
        with logger:
            raise RuntimeError("training failed")

    assert finish_calls == 1
    assert local._file is None
    assert not local._initialized
    assert not logger._initialized
