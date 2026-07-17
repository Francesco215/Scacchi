from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from scacchi.logger import Logger, PrecomputedHistogram, WandbLogger


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

    def set_postfix(self, **postfix: str) -> None:
        self.postfix = postfix


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
