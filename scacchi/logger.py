"""Training logger with tqdm progress bar and optional W&B backend."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, Self, TypeGuard

import numpy as np
from tqdm import tqdm

from .types import config_to_dict

Scalar = float | int


class PrecomputedHistogram:
    """Backend-neutral histogram represented by bucket counts and bin edges."""

    __slots__ = ("counts", "bin_edges")

    counts: np.ndarray
    bin_edges: np.ndarray

    def __init__(self, counts: Any, bin_edges: Any) -> None:
        try:
            counts_array = np.asarray(counts, dtype=np.float64)
            edges_array = np.asarray(bin_edges, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("histogram counts and bin edges must be numeric") from error

        if counts_array.ndim != 1:
            raise ValueError("histogram counts must be one-dimensional")
        if edges_array.ndim != 1:
            raise ValueError("histogram bin edges must be one-dimensional")
        if counts_array.size == 0:
            raise ValueError("a histogram must contain at least one bucket")
        if counts_array.size > 512:
            raise ValueError("a histogram may contain at most 512 buckets")
        if edges_array.size != counts_array.size + 1:
            raise ValueError("histogram bin edges must have len(counts) + 1 entries")

        if not np.all(np.isfinite(counts_array)):
            raise ValueError("histogram counts must be finite")
        if np.any(counts_array < 0):
            raise ValueError("histogram counts must be nonnegative")
        rounded_counts = np.rint(counts_array)
        if not np.array_equal(counts_array, rounded_counts):
            raise ValueError("histogram counts must be integer-like")
        if np.any(rounded_counts > np.iinfo(np.int64).max):
            raise ValueError("histogram counts exceed the supported integer range")

        if not np.all(np.isfinite(edges_array)):
            raise ValueError("histogram bin edges must be finite")
        if not np.all(np.diff(edges_array) > 0):
            raise ValueError("histogram bin edges must be strictly increasing")

        self.counts = rounded_counts.astype(np.int64)
        self.bin_edges = edges_array
        self.counts.setflags(write=False)
        self.bin_edges.setflags(write=False)


Metric = Scalar | PrecomputedHistogram


def _to_scalar(value: Any) -> Scalar:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (float, int)):
        return value
    return float(value)


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return config_to_dict(config)
    if hasattr(config, "dict"):
        return dict(config.dict())
    return dict(config)


def returns_metrics(prefix: str, returns: Any) -> dict[str, Scalar]:
    """Convert eval returns into scalar win/draw/loss metrics."""

    return {
        f"{prefix}/avg_R": _to_scalar(returns.mean()),
        f"{prefix}/win_rate": _to_scalar((returns == 1).sum() / returns.size),
        f"{prefix}/draw_rate": _to_scalar((returns == 0).sum() / returns.size),
        f"{prefix}/lose_rate": _to_scalar((returns == -1).sum() / returns.size),
    }


class _HasAsDict(Protocol):
    def _asdict(self) -> dict[str, Any]: ...


def _has_asdict(obj: Any) -> TypeGuard[_HasAsDict]:
    return hasattr(obj, "_asdict")


class Logger:
    """Base logger with no-op backend and tqdm progress bar updates."""

    def __init__(self, log_every: int = 1, max_steps: int | None = None) -> None:
        self.log_every = log_every
        self.max_steps = max_steps
        self._initialized = False
        self.run_name: str = f"local-{time.strftime('%Y%m%d-%H%M%S')}"

    def should_log(self, step: int) -> bool:
        if self.log_every <= 0:
            return False
        is_periodic = step % self.log_every == 0
        is_last = self.max_steps is not None and step == self.max_steps
        return is_periodic or is_last

    def __enter__(self) -> Self:
        self._initialized = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        self._initialized = False
        return False

    def log_metrics(self, step: int, metrics: Mapping[str, Metric], prefix: str) -> None:
        pass

    def log_returns(
        self,
        step: int,
        returns: Any,
        prefix: str = "eval/returns",
    ) -> None:
        self.log_metrics(step, returns_metrics(prefix, returns), prefix="")

    def log_image(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_video(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log(
        self,
        step: int,
        metrics: Any,
        pbar: tqdm[Any] | None = None,
        prefix: str = "train/",
        float_fmt: str = ".4f",
        pbar_filter: str | None = None,
        **extra: Scalar,
    ) -> None:
        if not self.should_log(step):
            return

        raw: dict[str, Any] = (
            dict(metrics._asdict()) if _has_asdict(metrics) else dict(metrics)
        )
        raw.update(extra)
        clean = self._convert_metrics(raw)

        if pbar is not None:
            self._update_pbar(pbar, clean, float_fmt, pbar_filter)

        self.log_metrics(step, clean, prefix)

    def _convert_metrics(self, metrics: dict[str, Any]) -> dict[str, Metric]:
        clean: dict[str, Metric] = {}
        for key, value in metrics.items():
            if isinstance(value, PrecomputedHistogram):
                clean[key] = value
                continue
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (float, int)):
                clean[key] = value
                continue
            try:
                clean[key] = float(value)
            except (ValueError, TypeError):
                continue
        return clean

    def _update_pbar(
        self,
        pbar: tqdm[Any],
        metrics: Mapping[str, Metric],
        float_fmt: str,
        pbar_filter: str | None,
    ) -> None:
        filtered = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (float, int))
        }
        if pbar_filter is not None:
            pattern = re.compile(pbar_filter)
            filtered = {
                key: value for key, value in filtered.items() if pattern.search(key)
            }

        postfix = {
            key: f"{value:{float_fmt}}" if isinstance(value, float) else str(value)
            for key, value in filtered.items()
        }
        pbar.set_postfix(**postfix)


class WandbLogger(Logger):
    def __init__(
        self,
        project: str,
        log_every: int = 1,
        max_steps: int | None = None,
        config: Any = None,
        dir: str | None = None,
    ) -> None:
        super().__init__(log_every=log_every, max_steps=max_steps)
        self.project = project
        self.config = config
        self.dir = dir
        self._run: Any = None

    def __enter__(self) -> Self:
        import wandb

        self._run = wandb.init(
            project=self.project,
            config=self.config,
            dir=self.dir,
            save_code=True,
        )
        if self._run is not None and getattr(self._run, "name", None):
            self.run_name = str(self._run.name)
        self._initialized = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        if self._run is not None:
            import wandb

            wandb.finish()
        self._initialized = False
        return False

    def log_metrics(
        self,
        step: int,
        metrics: Mapping[str, Metric],
        prefix: str = "train/",
    ) -> None:
        import wandb

        payload = {
            f"{prefix}{key}": (
                wandb.Histogram(
                    np_histogram=(value.counts, value.bin_edges),
                )
                if isinstance(value, PrecomputedHistogram)
                else value
            )
            for key, value in metrics.items()
        }
        wandb.log(
            payload,
            step=step,
        )

    def log_image(
        self,
        step: int,
        key: str,
        image: Any,
        caption: str | None = None,
    ) -> None:
        import numpy as np
        import wandb

        if hasattr(image, "__array__"):
            image = np.asarray(image)
        wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def log_video(
        self,
        step: int,
        key: str,
        video_path: Path,
        format: Literal["gif", "mp4", "webm", "ogg"] | None = "mp4",
        **kwargs: Any,
    ) -> None:
        import wandb

        wandb.log(
            {key: wandb.Video(str(video_path), format=format, **kwargs)},
            step=step,
        )


def build_logger(
    training_config: Any,
    dir: str | None = None,
) -> Logger:
    wandb_enabled = bool(training_config.logging.wandb.enabled)
    wandb_project = str(training_config.logging.wandb.project)
    log_every = int(training_config.logging.interval)
    max_steps = int(training_config.run.max_num_iters)
    config = _config_to_dict(training_config)

    if wandb_enabled:
        return WandbLogger(
            wandb_project,
            log_every=log_every,
            max_steps=max_steps,
            config=config,
            dir=dir,
        )
    return Logger(log_every=log_every, max_steps=max_steps)
