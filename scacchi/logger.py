"""Training logger with tqdm progress bar and optional W&B backend."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, Self, TypeGuard

import jax
import numpy as np
from tqdm import tqdm

from .histogram import CONCENTRATION_HISTOGRAM_BIN_EDGES, CONCENTRATION_HISTOGRAM_NUM_BINS, CONCENTRATION_HISTOGRAM_SERIES
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


def _weighted_mean(values: Any, counts: Any) -> float:
    values = np.asarray(jax.device_get(values), dtype=np.float64)
    counts = np.asarray(jax.device_get(counts), dtype=np.float64)
    total = float(np.sum(counts))
    return 0.0 if total <= 0.0 else float(np.sum(values * counts) / total)


def _pooled_std(means: Any, stds: Any, counts: Any) -> float:
    means = np.asarray(jax.device_get(means), dtype=np.float64)
    stds = np.asarray(jax.device_get(stds), dtype=np.float64)
    counts = np.asarray(jax.device_get(counts), dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    mean = float(np.sum(means * counts) / total)
    second_moment = float(np.sum((np.square(stds) + np.square(means)) * counts) / total)
    return float(np.sqrt(max(second_moment - mean * mean, 0.0)))


def _count(values: Any) -> float:
    return float(np.sum(np.asarray(jax.device_get(values), dtype=np.float64)))


def _concentration_metrics(train_metrics: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    legacy_flags = np.asarray(
        jax.device_get(train_metrics.dirichlet_head_is_legacy),
        dtype=np.bool_,
    )
    is_legacy_head = bool(np.all(legacy_flags))
    for head, lower in (("V", "v"), ("Q", "q")):
        dir_count = getattr(train_metrics, f"{lower}_dirichlet_target_count")
        cat_count = getattr(train_metrics, f"{lower}_categorical_target_count")
        native_count = getattr(train_metrics, f"{lower}_native_target_count")
        native_total = _count(native_count)
        head_metrics: dict[str, float] = {
            f"train/alpha_{head}_concentration": _weighted_mean(getattr(train_metrics, f"alpha_{head}_concentration"), native_count),
            f"train/alpha_{head}_concentration_std": _pooled_std(getattr(train_metrics, f"alpha_{head}_concentration"), getattr(train_metrics, f"alpha_{head}_concentration_std"), native_count),
            f"train/{head}_C_pred_mean_dir": _weighted_mean(getattr(train_metrics, f"alpha_{head}_dirichlet_concentration"), dir_count),
            f"train/{head}_C_pred_std_dir": _pooled_std(getattr(train_metrics, f"alpha_{head}_dirichlet_concentration"), getattr(train_metrics, f"alpha_{head}_dirichlet_concentration_std"), dir_count),
            f"train/{head}_C_target_mean_dir": _weighted_mean(getattr(train_metrics, f"beta_{head}_concentration"), dir_count),
            f"train/{head}_C_target_std_dir": _pooled_std(getattr(train_metrics, f"beta_{head}_concentration"), getattr(train_metrics, f"beta_{head}_concentration_std"), dir_count),
            f"train/{head}_C_log_mae_dir": _weighted_mean(getattr(train_metrics, f"{lower}_dirichlet_log_concentration_mae"), dir_count),
            f"train/{head}_C_pred_mean_cat": _weighted_mean(getattr(train_metrics, f"alpha_{head}_categorical_concentration"), cat_count),
            f"train/{head}_C_at_or_above_reference_fraction_cat": _weighted_mean(getattr(train_metrics, f"alpha_{head}_categorical_concentration_reference_fraction"), cat_count),
            f"data/{lower}_categorical_target_fraction": 0.0 if native_total <= 0.0 else _count(cat_count) / native_total,
            f"data/{lower}_dirichlet_target_count": _count(dir_count),
            f"data/{lower}_categorical_target_count": _count(cat_count),
        }
        if is_legacy_head:
            head_metrics.update({
                f"train/{head}_C_at_floor_fraction_dir": _weighted_mean(getattr(train_metrics, f"alpha_{head}_dirichlet_concentration_floor_fraction"), dir_count),
                f"train/{head}_C_at_clip_fraction_dir": _weighted_mean(getattr(train_metrics, f"alpha_{head}_dirichlet_concentration_clip_fraction"), dir_count),
                f"train/{head}_C_at_clip_fraction_cat": _weighted_mean(getattr(train_metrics, f"alpha_{head}_categorical_concentration_reference_fraction"), cat_count),
            })
        result.update(head_metrics)
    return result


def concentration_histograms(train_metrics: Any) -> dict[str, PrecomputedHistogram]:
    counts = np.asarray(jax.device_get(train_metrics.dirichlet_concentration_histogram_counts), dtype=np.float64)
    expected_tail = (len(CONCENTRATION_HISTOGRAM_SERIES), CONCENTRATION_HISTOGRAM_NUM_BINS)
    if counts.ndim < 2 or counts.shape[-2:] != expected_tail:
        raise ValueError(f"concentration histogram counts must end in {expected_tail}; got {counts.shape}.")
    leading_axes = tuple(range(counts.ndim - 2))
    pooled = np.sum(counts, axis=leading_axes) if leading_axes else counts
    edges = np.asarray(CONCENTRATION_HISTOGRAM_BIN_EDGES, dtype=np.float64)
    histograms: dict[str, PrecomputedHistogram] = {}
    for index, series in enumerate(CONCENTRATION_HISTOGRAM_SERIES):
        head, role = series.split("_", maxsplit=1)
        histograms[f"train/{head}_C_{role}_hist_dir"] = PrecomputedHistogram(pooled[index], edges)
    return histograms


def training_metrics(train_metrics: Any, *, seconds: float, hours: float, frames: int, frames_this_iteration: int) -> dict[str, Metric]:
    """Turn one iteration's device metrics into the experiment log payload."""

    metrics: dict[str, Metric] = {
        "train/policy_loss": train_metrics.policy_loss.mean().item(),
        "train/value_loss": train_metrics.value_loss.mean().item(),
        "train/policy_nll_loss": train_metrics.policy_nll_loss.mean().item(),
        "train/policy_kl_hat": train_metrics.policy_kl_hat.mean().item(),
        "train/policy_target_entropy": train_metrics.policy_target_entropy.mean().item(),
        "train/value_dir_kl_loss": train_metrics.value_dir_kl_loss.mean().item(),
        "train/q_dir_kl_loss": train_metrics.q_dir_kl_loss.mean().item(),
        "train/value_outcome_loss": train_metrics.value_outcome_loss.mean().item(),
        "train/q_outcome_loss": train_metrics.q_outcome_loss.mean().item(),
        "train/q_supervised_actions_per_row": (
            train_metrics.q_supervised_actions_per_row.mean().item()
        ),
        "data/q_positive_evidence_action_count": _count(
            train_metrics.q_positive_evidence_action_count
        ),
        "data/q_positive_policy_action_count": _count(
            train_metrics.q_positive_policy_action_count
        ),
        "data/q_solved_action_count": _count(
            train_metrics.q_solved_action_count
        ),
        "data/q_supervised_action_count": _count(
            train_metrics.q_supervised_action_count
        ),
        "data/q_supervised_action_fraction": (
            train_metrics.q_supervised_action_fraction.mean().item()
        ),
        "data/value_mask_fraction": train_metrics.data_value_mask_fraction.mean().item(),
        "data/pass_fraction": train_metrics.data_pass_fraction.mean().item(),
        "data/terminations_per_row": train_metrics.data_terminations_per_row.mean().item(),
        "data/psk_termination_fraction": train_metrics.data_psk_termination_fraction.mean().item(),
        "train/iter_seconds": seconds,
        "train/frames_per_second": frames_this_iteration / max(seconds, 1e-12),
        "train/hours": hours,
        "train/frames": frames,
    }
    metrics.update(_concentration_metrics(train_metrics))
    metrics.update(concentration_histograms(train_metrics))
    return metrics


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
