"""Training logger with tqdm progress bar and optional W&B backend."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal, Protocol, Self, TypeGuard

import jax
import numpy as np
from tqdm import tqdm

from .loss import CONCENTRATION_HISTOGRAM_BIN_EDGES, CONCENTRATION_HISTOGRAM_NUM_BINS, CONCENTRATION_HISTOGRAM_SERIES
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
    for head, lower in (("V", "v"), ("Q", "q")):
        dir_count = getattr(train_metrics, f"{lower}_dirichlet_target_count")
        cat_count = getattr(train_metrics, f"{lower}_categorical_target_count")
        native_count = getattr(train_metrics, f"{lower}_native_target_count")
        native_total = _count(native_count)
        result.update({
            f"train/alpha_{head}_concentration": _weighted_mean(getattr(train_metrics, f"alpha_{head}_concentration"), native_count),
            f"train/alpha_{head}_concentration_std": _pooled_std(getattr(train_metrics, f"alpha_{head}_concentration"), getattr(train_metrics, f"alpha_{head}_concentration_std"), native_count),
            f"train/{head}_C_pred_mean_dir": _weighted_mean(getattr(train_metrics, f"alpha_{head}_dirichlet_concentration"), dir_count),
            f"train/{head}_C_pred_std_dir": _pooled_std(getattr(train_metrics, f"alpha_{head}_dirichlet_concentration"), getattr(train_metrics, f"alpha_{head}_dirichlet_concentration_std"), dir_count),
            f"train/{head}_C_target_mean_dir": _weighted_mean(getattr(train_metrics, f"beta_{head}_concentration"), dir_count),
            f"train/{head}_C_target_std_dir": _pooled_std(getattr(train_metrics, f"beta_{head}_concentration"), getattr(train_metrics, f"beta_{head}_concentration_std"), dir_count),
            f"train/{head}_C_log_mae_dir": _weighted_mean(getattr(train_metrics, f"{lower}_dirichlet_log_concentration_mae"), dir_count),
            f"train/{head}_C_at_floor_fraction_dir": _weighted_mean(getattr(train_metrics, f"alpha_{head}_dirichlet_concentration_floor_fraction"), dir_count),
            f"train/{head}_C_at_clip_fraction_dir": _weighted_mean(getattr(train_metrics, f"alpha_{head}_dirichlet_concentration_clip_fraction"), dir_count),
            f"train/{head}_C_pred_mean_cat": _weighted_mean(getattr(train_metrics, f"alpha_{head}_categorical_concentration"), cat_count),
            f"train/{head}_C_at_clip_fraction_cat": _weighted_mean(getattr(train_metrics, f"alpha_{head}_categorical_concentration_clip_fraction"), cat_count),
            f"data/{lower}_categorical_target_fraction": 0.0 if native_total <= 0.0 else _count(cat_count) / native_total,
            f"data/{lower}_dirichlet_target_count": _count(dir_count),
            f"data/{lower}_categorical_target_count": _count(cat_count),
        })
    return result


def _search_metrics(train_metrics: Any) -> dict[str, float]:
    """Reduce additive search numerators and denominators exactly once."""

    def total(field: str) -> float:
        return _count(getattr(train_metrics, field))

    def mean(sum_field: str, count_field: str) -> float:
        denominator = total(count_field)
        return (
            0.0
            if denominator <= 0.0
            else total(sum_field) / denominator
        )

    root_count = total("search_root_count")
    legal_count = total("search_legal_action_count")
    requested_simulations = total("search_requested_simulation_count")
    expanded_nodes = total("search_expanded_node_count")
    active_simulation_rows = total("search_simulation_active_count")
    executed_simulation_rows = total(
        "search_executed_simulation_row_count"
    )
    policy_count = total("search_policy_kl_count")
    return {
        "search/policy_displacement_kl_nats": mean(
            "search_policy_kl_sum",
            "search_policy_kl_count",
        ),
        "search/v_semantic_displacement_kl_nats": mean(
            "search_v_semantic_kl_sum",
            "search_v_semantic_kl_count",
        ),
        "search/v_full_dirichlet_displacement_kl_nats": mean(
            "search_v_dirichlet_kl_sum",
            "search_v_dirichlet_kl_count",
        ),
        "search/v_categorical_prior_surprisal_nats": mean(
            "search_v_categorical_surprisal_sum",
            "search_v_categorical_surprisal_count",
        ),
        "search/q_semantic_displacement_kl_nats": mean(
            "search_q_semantic_kl_sum",
            "search_q_semantic_kl_count",
        ),
        "search/q_policy_weighted_semantic_displacement_kl_nats": mean(
            "search_q_policy_semantic_kl_sum",
            "search_q_policy_semantic_kl_count",
        ),
        "search/q_semantic_displacement_total_per_root_nats": (
            0.0
            if root_count <= 0.0
            else total("search_q_semantic_kl_sum") / root_count
        ),
        "search/q_full_dirichlet_displacement_kl_nats": mean(
            "search_q_dirichlet_kl_sum",
            "search_q_dirichlet_kl_count",
        ),
        "search/q_full_dirichlet_displacement_total_per_root_nats": (
            0.0
            if root_count <= 0.0
            else total("search_q_dirichlet_kl_sum") / root_count
        ),
        "search/q_categorical_prior_surprisal_nats": mean(
            "search_q_categorical_surprisal_sum",
            "search_q_categorical_surprisal_count",
        ),
        "search/root_count": root_count,
        "search/policy_target_count": total("search_policy_kl_count"),
        "search/v_dirichlet_target_count": total(
            "search_v_dirichlet_kl_count"
        ),
        "search/v_categorical_target_count": total(
            "search_v_categorical_surprisal_count"
        ),
        "search/q_dirichlet_target_count": total(
            "search_q_dirichlet_kl_count"
        ),
        "search/q_categorical_target_count": total(
            "search_q_categorical_surprisal_count"
        ),
        "search/legal_actions_mean": (
            0.0
            if root_count <= 0.0
            else legal_count / root_count
        ),
        "search/root_action_coverage": (
            0.0
            if legal_count <= 0.0
            else total("search_visited_action_count") / legal_count
        ),
        "search/root_repaired_action_fraction": (
            0.0
            if legal_count <= 0.0
            else total("search_repaired_action_count") / legal_count
        ),
        "search/root_categorical_action_fraction": (
            0.0
            if legal_count <= 0.0
            else total("search_categorical_action_count") / legal_count
        ),
        "search/solved_root_fraction": (
            0.0
            if root_count <= 0.0
            else total("search_solved_root_count") / root_count
        ),
        "search/expanded_nodes_mean": (
            0.0
            if root_count <= 0.0
            else expanded_nodes / root_count
        ),
        "search/initialized_node_rows_total": expanded_nodes,
        "search/useful_recurrent_rows_total": active_simulation_rows,
        "search/useful_recurrent_rows_mean": (
            0.0
            if root_count <= 0.0
            else active_simulation_rows / root_count
        ),
        "search/executed_recurrent_rows_total": executed_simulation_rows,
        "search/executed_recurrent_rows_mean": (
            0.0
            if root_count <= 0.0
            else executed_simulation_rows / root_count
        ),
        "search/recurrent_row_utilization": (
            0.0
            if executed_simulation_rows <= 0.0
            else active_simulation_rows / executed_simulation_rows
        ),
        "search/requested_recurrent_rows_total": requested_simulations,
        "search/requested_simulations_mean": (
            0.0
            if root_count <= 0.0
            else requested_simulations / root_count
        ),
        "search/expanded_node_fraction_of_requested": (
            0.0
            if requested_simulations <= 0.0
            else expanded_nodes / requested_simulations
        ),
        "search/useful_model_evaluation_rows_total": (
            root_count + active_simulation_rows
        ),
        "search/useful_model_evaluation_rows_mean": (
            0.0
            if root_count <= 0.0
            else (root_count + active_simulation_rows) / root_count
        ),
        "search/executed_model_evaluation_rows_total": (
            root_count + executed_simulation_rows
        ),
        "search/model_evaluation_row_requested_upper_total": (
            root_count + requested_simulations
        ),
        "search/structural_support_mean": (
            0.0
            if root_count <= 0.0
            else total("search_structural_support_sum") / root_count
        ),
        "search/max_depth_mean": (
            0.0
            if root_count <= 0.0
            else total("search_max_depth_sum") / root_count
        ),
        "search/policy_support_mean": (
            0.0
            if policy_count <= 0.0
            else total("search_policy_support_sum") / policy_count
        ),
        "search/policy_ess_mean": (
            0.0
            if policy_count <= 0.0
            else total("search_policy_ess_sum") / policy_count
        ),
        "search/policy_prior_target_top1_agreement": (
            0.0
            if policy_count <= 0.0
            else total("search_policy_top1_agreement_count")
            / policy_count
        ),
    }


def _capture_metrics(train_metrics: Any) -> dict[str, float]:
    """Report raw fixed-probe gaps and their optimizer capture fractions."""

    def total(field: str) -> float:
        return _count(getattr(train_metrics, field))

    result: dict[str, float] = {}
    populations = {
        "policy": ("policy", "count"),
        "v_semantic": ("v_semantic", "count"),
        "v_full_dirichlet": ("v_dirichlet", "count"),
        "q_semantic": ("q_semantic", "count"),
        "q_full_dirichlet": ("q_dirichlet", "count"),
        "q_loss_weighted_semantic": ("q_weighted_semantic", "weight"),
        "q_loss_weighted_full_dirichlet": (
            "q_weighted_dirichlet",
            "weight",
        ),
    }
    for log_name, (field_name, denominator_name) in populations.items():
        before_count = total(
            f"capture_{field_name}_before_{denominator_name}"
        )
        after_count = total(
            f"capture_{field_name}_after_{denominator_name}"
        )
        before = (
            0.0
            if before_count <= 0.0
            else total(f"capture_{field_name}_before_sum") / before_count
        )
        after = (
            0.0
            if after_count <= 0.0
            else total(f"capture_{field_name}_after_sum") / after_count
        )
        delta = before - after
        # A changed finite population makes the two means incomparable.  The
        # raw gaps and both counts remain visible for diagnosis.
        comparable = (
            before_count > 0.0
            and after_count > 0.0
            and before_count == after_count
        )
        fraction_defined = (
            comparable
            and before > float(np.finfo(np.float32).eps)
        )
        capture = delta / before if fraction_defined else 0.0
        prefix = f"capture/train_probe/{log_name}"
        denominator_label = (
            "count" if denominator_name == "count" else "total_weight"
        )
        result.update(
            {
                f"{prefix}_gap_before_nats": before,
                f"{prefix}_gap_after_nats": after,
                f"{prefix}_gap_delta_nats": delta,
                f"{prefix}_fraction": capture,
                f"{prefix}_fraction_defined": int(fraction_defined),
                f"{prefix}_{denominator_label}_before": before_count,
                f"{prefix}_{denominator_label}_after": after_count,
            }
        )
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


def training_metrics(
    train_metrics: Any,
    *,
    seconds: float,
    hours: float,
    frames: int,
    frames_this_iteration: int,
    optimizer_updates: int | None = None,
    optimizer_updates_this_iteration: int | None = None,
    completed_iterations: int | None = None,
) -> dict[str, Metric]:
    """Turn one iteration's device metrics into the experiment log payload."""

    data_frame_count = _count(train_metrics.data_frame_count)
    data_termination_count = _count(train_metrics.data_termination_count)
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
        "train/q_loss_weight_mean": train_metrics.q_loss_weight_mean.mean().item(),
        "data/value_mask_fraction": train_metrics.data_value_mask_fraction.mean().item(),
        "data/frame_count": data_frame_count,
        "data/termination_count": data_termination_count,
        "data/terminal_events_per_1k_frames": (
            0.0
            if data_frame_count <= 0.0
            else 1000.0 * data_termination_count / data_frame_count
        ),
        "data/pass_fraction": train_metrics.data_pass_fraction.mean().item(),
        "data/terminations_per_row": train_metrics.data_terminations_per_row.mean().item(),
        "data/psk_termination_fraction": train_metrics.data_psk_termination_fraction.mean().item(),
        "train/iter_seconds": seconds,
        "train/frames_per_second": frames_this_iteration / max(seconds, 1e-12),
        "train/updates_this_iteration": int(
            np.asarray(jax.device_get(train_metrics.policy_loss)).size
        ),
        "train/hours": hours,
        "train/frames": frames,
    }
    metrics.update(_concentration_metrics(train_metrics))
    metrics.update(_search_metrics(train_metrics))
    metrics.update(_capture_metrics(train_metrics))
    metrics.update(concentration_histograms(train_metrics))
    if optimizer_updates is not None:
        metrics["train/optimizer_updates"] = optimizer_updates
    if optimizer_updates_this_iteration is not None:
        metrics["train/optimizer_updates_this_iteration"] = (
            optimizer_updates_this_iteration
        )
    if completed_iterations is not None:
        metrics["train/completed_iterations"] = completed_iterations
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


def _original_working_directory() -> Path:
    """Return Hydra's launch directory, falling back outside Hydra."""

    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd())
    except (RuntimeError, ValueError):
        return Path.cwd()


def _resolve_local_path(path: str | Path) -> Path:
    raw = str(path)
    if not raw.strip():
        raise ValueError("logging.jsonl_path must not be empty")
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = _original_working_directory() / resolved
    return resolved.resolve()


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
            save_code=False,
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


class LocalJsonlLogger(Logger):
    """Durable append-only scalar and precomputed-histogram logger."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        log_every: int = 1,
        max_steps: int | None = None,
        config: Any = None,
    ) -> None:
        super().__init__(log_every=log_every, max_steps=max_steps)
        self.path = _resolve_local_path(path)
        self.config = config
        self._file: Any = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", newline="\n")
        self._initialized = True
        record: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": "run_start",
            "run_name": self.run_name,
            "started_at_unix": time.time(),
        }
        if self.config is not None:
            record["config"] = self.config
        self._write_record(record)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        if self._file is not None:
            self._sync()
            self._file.close()
            self._file = None
        self._initialized = False
        return False

    @staticmethod
    def _metric_value(value: Metric) -> Any:
        if isinstance(value, PrecomputedHistogram):
            return {
                "_type": "histogram",
                "counts": value.counts.tolist(),
                "bin_edges": value.bin_edges.tolist(),
            }
        if isinstance(value, float) and not np.isfinite(value):
            if np.isnan(value):
                encoded = "nan"
            elif value > 0:
                encoded = "infinity"
            else:
                encoded = "-infinity"
            return {"_type": "nonfinite_float", "value": encoded}
        return value

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(
            f"value of type {type(value).__name__} is not JSON serializable"
        )

    def _sync(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        os.fsync(self._file.fileno())

    def _write_record(self, record: Mapping[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError(
                "LocalJsonlLogger must be entered before logging"
            )
        encoded = json.dumps(
            record,
            allow_nan=False,
            default=self._json_default,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._file.write(encoded)
        self._file.write("\n")
        self._sync()

    def log_metrics(
        self,
        step: int,
        metrics: Mapping[str, Metric],
        prefix: str = "train/",
    ) -> None:
        payload = {
            f"{prefix}{key}": self._metric_value(value)
            for key, value in metrics.items()
        }
        self._write_record(
            {
                "schema_version": self.SCHEMA_VERSION,
                "record_type": "metrics",
                "run_name": self.run_name,
                "logged_at_unix": time.time(),
                "step": int(step),
                "metrics": payload,
            }
        )


class CompositeLogger(Logger):
    """Log once to W&B and a durable local JSONL file."""

    def __init__(
        self,
        wandb_logger: WandbLogger,
        local_logger: LocalJsonlLogger,
    ) -> None:
        if wandb_logger.log_every != local_logger.log_every:
            raise ValueError("composite loggers must use the same log interval")
        if wandb_logger.max_steps != local_logger.max_steps:
            raise ValueError("composite loggers must use the same maximum step")
        super().__init__(
            log_every=wandb_logger.log_every,
            max_steps=wandb_logger.max_steps,
        )
        self.wandb_logger = wandb_logger
        self.local_logger = local_logger
        self._exit_stack: ExitStack | None = None

    def __enter__(self) -> Self:
        # ExitStack makes entry transactional: if the local file cannot be
        # opened, the already-started W&B run is still finished.
        with ExitStack() as stack:
            stack.enter_context(self.wandb_logger)
            self.run_name = self.wandb_logger.run_name
            self.local_logger.run_name = self.run_name
            stack.enter_context(self.local_logger)
            self._exit_stack = stack.pop_all()
        self._initialized = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        stack = self._exit_stack
        self._exit_stack = None
        try:
            if stack is not None:
                stack.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._initialized = False
        return False

    def log_metrics(
        self,
        step: int,
        metrics: Mapping[str, Metric],
        prefix: str = "train/",
    ) -> None:
        # Logger.log owns cadence, conversion, and progress-bar updates. Calling
        # the backend primitives here avoids doing any of those twice.
        self.wandb_logger.log_metrics(step, metrics, prefix)
        self.local_logger.log_metrics(step, metrics, prefix)

    def log_image(self, *args: Any, **kwargs: Any) -> None:
        self.wandb_logger.log_image(*args, **kwargs)

    def log_video(self, *args: Any, **kwargs: Any) -> None:
        self.wandb_logger.log_video(*args, **kwargs)


def build_logger(
    training_config: Any,
    dir: str | None = None,
) -> Logger:
    wandb_enabled = bool(training_config.logging.wandb.enabled)
    wandb_project = str(training_config.logging.wandb.project)
    jsonl_path = getattr(training_config.logging, "jsonl_path", None)
    log_every = int(training_config.logging.interval)
    max_steps = int(training_config.run.max_num_iters)
    config = _config_to_dict(training_config)

    if wandb_enabled:
        wandb_logger = WandbLogger(
            wandb_project,
            log_every=log_every,
            max_steps=max_steps,
            config=config,
            dir=dir,
        )
        if jsonl_path is not None:
            return CompositeLogger(
                wandb_logger,
                LocalJsonlLogger(
                    jsonl_path,
                    log_every=log_every,
                    max_steps=max_steps,
                    config=config,
                ),
            )
        return wandb_logger
    if jsonl_path is not None:
        return LocalJsonlLogger(
            jsonl_path,
            log_every=log_every,
            max_steps=max_steps,
            config=config,
        )
    return Logger(log_every=log_every, max_steps=max_steps)
