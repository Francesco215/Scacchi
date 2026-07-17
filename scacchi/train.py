# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "scacchi"

if any(Path("/dev").glob("accel*")):
    os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
from hydra.utils import get_original_cwd
import jax

from .distributed import constrain_batch_axis, initialize_distributed, make_batch_parallel

initialize_distributed()

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from flax import nnx
import optax
import pgx

from .checkpoint import build_checkpoint_manager, from_pretrained, maybe_save, restore
from .envs import make_env
from .evaluations import make_mcts_evaluate
from .logger import PrecomputedHistogram, build_logger, returns_metrics
from .loss import (
    CONCENTRATION_HISTOGRAM_BIN_EDGES,
    CONCENTRATION_HISTOGRAM_NUM_BINS,
    CONCENTRATION_HISTOGRAM_SERIES,
)
from .network import build_model
from .pipeline import make_training_iteration
from .types import Config, EvalBaseline, load_config


def report_jax_backend() -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    process_index = jax.process_index()
    process_count = jax.process_count()
    if process_index == 0:
        print(f"JAX backend: {backend}")
        print(f"JAX process count: {process_count}")
        print(f"JAX global devices: {devices}")
    print(f"JAX process {process_index} local devices: {jax.local_devices()}")
    if os.environ.get("SCACCHI_ALLOW_CPU") != "1" and backend not in {"gpu", "tpu"}:
        raise RuntimeError(
            "JAX is not using a GPU or TPU backend. Set SCACCHI_ALLOW_CPU=1 "
            "only for intentional CPU runs."
        )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        parsed = default
    else:
        parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {parsed}.")
    return parsed


def _profile_enabled_for_process() -> bool:
    value = os.environ.get("SCACCHI_PROFILE_PROCESS", "0").strip().lower()
    if value in {"all", "*"}:
        return True
    selected = {int(part.strip()) for part in value.split(",") if part.strip()}
    return jax.process_index() in selected


def _profile_log_dir() -> Path | None:
    profile_dir = os.environ.get("SCACCHI_PROFILE_DIR")
    if not profile_dir or not _profile_enabled_for_process():
        return None
    return Path(profile_dir).expanduser() / f"process_{jax.process_index():03d}"


def _block_until_ready(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def _weighted_metric_mean(values: Any, counts: Any) -> float:
    values_np = np.asarray(jax.device_get(values), dtype=np.float64)
    counts_np = np.asarray(jax.device_get(counts), dtype=np.float64)
    total = float(np.sum(counts_np))
    if total <= 0.0:
        return 0.0
    return float(np.sum(values_np * counts_np) / total)


def _pooled_metric_std(means: Any, stds: Any, counts: Any) -> float:
    means_np = np.asarray(jax.device_get(means), dtype=np.float64)
    stds_np = np.asarray(jax.device_get(stds), dtype=np.float64)
    counts_np = np.asarray(jax.device_get(counts), dtype=np.float64)
    total = float(np.sum(counts_np))
    if total <= 0.0:
        return 0.0
    mean = float(np.sum(means_np * counts_np) / total)
    second_moment = float(
        np.sum((np.square(stds_np) + np.square(means_np)) * counts_np)
        / total
    )
    return float(np.sqrt(max(second_moment - mean * mean, 0.0)))


def _count_fraction(numerator: Any, denominator: Any) -> float:
    numerator_total = float(
        np.sum(np.asarray(jax.device_get(numerator), dtype=np.float64))
    )
    denominator_total = float(
        np.sum(np.asarray(jax.device_get(denominator), dtype=np.float64))
    )
    if denominator_total <= 0.0:
        return 0.0
    return numerator_total / denominator_total


def _concentration_metrics_for_logging(train_metrics: Any) -> dict[str, float]:
    """Aggregate per-minibatch concentration diagnostics by target count."""

    result: dict[str, float] = {}
    for head, lower in (("V", "v"), ("Q", "q")):
        dir_count = getattr(train_metrics, f"{lower}_dirichlet_target_count")
        cat_count = getattr(train_metrics, f"{lower}_categorical_target_count")
        native_count = getattr(train_metrics, f"{lower}_native_target_count")
        alpha_mean = getattr(train_metrics, f"alpha_{head}_concentration")
        alpha_std = getattr(train_metrics, f"alpha_{head}_concentration_std")
        pred_dir_mean = getattr(
            train_metrics,
            f"alpha_{head}_dirichlet_concentration",
        )
        pred_dir_std = getattr(
            train_metrics,
            f"alpha_{head}_dirichlet_concentration_std",
        )
        target_dir_mean = getattr(train_metrics, f"beta_{head}_concentration")
        target_dir_std = getattr(
            train_metrics,
            f"beta_{head}_concentration_std",
        )
        pred_cat_mean = getattr(
            train_metrics,
            f"alpha_{head}_categorical_concentration",
        )

        result[f"train/alpha_{head}_concentration"] = _weighted_metric_mean(
            alpha_mean,
            native_count,
        )
        result[f"train/alpha_{head}_concentration_std"] = _pooled_metric_std(
            alpha_mean,
            alpha_std,
            native_count,
        )
        result[f"train/{head}_C_pred_mean_dir"] = _weighted_metric_mean(
            pred_dir_mean,
            dir_count,
        )
        result[f"train/{head}_C_pred_std_dir"] = _pooled_metric_std(
            pred_dir_mean,
            pred_dir_std,
            dir_count,
        )
        result[f"train/{head}_C_target_mean_dir"] = _weighted_metric_mean(
            target_dir_mean,
            dir_count,
        )
        result[f"train/{head}_C_target_std_dir"] = _pooled_metric_std(
            target_dir_mean,
            target_dir_std,
            dir_count,
        )
        result[f"train/{head}_C_log_mae_dir"] = _weighted_metric_mean(
            getattr(
                train_metrics,
                f"{lower}_dirichlet_log_concentration_mae",
            ),
            dir_count,
        )
        result[f"train/{head}_C_at_floor_fraction_dir"] = (
            _weighted_metric_mean(
                getattr(
                    train_metrics,
                    f"alpha_{head}_dirichlet_concentration_floor_fraction",
                ),
                dir_count,
            )
        )
        result[f"train/{head}_C_at_clip_fraction_dir"] = _weighted_metric_mean(
            getattr(
                train_metrics,
                f"alpha_{head}_dirichlet_concentration_clip_fraction",
            ),
            dir_count,
        )
        result[f"train/{head}_C_pred_mean_cat"] = _weighted_metric_mean(
            pred_cat_mean,
            cat_count,
        )
        result[f"train/{head}_C_at_clip_fraction_cat"] = (
            _weighted_metric_mean(
                getattr(
                    train_metrics,
                    f"alpha_{head}_categorical_concentration_clip_fraction",
                ),
                cat_count,
            )
        )
        result[f"data/{lower}_categorical_target_fraction"] = _count_fraction(
            cat_count,
            native_count,
        )
        result[f"data/{lower}_dirichlet_target_count"] = float(
            np.sum(np.asarray(jax.device_get(dir_count), dtype=np.float64))
        )
        result[f"data/{lower}_categorical_target_count"] = float(
            np.sum(np.asarray(jax.device_get(cat_count), dtype=np.float64))
        )
    return result


def _concentration_histograms_for_logging(
    train_metrics: Any,
) -> dict[str, PrecomputedHistogram]:
    """Pool minibatch bucket counts into matched prior/posterior histograms."""

    counts = np.asarray(
        jax.device_get(
            train_metrics.dirichlet_concentration_histogram_counts
        ),
        dtype=np.float64,
    )
    expected_tail = (
        len(CONCENTRATION_HISTOGRAM_SERIES),
        CONCENTRATION_HISTOGRAM_NUM_BINS,
    )
    if counts.ndim < 2 or counts.shape[-2:] != expected_tail:
        raise ValueError(
            "concentration histogram counts must end in "
            f"{expected_tail}; got {counts.shape}."
        )
    leading_axes = tuple(range(counts.ndim - 2))
    pooled = np.sum(counts, axis=leading_axes) if leading_axes else counts
    edges = np.asarray(CONCENTRATION_HISTOGRAM_BIN_EDGES, dtype=np.float64)

    histograms: dict[str, PrecomputedHistogram] = {}
    for index, series in enumerate(CONCENTRATION_HISTOGRAM_SERIES):
        head, role = series.split("_", maxsplit=1)
        histograms[f"train/{head}_C_{role}_hist_dir"] = PrecomputedHistogram(
            pooled[index],
            edges,
        )
    return histograms


def _load_eval_baseline(config: Config, env: pgx.Env, parallel=None):
    if config.eval.interval <= 0:
        return None
    if config.eval.baseline == EvalBaseline.none:
        raise ValueError("eval.baseline=none requires eval.interval=0.")
    if config.eval.baseline == EvalBaseline.checkpoint:
        checkpoint_path = (
            config.eval.checkpoint_path
            if config.eval.checkpoint_path is not None
            else f"checkpoints/{config.env.board_size}_solved"
        )
        return from_pretrained(checkpoint_path, env, rngs=nnx.Rngs(0))
    if config.eval.baseline == EvalBaseline.random:
        num_actions = env.num_actions

        def random_baseline_model(observation: jax.Array):
            logits = jnp.zeros(
                (*observation.shape[:1], num_actions),
                dtype=jnp.float32,
            )
            return constrain_batch_axis(logits, parallel, batch_axis=0)

        return random_baseline_model
    if config.eval.baseline == EvalBaseline.pgx:
        baseline_id = config.eval.baseline_id or f"{env.id}_v0"
        try:
            baseline_model = pgx.make_baseline_model(cast(Any, baseline_id))
        except AssertionError as exc:
            raise ValueError(
                f"PGX does not provide baseline model {baseline_id!r}. "
                "Use eval.baseline=none with eval.interval=0, or provide a "
                "checkpoint baseline."
            ) from exc
        baseline_model = _compact_pgx_baseline_model_for_env(baseline_model, env)
        _validate_pgx_eval_baseline(baseline_model, baseline_id, env)
        return baseline_model
    raise ValueError(f"unknown eval.baseline: {config.eval.baseline!r}")


def _compact_pgx_baseline_model_for_env(baseline_model: Any, env: pgx.Env):
    action_labels = getattr(env, "compact_action_labels", None)
    if action_labels is None:
        return baseline_model
    full_num_actions = int(getattr(env, "full_num_actions"))
    action_labels = jnp.asarray(action_labels, dtype=jnp.int32)

    def compact_model(observation: jax.Array):
        output = baseline_model(observation)
        if isinstance(output, tuple):
            logits, *rest = output
            if logits.shape[-1] != full_num_actions:
                raise ValueError(
                    f"baseline logits have {logits.shape[-1]} actions; "
                    f"expected {full_num_actions}"
                )
            return (jnp.take(logits, action_labels, axis=-1), *rest)
        if output.shape[-1] != full_num_actions:
            raise ValueError(
                f"baseline logits have {output.shape[-1]} actions; "
                f"expected {full_num_actions}"
            )
        return jnp.take(output, action_labels, axis=-1)

    return compact_model


def _validate_pgx_eval_baseline(
    baseline_model: Any,
    baseline_id: str,
    env: pgx.Env,
) -> None:
    observation = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    try:
        output = baseline_model(observation)
    except Exception as exc:
        raise ValueError(
            f"PGX baseline model {baseline_id!r} is incompatible with "
            f"env {env.id!r} observation_shape={env.observation_shape}."
        ) from exc

    logits = output[0] if isinstance(output, tuple) else output
    if tuple(logits.shape) != (1, env.num_actions):
        raise ValueError(
            f"PGX baseline model {baseline_id!r} returned logits shape "
            f"{tuple(logits.shape)} for env {env.id!r}; expected "
            f"{(1, env.num_actions)}."
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    config = load_config(cfg)
    initialize_distributed()
    report_jax_backend()
    parallel = make_batch_parallel(config)
    if jax.process_index() != 0:
        config.logging.wandb.enabled = False

    env = make_env(config.env.id, config.env.board_size)
    baseline_model = _load_eval_baseline(config, env, parallel)

    model = build_model(
        config,
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        rngs=nnx.Rngs(config.run.seed),
    )
    optimizer_transforms: list[optax.GradientTransformation] = []
    if config.training.grad_clip_norm is not None:
        optimizer_transforms.append(
            optax.clip_by_global_norm(config.training.grad_clip_norm)
        )
    learning_rate: float | optax.Schedule = config.training.learning_rate
    if (
        config.training.lr_decay_after_iters is not None
        and config.training.lr_decay_factor != 1.0
    ):
        rows_per_iter = max(
            1,
            config.selfplay.batch_size * config.selfplay.max_num_steps,
        )
        updates_per_iter = max(1, rows_per_iter // config.training.batch_size)
        if config.training.max_updates_per_iter is not None:
            updates_per_iter = min(
                updates_per_iter,
                config.training.max_updates_per_iter,
            )
        learning_rate = optax.piecewise_constant_schedule(
            init_value=config.training.learning_rate,
            boundaries_and_scales={
                config.training.lr_decay_after_iters
                * updates_per_iter: config.training.lr_decay_factor,
            },
        )
    optimizer_transforms.append(optax.adam(learning_rate=learning_rate))
    optimizer = nnx.Optimizer(model, optax.chain(*optimizer_transforms), wrt=nnx.Param)

    training_iteration = make_training_iteration(env, config, parallel=parallel)

    evaluate = (
        None
        if baseline_model is None
        else make_mcts_evaluate(env, config, baseline_model, parallel=parallel)
    )

    hours: float = 0.0
    frames: int = 0

    rng_key = jax.random.PRNGKey(config.run.seed)
    with build_logger(config) as logger:
        eval_avg_return_history: list[float] = []
        previous_eval_avg_return: float | None = None
        board_size = ("none" if config.env.board_size is None else str(config.env.board_size))
        ckpt_dir = (
            Path(get_original_cwd())
            / "checkpoints"
            / (
                f"{config.env.id}_bs{board_size}_{config.model.network}"
                f"_c{config.model.num_channels}_l{config.model.num_layers}"
                f"_seed{config.run.seed}"
            )
        ).resolve()
        if config.checkpointing.directory is not None:
            ckpt_dir = (
                Path(get_original_cwd()) / config.checkpointing.directory
            ).resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with build_checkpoint_manager(config, ckpt_dir) as ckpt_mgr:
            start_iter, rng_key, hours, frames = restore(ckpt_mgr, model, optimizer, rng_key)
            profile_log_dir = _profile_log_dir()
            profile_start_iter = _env_int(
                "SCACCHI_PROFILE_START_ITER",
                start_iter + 2,
                minimum=0,
            )
            profile_num_iters = _env_int(
                "SCACCHI_PROFILE_NUM_ITERS",
                3,
                minimum=1,
            )
            profile_stop_iter = profile_start_iter + profile_num_iters
            profile_create_perfetto = _env_flag("SCACCHI_PROFILE_PERFETTO")
            profile_trace_active = False
            if jax.process_index() == 0:
                print(
                    "Training loop starting: "
                    f"start_iter={start_iter}, "
                    f"max_num_iters={config.run.max_num_iters}, "
                    f"eval_interval={config.eval.interval}",
                    flush=True,
                )
            pbar = tqdm(range(start_iter, config.run.max_num_iters), desc="training", dynamic_ncols=True, total=config.run.max_num_iters, initial=start_iter, disable=jax.process_index() != 0)
            pbar.refresh()
            try:
                for iteration in pbar:
                    dict_to_log = {}
                    rng_key, eval_key, train_key = jax.random.split(rng_key, 3)
                    if evaluate is not None and config.eval.interval > 0 and (
                        iteration == config.run.max_num_iters - 1
                        or iteration % config.eval.interval == 0
                    ):
                        if jax.process_index() == 0:
                            print(f"Iteration {iteration}: evaluation starting", flush=True)
                        with parallel.mesh_context():
                            returns = evaluate(eval_key, model)
                        dict_to_log.update(returns_metrics("eval/vs_baseline", returns))
                        eval_avg_return = float(jax.device_get(returns.mean()))
                        eval_avg_return_history.append(eval_avg_return)
                        eval_window = eval_avg_return_history[-10:]
                        eval_mean_10 = float(np.mean(eval_window))
                        eval_std_10 = float(np.std(eval_window))
                        eval_delta = (
                            0.0
                            if previous_eval_avg_return is None
                            else abs(eval_avg_return - previous_eval_avg_return)
                        )
                        previous_eval_avg_return = eval_avg_return
                        dict_to_log.update(
                            {
                                "eval/vs_baseline/avg_R_rolling_mean_10": eval_mean_10,
                                "eval/vs_baseline/avg_R_rolling_std_10": eval_std_10,
                                "eval/vs_baseline/avg_R_step_delta_abs": eval_delta,
                            }
                        )

                    profile_this_iter = (
                        profile_log_dir is not None
                        and profile_start_iter <= iteration < profile_stop_iter
                    )
                    if profile_this_iter and not profile_trace_active:
                        assert profile_log_dir is not None
                        profile_log_dir.mkdir(parents=True, exist_ok=True)
                        print(
                            "Starting JAX profile trace at "
                            f"iteration {iteration}: {profile_log_dir}",
                            flush=True,
                        )
                        jax.profiler.start_trace(
                            str(profile_log_dir),
                            create_perfetto_trace=profile_create_perfetto,
                        )
                        profile_trace_active = True

                    st = time.perf_counter()
                    if profile_this_iter:
                        with jax.profiler.StepTraceAnnotation(
                            "train_iteration",
                            step_num=iteration,
                        ):
                            with parallel.mesh_context():
                                train_metrics = training_iteration(
                                    model,
                                    optimizer,
                                    train_key,
                                )
                                train_metrics = _block_until_ready(train_metrics)
                    else:
                        with parallel.mesh_context():
                            train_metrics = training_iteration(model, optimizer, train_key)
                            train_metrics = _block_until_ready(train_metrics)
                    frames_this_iter = (config.selfplay.batch_size * config.selfplay.max_num_steps)
                    frames += frames_this_iter

                    et = time.perf_counter()
                    iter_seconds = et - st
                    hours += iter_seconds / 3600
                    if profile_trace_active and iteration + 1 >= profile_stop_iter:
                        jax.profiler.stop_trace()
                        print(
                            "Stopped JAX profile trace at "
                            f"iteration {iteration}: {profile_log_dir}",
                            flush=True,
                        )
                        profile_trace_active = False

                    dict_to_log.update(
                        {
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
                            "data/pass_fraction": train_metrics.data_pass_fraction.mean().item(),
                            "data/terminations_per_row": train_metrics.data_terminations_per_row.mean().item(),
                            "data/psk_termination_fraction": train_metrics.data_psk_termination_fraction.mean().item(),
                            "train/iter_seconds": iter_seconds,
                            "train/frames_per_second": frames_this_iter
                            / max(iter_seconds, 1e-12),
                            "train/hours": hours,
                            "train/frames": frames,
                        }
                    )
                    dict_to_log.update(
                        _concentration_metrics_for_logging(train_metrics)
                    )
                    dict_to_log.update(
                        _concentration_histograms_for_logging(train_metrics)
                    )
                    logger.log(iteration, dict_to_log, pbar=pbar, prefix="", pbar_filter=r"loss|avg_R")
                    maybe_save(ckpt_mgr, iteration, model, optimizer, rng_key, config, hours, frames)
                    raw_snapshot_dir = os.environ.get("SCACCHI_RAW_SNAPSHOT_DIR")
                    if (
                        raw_snapshot_dir
                        and jax.process_index() == 0
                        and (
                            iteration % config.checkpointing.save_interval_steps == 0
                            or iteration == config.run.max_num_iters - 1
                        )
                    ):
                        # Plain pickle snapshot from process 0 only: orbax
                        # multihost saves deadlock on pod-local disks.
                        import pickle

                        snap_dir = Path(raw_snapshot_dir)
                        snap_dir.mkdir(parents=True, exist_ok=True)
                        # Pure nested dict of numpy arrays: portable across
                        # flax/python versions, unlike pickled nnx State.
                        snap_state = jax.device_get(
                            nnx.to_pure_dict(nnx.state(model))
                        )
                        snap_path = snap_dir / f"model_{iteration:06d}.pkl"
                        with open(snap_path, "wb") as f:
                            pickle.dump(snap_state, f)
            finally:
                if profile_trace_active:
                    jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
