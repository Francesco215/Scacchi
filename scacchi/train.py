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

from .distributed import initialize_distributed, make_batch_parallel

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
from .logger import build_logger, returns_metrics
from .network import build_model
from .pipeline import make_training_iteration
from .types import Config, EvalBaseline, RngSplitMode, load_config


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


def _load_eval_baseline(config: Config, env: pgx.Env):
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
        _validate_pgx_eval_baseline(baseline_model, baseline_id, env)
        return baseline_model
    raise ValueError(f"unknown eval.baseline: {config.eval.baseline!r}")


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

    with parallel.mesh_context():
        env = make_env(config.env.id, config.env.board_size)
        baseline_model = _load_eval_baseline(config, env)

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
                                train_metrics = training_iteration(model, optimizer, train_key)
                                train_metrics = _block_until_ready(train_metrics)
                        else:
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
                                "train/alpha_V_concentration": train_metrics.alpha_V_concentration.mean().item(),
                                "train/alpha_Q_concentration": train_metrics.alpha_Q_concentration.mean().item(),
                                "train/q_evidence_mass_mean": train_metrics.q_evidence_mass_mean.mean().item(),
                                "train/q_loss_weight_mean": train_metrics.q_loss_weight_mean.mean().item(),
                                "search/path_depth_mean": train_metrics.search_path_depth_mean.mean().item(),
                                "search/path_depth_p50": train_metrics.search_path_depth_p50.mean().item(),
                                "search/path_depth_p90": train_metrics.search_path_depth_p90.mean().item(),
                                "search/path_depth_max": train_metrics.search_path_depth_max.mean().item(),
                                "search/expanded_nodes": train_metrics.search_expanded_nodes.mean().item(),
                                "search/terminal_fraction": train_metrics.search_terminal_fraction.mean().item(),
                                "search/root_policy_entropy": train_metrics.search_root_policy_entropy.mean().item(),
                                "search/root_gamma": train_metrics.search_root_gamma.mean().item(),
                                "search/root_downstream_eval_count": train_metrics.search_root_downstream_eval_count.mean().item(),
                                "search/root_q_concentration": train_metrics.search_root_q_concentration.mean().item(),
                                "train/iter_seconds": iter_seconds,
                                "train/frames_per_second": frames_this_iter
                                / max(iter_seconds, 1e-12),
                                "train/hours": hours,
                                "train/frames": frames,
                            }
                        )
                        logger.log(iteration, dict_to_log, pbar=pbar, prefix="", pbar_filter=r"loss|avg_R")
                        maybe_save(ckpt_mgr, iteration, model, optimizer, rng_key, config, hours, frames)
                finally:
                    if profile_trace_active:
                        jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
