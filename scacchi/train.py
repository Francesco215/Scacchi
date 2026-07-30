"""Hydra entry point for Scacchi training."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "scacchi"

if any(Path("/dev").glob("accel*")):
    os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import nnx
import hydra
from hydra.utils import get_original_cwd
import jax
from omegaconf import DictConfig
import optax
from tqdm import tqdm

from .checkpoint import build_checkpoint_manager, maybe_save, restore
from .distributed import initialize_distributed, make_batch_parallel
from .envs import make_env
from .evaluations import evaluation_metrics, load_eval_baseline, make_mcts_evaluate
from .logger import Metric, build_logger, returns_metrics, training_metrics
from .network import build_model
from .pipeline import make_training_iteration
from .types import Config, load_config


def report_jax_backend() -> None:
    backend = jax.default_backend()
    if jax.process_index() == 0:
        print(f"JAX backend: {backend}")
        print(f"JAX process count: {jax.process_count()}")
        print(f"JAX global devices: {jax.devices()}")
    print(f"JAX process {jax.process_index()} local devices: {jax.local_devices()}")
    if os.environ.get("SCACCHI_ALLOW_CPU") != "1" and backend not in {"gpu", "tpu"}:
        raise RuntimeError("JAX is not using a GPU or TPU backend. Set SCACCHI_ALLOW_CPU=1 only for intentional CPU runs.")


def _block_until_ready(value: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, value)


def _build_optimizer(model: nnx.Module, config: Config) -> nnx.Optimizer:
    transforms: list[optax.GradientTransformation] = []
    if config.training.grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(config.training.grad_clip_norm))

    learning_rate: float | optax.Schedule = config.training.learning_rate
    if config.training.lr_decay_after_iters is not None and config.training.lr_decay_factor != 1.0:
        rows_per_iteration = max(1, config.selfplay.batch_size * config.selfplay.max_num_steps)
        updates_per_iteration = max(1, rows_per_iteration // config.training.batch_size)
        if config.training.max_updates_per_iter is not None:
            updates_per_iteration = min(updates_per_iteration, config.training.max_updates_per_iter)
        boundary = config.training.lr_decay_after_iters * updates_per_iteration
        learning_rate = optax.piecewise_constant_schedule(config.training.learning_rate, {boundary: config.training.lr_decay_factor})

    transforms.append(optax.adam(learning_rate))
    return nnx.Optimizer(model, optax.chain(*transforms), wrt=nnx.Param)


def _checkpoint_directory(config: Config) -> Path:
    if config.checkpointing.directory is not None:
        return (Path(get_original_cwd()) / config.checkpointing.directory).resolve()
    board_size = "none" if config.env.board_size is None else str(config.env.board_size)
    run_name = f"{config.env.id}_bs{board_size}_{config.model.network}_c{config.model.num_channels}_l{config.model.num_layers}_seed{config.run.seed}"
    return (Path(get_original_cwd()) / "checkpoints" / run_name).resolve()


def _should_evaluate(iteration: int, config: Config) -> bool:
    return config.eval.interval > 0 and (iteration % config.eval.interval == 0 or iteration == config.run.max_num_iters - 1)


def _evaluate(iteration: int, rng_key: jax.Array, model: nnx.Module, evaluate, parallel, history: list[float], config: Config) -> dict[str, Metric]:
    if evaluate is None or not _should_evaluate(iteration, config):
        return {}
    with parallel.mesh_context():
        returns = evaluate(rng_key, model)
    metrics: dict[str, Metric] = {}
    metrics.update(returns_metrics("eval/vs_baseline", returns))
    metrics.update(evaluation_metrics(returns, history))
    return metrics


def _train_iteration(rng_key: jax.Array, model: nnx.Module, optimizer: nnx.Optimizer, training_iteration, parallel):
    started = time.perf_counter()
    with parallel.mesh_context():
        metrics = _block_until_ready(training_iteration(model, optimizer, rng_key))
    return metrics, time.perf_counter() - started


def _progress_bar(start_iteration: int, config: Config):
    return tqdm(range(start_iteration, config.run.max_num_iters), desc="training", dynamic_ncols=True, total=config.run.max_num_iters, initial=start_iteration, disable=jax.process_index() != 0)


def _run_loop(config: Config, model: nnx.Module, optimizer: nnx.Optimizer, training_iteration, evaluate, parallel, checkpoint_directory: Path) -> None:
    rng_key = jax.random.PRNGKey(config.run.seed)
    evaluation_history: list[float] = []

    with build_logger(config) as logger, build_checkpoint_manager(config, checkpoint_directory) as checkpoint_manager:
        start_iteration, rng_key, hours, frames = restore(checkpoint_manager, model, optimizer, rng_key)
        if jax.process_index() == 0:
            print(f"Training loop starting: start_iter={start_iteration}, max_num_iters={config.run.max_num_iters}, eval_interval={config.eval.interval}", flush=True)
        progress = _progress_bar(start_iteration, config)
        progress.refresh()

        for iteration in progress:
            rng_key, eval_key, train_key = jax.random.split(rng_key, 3)
            metrics = _evaluate(iteration, eval_key, model, evaluate, parallel, evaluation_history, config)
            if "eval/vs_baseline/win_rate" in metrics:
                progress.set_postfix(win_rate=f"{metrics['eval/vs_baseline/win_rate']:.1%}")
            train_result, seconds = _train_iteration(train_key, model, optimizer, training_iteration, parallel)
            frames_this_iteration = config.selfplay.batch_size * config.selfplay.max_num_steps
            frames += frames_this_iteration
            hours += seconds / 3600
            metrics.update(training_metrics(train_result, seconds=seconds, hours=hours, frames=frames, frames_this_iteration=frames_this_iteration))
            logger.log(iteration, metrics, prefix="")
            maybe_save(checkpoint_manager, iteration, model, optimizer, rng_key, config, hours, frames)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    config = load_config(cfg)
    initialize_distributed()
    report_jax_backend()
    parallel = make_batch_parallel(config)
    if jax.process_index() != 0:
        config.logging.wandb.enabled = False

    env = make_env(config.env.id, config.env.board_size)
    model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(config.run.seed))
    optimizer = _build_optimizer(model, config)
    training_iteration = make_training_iteration(env, config, parallel=parallel)
    baseline = load_eval_baseline(config, env, parallel)
    evaluate = None if baseline is None else make_mcts_evaluate(env, config, baseline, parallel=parallel)
    checkpoint_directory = _checkpoint_directory(config)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    _run_loop(config, model, optimizer, training_iteration, evaluate, parallel, checkpoint_directory)


if __name__ == "__main__":
    main()
