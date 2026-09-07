"""Hydra entry point for Scacchi training."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

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
from tqdm import tqdm

from .checkpoint import build_checkpoint_manager, maybe_save, resolve_checkpoint_directory, restore
from .distributed import initialize_distributed, make_batch_parallel
from .envs import make_env
from .evaluations import evaluation_metrics, load_eval_baseline, make_mcts_evaluate
from .logger import Metric, build_logger, returns_metrics, training_metrics
from .network import build_model
from .pipeline import build_optimizer, make_training_iteration
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


def _run_loop(config: Config, model: nnx.Module, optimizer: nnx.Optimizer, training_iteration, evaluate, parallel, checkpoint_directory: Path) -> None:
    rng_key = jax.random.PRNGKey(config.run.seed)
    evaluation_history: list[float] = []

    with build_logger(config) as logger, build_checkpoint_manager(config, checkpoint_directory) as checkpoint_manager:
        start_iteration, rng_key, hours, frames = restore(checkpoint_manager, model, optimizer, rng_key)
        if jax.process_index() == 0:
            print(f"Training loop starting: start_iter={start_iteration}, max_num_iters={config.run.max_num_iters}, eval_interval={config.eval.interval}", flush=True)
        progress = tqdm(range(start_iteration, config.run.max_num_iters), desc="training", dynamic_ncols=True, total=config.run.max_num_iters, initial=start_iteration, disable=jax.process_index() != 0)
        progress.refresh()

        for iteration in progress:
            rng_key, eval_key, train_key = jax.random.split(rng_key, 3)
            metrics: dict[str, Metric] = {}
            if evaluate is not None and config.eval.interval > 0 and (
                iteration % config.eval.interval == 0 or iteration == config.run.max_num_iters - 1
            ):
                with parallel.mesh_context():
                    returns = evaluate(eval_key, model)
                metrics.update(returns_metrics("eval/vs_baseline", returns))
                metrics.update(evaluation_metrics(returns, evaluation_history))
            if "eval/vs_baseline/win_rate" in metrics:
                progress.set_postfix(win_rate=f"{metrics['eval/vs_baseline/win_rate']:.1%}")

            started = time.perf_counter()
            with parallel.mesh_context():
                train_result = jax.block_until_ready(training_iteration(model, optimizer, train_key))
            seconds = time.perf_counter() - started
            frames_this_iteration = config.selfplay.batch_size * config.selfplay.max_num_steps
            frames += frames_this_iteration
            hours += seconds / 3600
            metrics.update(training_metrics(train_result, seconds=seconds, hours=hours, frames=frames, frames_this_iteration=frames_this_iteration))
            logger.log(iteration, metrics, prefix="")
            maybe_save(
                checkpoint_manager,
                iteration,
                model,
                optimizer,
                rng_key,
                config,
                hours,
                frames,
                force=iteration == config.run.max_num_iters - 1,
            )

        checkpoint_manager.wait_until_finished()
        if jax.process_index() == 0 and start_iteration < config.run.max_num_iters:
            print(
                f"Saved final checkpoint at step {config.run.max_num_iters - 1} "
                f"to {checkpoint_directory}",
                flush=True,
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
    model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(config.run.seed))
    optimizer = build_optimizer(model, config)
    training_iteration = make_training_iteration(env, config, parallel=parallel)
    baseline = load_eval_baseline(config, env, parallel)
    evaluate = None if baseline is None else make_mcts_evaluate(env, config, baseline, parallel=parallel)
    checkpoint_directory = resolve_checkpoint_directory(config, Path(get_original_cwd()))
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    _run_loop(config, model, optimizer, training_iteration, evaluate, parallel, checkpoint_directory)


if __name__ == "__main__":
    main()
