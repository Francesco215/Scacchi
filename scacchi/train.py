"""Hydra entrypoint for Gumbel AlphaZero training."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, cast

import hydra
import jax
import pgx
from absl import logging as absl_logging
from flax import nnx
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scacchi.checkpoint import (
    build_checkpoint_manager,
    maybe_save_checkpoint,
    restore_checkpoint,
)
from scacchi.config import config_from_dict_config
from scacchi.evaluation import (
    baseline_log,
    evaluate_baseline,
)
from scacchi.logger import build_logger
from scacchi.models import AlphaZeroResNet
from scacchi.optim import make_optimizer
from scacchi.runtime import create_mesh, validate_batch_size
from scacchi.training import init_optimizer, make_iteration_step


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    cfg = config_from_dict_config(raw_cfg)
    cfg_dict = asdict(cfg)
    absl_logging.set_verbosity(absl_logging.WARNING)
    env = pgx.make(cast(pgx.EnvId, cfg.env.id))
    observation_shape = tuple(env.observation_shape)
    assert len(observation_shape) == 3, (
        f"Expected a spatial PGX observation with rank 3, got {observation_shape}."
    )

    mesh = create_mesh(cfg.runtime)
    validate_batch_size("train.selfplay_batch_size", cfg.train.selfplay_batch_size, mesh)
    validate_batch_size("train.batch_size", cfg.train.batch_size, mesh)
    if cfg.eval.enabled:
        validate_batch_size("eval.batch_size", cfg.eval.batch_size, mesh)

    model = AlphaZeroResNet(
        cfg.model,
        observation_shape=observation_shape,
        num_actions=env.num_actions,
        seed=cfg.train.seed,
    )
    tx = make_optimizer(cfg.optimizer)
    optimizer = init_optimizer(model, tx)
    iteration_step = make_iteration_step(env, cfg.train)
    baseline_id = f"{cfg.env.id}_v0"
    baseline = None
    if cfg.eval.enabled:
        try:
            baseline = pgx.make_baseline_model(cast(Any, baseline_id))
        except AssertionError:
            print({"eval/baseline_id": baseline_id, "eval/baseline_available": 0})
    baseline_eval_fn = (
        None
        if baseline is None
        else nnx.jit(
            lambda candidate_model, key: evaluate_baseline(
                env=env,
                model=candidate_model,
                baseline=baseline,
                rng_key=key,
                cfg=cfg.eval,
            )
        )
    )
    rng_key = jax.random.key(cfg.train.seed)
    print(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))
    with build_logger(
        cfg.logging,
        log_every=cfg.train.log_interval,
        max_steps=cfg.train.num_iters,
        config=cfg_dict,
    ) as logger, build_checkpoint_manager(
        cfg.checkpoint, cfg.checkpoint.dir, max_steps=cfg.train.num_iters
    ) as checkpoint_manager:
        start_iteration = 0
        frames = 0
        hours = 0.0
        if cfg.checkpoint.resume:
            restored = restore_checkpoint(
                checkpoint_manager, model=model, optimizer=optimizer, rng_key=rng_key
            )
            start_iteration = restored.start_step
            rng_key = restored.rng_key
            frames = int(restored.meta.get("metadata", {}).get("frames", frames))
            hours = float(restored.meta.get("metadata", {}).get("hours", hours))

        with tqdm(total=cfg.train.num_iters, initial=start_iteration, desc="training") as pbar:
            for iteration in range(start_iteration, cfg.train.num_iters):
                rng_key, train_key = jax.random.split(rng_key)
                start_time = time.time()
                metrics = iteration_step(model, optimizer, train_key)
                frames += cfg.train.max_num_steps * cfg.train.selfplay_batch_size
                hours += (time.time() - start_time) / 3600

                metrics_host = jax.device_get(metrics)
                logger.log(iteration, metrics_host, pbar=pbar, prefix="train/", pbar_filter=r"loss", frames=frames, hours=hours)
                pbar.update(1)

                eval_ran = cfg.eval.enabled and iteration % cfg.eval.interval == 0
                if eval_ran:
                    rng_key, baseline_key = jax.random.split(rng_key)
                    if baseline_eval_fn is not None:
                        returns = jax.device_get(baseline_eval_fn(model, baseline_key))
                        logger.log_metrics(iteration, baseline_log("eval/vs_baseline", returns), prefix="")

                maybe_save_checkpoint(
                    checkpoint_manager,
                    iteration,
                    cfg=cfg_dict,
                    model=model,
                    optimizer=optimizer,
                    rng_key=rng_key,
                    metadata={
                        "baseline_id": baseline_id,
                        "baseline_available": baseline_eval_fn is not None,
                        "frames": frames,
                        "hours": hours,
                    },
                )
            checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
