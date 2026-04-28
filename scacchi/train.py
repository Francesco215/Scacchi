"""Hydra entrypoint for Gumbel AlphaZero training."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, cast

import hydra
import jax
import jax.numpy as jnp
import pgx
from absl import logging as absl_logging
from flax import nnx
from omegaconf import DictConfig, OmegaConf

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
from scacchi.models import AlphaZeroResNet
from scacchi.optim import make_optimizer
from scacchi.runtime import create_mesh, validate_batch_size
from scacchi.selfplay import compute_training_batch, run_selfplay
from scacchi.training import init_optimizer, make_minibatches, make_train_step


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
    train_step = make_train_step()
    selfplay_fn = nnx.jit(
        lambda model, key: run_selfplay(env=env, model=model, rng_key=key, cfg=cfg.train)
    )
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
    wandb_run = None
    if cfg.logging.wandb_enabled:
        import wandb

        wandb_run = wandb.init(project=cfg.logging.wandb_project, config=cfg_dict)
    with build_checkpoint_manager(
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

        for iteration in range(start_iteration, cfg.train.num_iters):
            rng_key, selfplay_key, shuffle_key = jax.random.split(rng_key, 3)
            start_time = time.time()
            data = selfplay_fn(model, selfplay_key)
            flat_batch = compute_training_batch(data)
            minibatches = make_minibatches(flat_batch, cfg.train.batch_size, shuffle_key)
            num_updates = int(minibatches.observation.shape[0])
            metrics = []
            for i in range(num_updates):
                batch = jax.tree_util.tree_map(lambda x: x[i], minibatches)
                metrics.append(train_step(model, optimizer, batch))
            metrics = jax.tree_util.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs)), *metrics
            )
            frames += int(data.observation.shape[0] * data.observation.shape[1])
            hours += (time.time() - start_time) / 3600

            metrics_host = jax.device_get(metrics)
            value_mask = minibatches.value_mask.reshape((-1,)).astype(jnp.float32)
            value_target = minibatches.value_target.reshape((-1,))
            log: dict[str, float | int] = {
                "iteration": iteration,
                "loss": float(metrics_host.loss),
                "policy_loss": float(metrics_host.policy_loss),
                "value_loss": float(metrics_host.value_loss),
                "samples": int(value_mask.shape[0]),
                "num_updates": num_updates,
                "frames": frames,
                "hours": hours,
                "value_mask_frac": float(jnp.mean(value_mask)),
                "mean_value_target": float(jnp.mean(value_target)),
                "mean_abs_value_target": float(
                    jnp.sum(jnp.abs(value_target) * value_mask)
                    / jnp.maximum(jnp.sum(value_mask), 1.0)
                ),
            }
            eval_ran = cfg.eval.enabled and iteration % cfg.eval.interval == 0
            if eval_ran:
                rng_key, baseline_key = jax.random.split(rng_key)
                if baseline_eval_fn is not None:
                    returns = jax.device_get(baseline_eval_fn(model, baseline_key))
                    log.update(baseline_log("eval/vs_baseline", returns))
            if iteration % cfg.train.log_interval == 0 or eval_ran:
                print(log)
            if wandb_run is not None:
                wandb_run.log(log)
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
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
