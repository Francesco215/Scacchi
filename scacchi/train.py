"""Hydra entrypoint for Gumbel AlphaZero training."""

from __future__ import annotations

from typing import cast

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
from scacchi.evaluation import (
    Anchor,
    add_anchor,
    anchor_summaries,
    append_eval_history,
    evaluate_vs_anchors,
    play_match_batch,
    report_to_log_dict,
)
from scacchi.models import AlphaZeroResNet
from scacchi.optim import make_optimizer
from scacchi.runtime import create_mesh, validate_batch_size
from scacchi.selfplay import compute_training_batch, run_selfplay
from scacchi.training import init_optimizer, make_train_step, shuffle_batch, take_batch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    absl_logging.set_verbosity(absl_logging.WARNING)
    env = pgx.make(cast(pgx.EnvId, str(cfg.env.id)))
    if tuple(env.observation_shape) != (8, 8, 119) or env.num_actions != 4672:
        raise ValueError("This entrypoint is configured for PGX full chess.")

    mesh = create_mesh(cfg.runtime)
    validate_batch_size(int(cfg.train.selfplay_batch_size), mesh)
    validate_batch_size(int(cfg.train.batch_size), mesh)
    eval_enabled = bool(cfg.eval.enabled)
    if eval_enabled:
        eval_batch_size = int(cfg.eval.batch_size)
        validate_batch_size(eval_batch_size, mesh)
        if eval_batch_size % 2 != 0:
            raise ValueError("eval.batch_size must be even for color-balanced matches.")
        if int(cfg.eval.interval) <= 0:
            raise ValueError("eval.interval must be positive.")
        if int(cfg.eval.anchor_interval) <= 0:
            raise ValueError("eval.anchor_interval must be positive.")
        if int(cfg.eval.max_anchors) < 1:
            raise ValueError("eval.max_anchors must be at least one.")

    model = AlphaZeroResNet(
        cfg.model,
        observation_shape=tuple(env.observation_shape),
        num_actions=env.num_actions,
        seed=int(cfg.train.seed),
    )
    tx = make_optimizer(cfg.optimizer)
    optimizer = init_optimizer(model, tx)
    train_step = make_train_step()
    selfplay_fn = nnx.jit(
        lambda model, key: run_selfplay(
            env=env,
            model=model,
            rng_key=key,
            batch_size=int(cfg.train.selfplay_batch_size),
            max_num_steps=int(cfg.train.max_num_steps),
            num_simulations=int(cfg.search.num_simulations),
            max_num_considered_actions=int(cfg.search.max_num_considered_actions),
            max_depth=cfg.search.max_depth,
            gumbel_scale=float(cfg.search.train_gumbel_scale),
        )
    )
    eval_fn = nnx.jit(
        lambda candidate_model, anchor_model, key: play_match_batch(
            env=env,
            candidate_model=candidate_model,
            anchor_model=anchor_model,
            rng_key=key,
            batch_size=int(cfg.eval.batch_size),
            max_num_steps=int(cfg.eval.max_num_steps),
            num_simulations=int(cfg.eval.num_simulations),
            max_num_considered_actions=int(cfg.eval.max_num_considered_actions),
            max_depth=cfg.eval.max_depth,
            gumbel_scale=float(cfg.eval.gumbel_scale),
        )
    )
    rng_key = jax.random.key(int(cfg.train.seed))
    print(OmegaConf.to_yaml(cfg))
    with build_checkpoint_manager(
        cfg.checkpoint,
        cfg.checkpoint.dir,
        max_steps=int(cfg.train.num_iters),
    ) as checkpoint_manager:
        start_iteration = 0
        current_elo = float(cfg.eval.initial_anchor_elo)
        if bool(cfg.checkpoint.resume):
            restored = restore_checkpoint(
                checkpoint_manager,
                model=model,
                optimizer=optimizer,
                rng_key=rng_key,
            )
            start_iteration = restored.start_step
            rng_key = restored.rng_key
            current_elo = float(restored.meta.get("metadata", {}).get("elo", current_elo))

        anchors: tuple[Anchor, ...] = ()
        if eval_enabled:
            anchors = (
                Anchor(
                    name="initial" if start_iteration == 0 else "resume",
                    model=nnx.clone(model),
                    elo=current_elo,
                    iteration=max(start_iteration - 1, 0),
                ),
            )

        for iteration in range(start_iteration, int(cfg.train.num_iters)):
            rng_key, selfplay_key, shuffle_key = jax.random.split(rng_key, 3)
            data = selfplay_fn(model, selfplay_key)
            batch = compute_training_batch(data)
            batch = take_batch(shuffle_batch(batch, shuffle_key), int(cfg.train.batch_size))
            metrics = train_step(model, optimizer, batch)
            metrics_host = jax.device_get(metrics)
            log: dict[str, float | int] = {
                "iteration": iteration,
                "loss": float(metrics_host.loss),
                "policy_loss": float(metrics_host.policy_loss),
                "value_loss": float(metrics_host.value_loss),
                "samples": int(batch.observation.shape[0]),
                "mean_value_target": float(jnp.mean(batch.value_target)),
            }
            eval_ran = eval_enabled and iteration % int(cfg.eval.interval) == 0
            if eval_ran:
                rng_key, eval_key = jax.random.split(rng_key)
                report = evaluate_vs_anchors(
                    eval_fn=eval_fn,
                    candidate_model=model,
                    anchors=anchors,
                    rng_key=eval_key,
                    iteration=iteration,
                )
                current_elo = report.elo
                log.update(report_to_log_dict(report))
                append_eval_history("eval_history.jsonl", report)
                if iteration > 0 and iteration % int(cfg.eval.anchor_interval) == 0:
                    anchors = add_anchor(
                        anchors,
                        model=model,
                        elo=current_elo,
                        iteration=iteration,
                        max_anchors=int(cfg.eval.max_anchors),
                    )
            if iteration % int(cfg.train.log_interval) == 0 or eval_ran:
                print(log)
            maybe_save_checkpoint(
                checkpoint_manager,
                iteration,
                cfg=OmegaConf.to_container(cfg, resolve=True),
                model=model,
                optimizer=optimizer,
                rng_key=rng_key,
                metadata={
                    "elo": current_elo,
                    "anchors": anchor_summaries(anchors),
                },
            )
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
