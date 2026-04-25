"""Hydra entrypoint for Gumbel AlphaZero training."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hydra
import jax
import jax.numpy as jnp
import pgx
from omegaconf import DictConfig, OmegaConf

from scacchi.checkpoint import save_checkpoint
from scacchi.models import make_model
from scacchi.optim import make_optimizer
from scacchi.runtime import create_mesh, validate_batch_size
from scacchi.selfplay import compute_training_batch, run_selfplay
from scacchi.training import init_train_state, make_train_step, shuffle_batch, take_batch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    env = pgx.make(cast(pgx.EnvId, str(cfg.env.id)))
    if tuple(env.observation_shape) != (8, 8, 119) or env.num_actions != 4672:
        raise ValueError("This entrypoint is configured for PGX full chess.")

    mesh = create_mesh(cfg.runtime)
    validate_batch_size(int(cfg.train.selfplay_batch_size), mesh)
    validate_batch_size(int(cfg.train.batch_size), mesh)

    model = make_model(
        cfg.model,
        observation_shape=tuple(env.observation_shape),
        num_actions=env.num_actions,
        seed=int(cfg.train.seed),
    )
    tx = make_optimizer(cfg.optimizer)
    graphdef, train_state = init_train_state(model, tx)
    train_step = make_train_step(graphdef, tx)
    selfplay_fn = jax.jit(
        lambda params, key: run_selfplay(
            env=env,
            graphdef=graphdef,
            params=params,
            rng_key=key,
            batch_size=int(cfg.train.selfplay_batch_size),
            max_num_steps=int(cfg.train.max_num_steps),
            num_simulations=int(cfg.search.num_simulations),
            max_num_considered_actions=int(cfg.search.max_num_considered_actions),
            max_depth=cfg.search.max_depth,
            gumbel_scale=float(cfg.search.train_gumbel_scale),
        )
    )

    rng_key = jax.random.key(int(cfg.train.seed))
    print(OmegaConf.to_yaml(cfg))
    for iteration in range(int(cfg.train.num_iters)):
        rng_key, selfplay_key, shuffle_key = jax.random.split(rng_key, 3)
        data = selfplay_fn(train_state.params, selfplay_key)
        batch = compute_training_batch(data)
        batch = take_batch(shuffle_batch(batch, shuffle_key), int(cfg.train.batch_size))
        train_state, metrics = train_step(train_state, batch)
        if iteration % int(cfg.train.log_interval) == 0:
            metrics_host = jax.device_get(metrics)
            print(
                {
                    "iteration": iteration,
                    "loss": float(metrics_host.loss),
                    "policy_loss": float(metrics_host.policy_loss),
                    "value_loss": float(metrics_host.value_loss),
                    "samples": int(batch.observation.shape[0]),
                    "mean_value_target": float(jnp.mean(batch.value_target)),
                }
            )
        if iteration % int(cfg.train.checkpoint_interval) == 0:
            save_checkpoint(
                Path("checkpoints") / f"{iteration:06d}.ckpt",
                cfg=OmegaConf.to_container(cfg, resolve=True),
                train_state=train_state,
                rng_key=rng_key,
                step=iteration,
            )


if __name__ == "__main__":
    main()
