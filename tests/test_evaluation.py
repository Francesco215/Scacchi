from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pgx
from omegaconf import OmegaConf

from scacchi.config import EvalConfig
from scacchi.evaluation import (
    baseline_log,
    evaluate_baseline,
)
from scacchi.models import AlphaZeroResNet
from scacchi.optim import make_optimizer
from scacchi.training import init_optimizer


def _small_model_cfg():
    return OmegaConf.create(
        {
            "name": "resnet",
            "channels": 4,
            "blocks": 1,
            "policy_channels": 1,
            "value_channels": 1,
            "value_hidden": 4,
        }
    )


def _small_state():
    env = pgx.make("chess")
    model = AlphaZeroResNet(
        _small_model_cfg(),
        observation_shape=tuple(env.observation_shape),
        num_actions=env.num_actions,
        seed=0,
    )
    tx = make_optimizer(
        OmegaConf.create({"name": "adamw", "learning_rate": 1.0e-3, "weight_decay": 0.0})
    )
    init_optimizer(model, tx)
    return env, model


def test_tiny_baseline_match_batch():
    env, model = _small_state()
    cfg = EvalConfig(
        enabled=True,
        interval=1,
        batch_size=2,
        max_num_steps=2,
    )

    def baseline(observation):
        return (
            jnp.zeros((*observation.shape[:-3], env.num_actions), dtype=jnp.float32),
            jnp.zeros(observation.shape[:-3], dtype=jnp.float32),
        )

    returns = evaluate_baseline(
        env=env,
        model=model,
        baseline=baseline,
        rng_key=jax.random.key(4),
        cfg=cfg,
    )
    log = baseline_log("eval/vs_baseline", returns)

    assert returns.shape == (2,)
    assert jnp.isfinite(returns).all()
    assert "eval/vs_baseline/avg_R" in log


def test_no_pmap_in_runtime_sources():
    for path in Path("scacchi").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "pmap" not in source
        assert "haiku" not in source
