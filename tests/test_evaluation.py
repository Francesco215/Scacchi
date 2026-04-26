from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pgx
from omegaconf import OmegaConf

from scacchi.evaluation import (
    Anchor,
    add_anchor,
    append_eval_history,
    evaluate_vs_anchors,
    play_match_batch,
    report_to_json,
    score_to_elo,
)
from scacchi.models import make_model
from scacchi.optim import make_optimizer
from scacchi.training import init_train_state


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
    model = make_model(
        _small_model_cfg(),
        observation_shape=tuple(env.observation_shape),
        num_actions=env.num_actions,
        seed=0,
    )
    tx = make_optimizer(
        OmegaConf.create({"name": "adamw", "learning_rate": 1.0e-3, "weight_decay": 0.0})
    )
    graphdef, train_state = init_train_state(model, tx)
    return env, graphdef, train_state


def test_score_to_elo_is_finite_and_ordered():
    assert score_to_elo(0.5, 0.0, 10) == 0.0
    assert score_to_elo(0.75, 0.0, 10) > 0.0
    assert score_to_elo(0.25, 0.0, 10) < 0.0
    assert jnp.isfinite(jnp.asarray(score_to_elo(1.0, 0.0, 2)))
    assert jnp.isfinite(jnp.asarray(score_to_elo(0.0, 0.0, 2)))


def test_tiny_match_batch_and_anchor_report(tmp_path: Path):
    env, graphdef, train_state = _small_state()
    eval_fn = jax.jit(
        lambda candidate_params, anchor_params, key: play_match_batch(
            env=env,
            graphdef=graphdef,
            candidate_params=candidate_params,
            anchor_params=anchor_params,
            rng_key=key,
            batch_size=2,
            max_num_steps=2,
            num_simulations=2,
            max_num_considered_actions=4,
            max_depth=2,
            gumbel_scale=0.0,
        )
    )
    anchors = (
        Anchor(name="initial", params=train_state.params, elo=0.0, iteration=0),
    )
    stats = eval_fn(train_state.params, train_state.params, jax.random.key(1))
    assert int(stats.games) == 2
    assert int(stats.white_games) == 1
    assert int(stats.black_games) == 1
    assert int(stats.wins + stats.draws + stats.losses) == 2
    assert jnp.isfinite(stats.score)

    report = evaluate_vs_anchors(
        eval_fn=eval_fn,
        candidate_params=train_state.params,
        anchors=anchors,
        rng_key=jax.random.key(2),
        iteration=3,
    )
    assert report.iteration == 3
    assert report.games == 2
    assert len(report.anchors) == 1
    assert jnp.isfinite(jnp.asarray(report.elo))

    path = tmp_path / "eval_history.jsonl"
    append_eval_history(path, report)
    assert path.read_text(encoding="utf-8").strip()
    assert report_to_json(report)["iteration"] == 3

    updated = add_anchor(
        anchors,
        params=train_state.params,
        elo=report.elo,
        iteration=4,
        max_anchors=1,
    )
    assert len(updated) == 1
    assert updated[0].name == "initial"


def test_no_pmap_in_runtime_sources():
    for path in Path("scacchi").rglob("*.py"):
        assert "pmap" not in path.read_text(encoding="utf-8")
