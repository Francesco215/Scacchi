"""Evaluation matches and relative Elo tracking."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import pgx
from flax import nnx
from jaxtyping import Array, Float, PRNGKeyArray

from scacchi.search import run_search


class MatchStats(NamedTuple):
    games: Float[Array, ""]
    wins: Float[Array, ""]
    draws: Float[Array, ""]
    losses: Float[Array, ""]
    score: Float[Array, ""]
    white_games: Float[Array, ""]
    black_games: Float[Array, ""]


@dataclass(frozen=True)
class Anchor:
    name: str
    model: nnx.Module
    elo: float
    iteration: int


@dataclass(frozen=True)
class AnchorResult:
    name: str
    iteration: int
    anchor_elo: float
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    implied_elo: float
    elo_delta: float


@dataclass(frozen=True)
class EloReport:
    iteration: int
    elo: float
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    anchors: tuple[AnchorResult, ...]


EvalFn = Callable[[nnx.Module, nnx.Module, PRNGKeyArray], MatchStats]


def _where_done(done: Array, done_value: Any, active_value: Any) -> Any:
    mask = done.reshape((done.shape[0],) + (1,) * (active_value.ndim - 1))
    return jnp.where(mask, done_value, active_value)


def play_match_batch(
    *,
    env: pgx.Env,
    candidate_model: nnx.Module,
    anchor_model: nnx.Module,
    rng_key: PRNGKeyArray,
    batch_size: int,
    max_num_steps: int,
    num_simulations: int,
    max_num_considered_actions: int,
    max_depth: int | None,
    gumbel_scale: float,
) -> MatchStats:
    """Play a batched candidate-vs-anchor match set with alternating colors."""

    if batch_size % 2 != 0:
        msg = "Evaluation batch_size must be even for color balancing."
        raise ValueError(msg)

    init = jax.vmap(env.init)
    step = jax.vmap(env.step)
    rng_key, init_key, dummy_key, scan_key = jax.random.split(rng_key, 4)
    state = init(jax.random.split(init_key, batch_size))
    dummy_state = init(jax.random.split(dummy_key, batch_size))
    candidate_player = jnp.arange(batch_size, dtype=jnp.int32) % 2
    batch_ix = jnp.arange(batch_size)

    def step_fn(
        carry: tuple[pgx.State, Float[Array, "batch"], Array],
        key: PRNGKeyArray,
    ) -> tuple[tuple[pgx.State, Float[Array, "batch"], Array], None]:
        state, candidate_return, done = carry
        candidate_key, anchor_key = jax.random.split(key)
        search_state = jax.tree_util.tree_map(
            lambda dummy, active: _where_done(done, dummy, active), dummy_state, state
        )

        #TODO: this is wasteful as it uses 2x compute. fix that.
        candidate_policy = run_search(
            env=env,
            model=candidate_model,
            rng_key=candidate_key,
            state=search_state,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            max_depth=max_depth,
            gumbel_scale=gumbel_scale,
        )
        anchor_policy = run_search(
            env=env,
            model=anchor_model,
            rng_key=anchor_key,
            state=search_state,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            max_depth=max_depth,
            gumbel_scale=gumbel_scale,
        )

        candidate_to_move = search_state.current_player == candidate_player
        action = jnp.where(candidate_to_move, candidate_policy.action, anchor_policy.action)
        next_state = step(search_state, action)
        reward = next_state.rewards[batch_ix, candidate_player]
        candidate_return = candidate_return + jnp.where(done, 0.0, reward)
        next_done = next_state.terminated | next_state.truncated
        state = jax.tree_util.tree_map(
            lambda old, new: _where_done(done, old, new), state, next_state
        )
        return (state, candidate_return, done | next_done), None

    keys = jax.random.split(scan_key, max_num_steps)
    init_carry = (
        state,
        jnp.zeros(batch_size, dtype=jnp.float32),
        jnp.zeros(batch_size, dtype=jnp.bool_),
    )
    _, candidate_return, _ = jax.lax.scan(step_fn, init_carry, keys)[0]

    wins = jnp.sum(candidate_return > 0).astype(jnp.float32)
    losses = jnp.sum(candidate_return < 0).astype(jnp.float32)
    games = jnp.asarray(batch_size, dtype=jnp.float32)
    draws = games - wins - losses
    score = (wins + 0.5 * draws) / games
    white_games = jnp.sum(candidate_player == 0).astype(jnp.float32)
    black_games = jnp.sum(candidate_player == 1).astype(jnp.float32)
    return MatchStats(
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        score=score,
        white_games=white_games,
        black_games=black_games,
    )


def score_to_elo(score: float, anchor_elo: float, games: int) -> float:
    """Convert a finite match score into implied Elo versus an anchor."""

    if games <= 0:
        msg = "games must be positive"
        raise ValueError(msg)
    epsilon = 0.5 / games
    clipped_score = min(max(float(score), epsilon), 1.0 - epsilon)
    return float(anchor_elo + 400.0 * math.log10(clipped_score / (1.0 - clipped_score)))


def evaluate_vs_anchors(
    *,
    eval_fn: EvalFn,
    candidate_model: nnx.Module,
    anchors: tuple[Anchor, ...],
    rng_key: PRNGKeyArray,
    iteration: int,
) -> EloReport:
    """Evaluate a candidate model against all frozen anchors."""

    if not anchors:
        msg = "At least one anchor is required for Elo evaluation."
        raise ValueError(msg)

    keys = jax.random.split(rng_key, len(anchors))
    results: list[AnchorResult] = []
    total_games = 0
    total_wins = 0
    total_draws = 0
    total_losses = 0
    weighted_elo = 0.0
    for anchor, key in zip(anchors, keys, strict=True):
        stats = jax.device_get(eval_fn(candidate_model, anchor.model, key))
        games = int(stats.games)
        wins = int(stats.wins)
        draws = int(stats.draws)
        losses = int(stats.losses)
        score = float(stats.score)
        implied_elo = score_to_elo(score, anchor.elo, games)
        results.append(
            AnchorResult(
                name=anchor.name,
                iteration=anchor.iteration,
                anchor_elo=float(anchor.elo),
                games=games,
                wins=wins,
                draws=draws,
                losses=losses,
                score=score,
                implied_elo=implied_elo,
                elo_delta=implied_elo - float(anchor.elo),
            )
        )
        total_games += games
        total_wins += wins
        total_draws += draws
        total_losses += losses
        weighted_elo += implied_elo * games

    elo = weighted_elo / total_games
    return EloReport(
        iteration=int(iteration),
        elo=float(elo),
        games=total_games,
        wins=total_wins,
        draws=total_draws,
        losses=total_losses,
        win_rate=total_wins / total_games,
        draw_rate=total_draws / total_games,
        loss_rate=total_losses / total_games,
        anchors=tuple(results),
    )


def add_anchor(
    anchors: tuple[Anchor, ...],
    *,
    model: nnx.Module,
    elo: float,
    iteration: int,
    max_anchors: int,
) -> tuple[Anchor, ...]:
    """Append a frozen anchor while keeping the initial anchor and recent anchors."""

    new_anchor = Anchor(
        name=f"iter_{iteration:06d}",
        model=nnx.clone(model),
        elo=float(elo),
        iteration=int(iteration),
    )
    updated = (*anchors, new_anchor)
    if max_anchors <= 1:
        return (updated[0],)
    if len(updated) <= max_anchors:
        return updated
    return (updated[0], *updated[-(max_anchors - 1) :])


def anchor_summaries(anchors: tuple[Anchor, ...]) -> list[dict[str, float | int | str]]:
    """Return JSON-safe anchor metadata without parameter arrays."""

    return [
        {"name": anchor.name, "iteration": anchor.iteration, "elo": float(anchor.elo)}
        for anchor in anchors
    ]


def report_to_log_dict(report: EloReport) -> dict[str, float | int]:
    """Flatten an Elo report into numeric metrics for stdout or loggers."""

    log: dict[str, float | int] = {
        "eval/elo": report.elo,
        "eval/games": report.games,
        "eval/win_rate": report.win_rate,
        "eval/draw_rate": report.draw_rate,
        "eval/loss_rate": report.loss_rate,
    }
    for i, anchor in enumerate(report.anchors):
        log[f"eval/anchor_{i}/score"] = anchor.score
        log[f"eval/anchor_{i}/elo_delta"] = anchor.elo_delta
        log[f"eval/anchor_{i}/implied_elo"] = anchor.implied_elo
        log[f"eval/anchor_{i}/games"] = anchor.games
    return log


def report_to_json(report: EloReport) -> dict[str, Any]:
    """Convert an Elo report to a JSON-safe dictionary."""

    return {
        "iteration": report.iteration,
        "elo": report.elo,
        "games": report.games,
        "wins": report.wins,
        "draws": report.draws,
        "losses": report.losses,
        "win_rate": report.win_rate,
        "draw_rate": report.draw_rate,
        "loss_rate": report.loss_rate,
        "anchors": [anchor.__dict__ for anchor in report.anchors],
    }


def append_eval_history(path: str | Path, report: EloReport) -> None:
    """Append one JSONL evaluation report."""

    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(report_to_json(report), sort_keys=True) + "\n")
