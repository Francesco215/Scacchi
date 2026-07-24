from dataclasses import replace
from typing import Any, cast

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pgx

from .distributed import (
    DISABLED_BATCH_PARALLEL,
    BatchParallel,
    assert_batch_axis_sharded,
    constrain_batch_axis,
)
from .checkpoint import from_pretrained
from .play import play_eval
from .play_search import (
    make_search_player,
)
from .types import EvalBaseline, SearchConfig, SearchKind


def load_eval_baseline(config, env: pgx.Env, parallel: BatchParallel | None = None):
    """Build the configured evaluation opponent, if evaluation is enabled."""

    if config.eval.interval <= 0:
        return None
    if config.eval.baseline == EvalBaseline.none:
        raise ValueError("eval.baseline=none requires eval.interval=0.")
    if config.eval.baseline == EvalBaseline.checkpoint:
        checkpoint_path = config.eval.checkpoint_path if config.eval.checkpoint_path is not None else f"checkpoints/{config.env.board_size}_solved"
        return from_pretrained(checkpoint_path, env, rngs=nnx.Rngs(0))
    if config.eval.baseline == EvalBaseline.random:
        num_actions = env.num_actions

        def random_baseline_model(observation: jax.Array):
            logits = jnp.zeros((*observation.shape[:1], num_actions), dtype=jnp.float32)
            return constrain_batch_axis(logits, parallel, batch_axis=0)

        return random_baseline_model
    if config.eval.baseline == EvalBaseline.pgx:
        baseline_id = config.eval.baseline_id or f"{env.id}_v0"
        try:
            baseline_model = pgx.make_baseline_model(cast(Any, baseline_id))
        except AssertionError as exc:
            raise ValueError(f"PGX does not provide baseline model {baseline_id!r}. Use eval.baseline=none with eval.interval=0, or provide a checkpoint baseline.") from exc
        baseline_model = _compact_pgx_baseline_model(baseline_model, env)
        _validate_pgx_baseline(baseline_model, baseline_id, env)
        return baseline_model
    raise ValueError(f"unknown eval.baseline: {config.eval.baseline!r}")


def _compact_pgx_baseline_model(baseline_model: Any, env: pgx.Env):
    action_labels = getattr(env, "compact_action_labels", None)
    if action_labels is None:
        return baseline_model
    full_num_actions = int(getattr(env, "full_num_actions"))
    action_labels = jnp.asarray(action_labels, dtype=jnp.int32)

    def compact_model(observation: jax.Array):
        output = baseline_model(observation)
        logits = output[0] if isinstance(output, tuple) else output
        if logits.shape[-1] != full_num_actions:
            raise ValueError(f"baseline logits have {logits.shape[-1]} actions; expected {full_num_actions}")
        compact_logits = jnp.take(logits, action_labels, axis=-1)
        return (compact_logits, *output[1:]) if isinstance(output, tuple) else compact_logits

    return compact_model


def _validate_pgx_baseline(baseline_model: Any, baseline_id: str, env: pgx.Env) -> None:
    observation = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    try:
        output = baseline_model(observation)
    except Exception as exc:
        raise ValueError(f"PGX baseline model {baseline_id!r} is incompatible with env {env.id!r} observation_shape={env.observation_shape}.") from exc
    logits = output[0] if isinstance(output, tuple) else output
    if tuple(logits.shape) != (1, env.num_actions):
        raise ValueError(f"PGX baseline model {baseline_id!r} returned logits shape {tuple(logits.shape)} for env {env.id!r}; expected {(1, env.num_actions)}.")


def evaluation_metrics(returns: jax.Array, history: list[float]) -> dict[str, float]:
    """Record one evaluation and return its rolling diagnostics."""

    average = float(jax.device_get(returns.mean()))
    delta = 0.0 if not history else abs(average - history[-1])
    history.append(average)
    window = history[-10:]
    return {
        "eval/vs_baseline/avg_R_rolling_mean_10": float(np.mean(window)),
        "eval/vs_baseline/avg_R_rolling_std_10": float(np.std(window)),
        "eval/vs_baseline/avg_R_step_delta_abs": delta,
    }


def seat_conditioned_evaluation_metrics(
    returns: jax.Array,
    *,
    env_id: str,
) -> dict[str, float | int]:
    """Summarize the alternating candidate-seat layout used by evaluation.

    Even rows assign the candidate to PGX player 0 (the first seat), and odd
    rows assign it to player 1.  The unweighted seat-balanced rate remains
    exact when an odd batch gives the first seat one extra row.
    """

    first_returns = returns[0::2]
    second_returns = returns[1::2]

    def win_summary(
        values: jax.Array,
    ) -> tuple[int, int, float, float, float]:
        """Return games, wins, rate, and a Wilson 95% interval."""

        if values.size == 0:
            return 0, 0, 0.0, 0.0, 1.0
        games = int(values.size)
        wins = int(jax.device_get(jnp.sum(values > 0)))
        rate = wins / games
        z = 1.959963984540054
        denominator = 1.0 + z * z / games
        center = (rate + z * z / (2.0 * games)) / denominator
        half_width = (
            z
            * np.sqrt(
                rate * (1.0 - rate) / games
                + z * z / (4.0 * games * games)
            )
            / denominator
        )
        return (
            games,
            wins,
            rate,
            max(0.0, center - half_width),
            min(1.0, center + half_width),
        )

    (
        first_games,
        first_wins,
        first_win_rate,
        first_wilson_low,
        first_wilson_high,
    ) = win_summary(first_returns)
    (
        second_games,
        second_wins,
        second_win_rate,
        second_wilson_low,
        second_wilson_high,
    ) = win_summary(second_returns)
    seat_balanced_win_rate = 0.5 * (
        first_win_rate + second_win_rate
    )
    both_seats_observed = first_games > 0 and second_games > 0
    if both_seats_observed:
        seat_balanced_se = 0.5 * np.sqrt(
            first_win_rate * (1.0 - first_win_rate) / first_games
            + second_win_rate * (1.0 - second_win_rate) / second_games
        )
        # Averaging the two component Wilson bounds is conservative near
        # p=0/1 and remains useful when a plug-in normal variance vanishes.
        seat_balanced_low = 0.5 * (
            first_wilson_low + second_wilson_low
        )
        seat_balanced_high = 0.5 * (
            first_wilson_high + second_wilson_high
        )
    else:
        seat_balanced_se = 0.0
        seat_balanced_low = 0.0
        seat_balanced_high = 1.0
    metrics: dict[str, float | int] = {
        "eval/vs_baseline/first_seat_games": first_games,
        "eval/vs_baseline/second_seat_games": second_games,
        "eval/vs_baseline/first_seat_wins": first_wins,
        "eval/vs_baseline/second_seat_wins": second_wins,
        "eval/vs_baseline/first_seat_win_rate": first_win_rate,
        "eval/vs_baseline/first_seat_win_rate_wilson95_low": (
            first_wilson_low
        ),
        "eval/vs_baseline/first_seat_win_rate_wilson95_high": (
            first_wilson_high
        ),
        "eval/vs_baseline/second_seat_win_rate": second_win_rate,
        "eval/vs_baseline/second_seat_win_rate_wilson95_low": (
            second_wilson_low
        ),
        "eval/vs_baseline/second_seat_win_rate_wilson95_high": (
            second_wilson_high
        ),
        "eval/vs_baseline/seat_balanced_win_rate": seat_balanced_win_rate,
        "eval/vs_baseline/seat_balanced_win_rate_se": seat_balanced_se,
        "eval/vs_baseline/seat_balanced_win_rate_stratified95_low": (
            seat_balanced_low
        ),
        "eval/vs_baseline/seat_balanced_win_rate_stratified95_high": (
            seat_balanced_high
        ),
        "eval/vs_baseline/both_seats_observed": int(both_seats_observed),
        "eval/vs_baseline/first_minus_second_win_rate": (
            first_win_rate - second_win_rate
        ),
    }
    if env_id == "hex":
        seat_optimal_error = 0.5 * (
            (1.0 - first_win_rate) + second_win_rate
        )
        metrics["eval/vs_baseline/seat_optimal_error"] = seat_optimal_error
        metrics["eval/vs_baseline/seat_optimal_error_stratified95_low"] = (
            0.5 * ((1.0 - first_wilson_high) + second_wilson_low)
        )
        metrics["eval/vs_baseline/seat_optimal_error_stratified95_high"] = (
            0.5 * ((1.0 - first_wilson_low) + second_wilson_high)
        )
    return metrics


def baseline_search_config(config) -> SearchConfig:
    search_config = config.eval.baseline_search
    if config.eval.baseline != EvalBaseline.pgx:
        return search_config
    if search_config.kind in {SearchKind.policy, SearchKind.gumbel}:
        return search_config

    # PGX baselines expose scalar policy/value heads, not Dirichlet heads.
    num_simulations = max(1, int(search_config.active().num_simulations))
    return replace(
        search_config,
        kind=SearchKind.gumbel,
        gumbel=replace(search_config.gumbel, num_simulations=num_simulations),
    )


def make_mcts_evaluate(
    env,
    config,
    baseline_model,
    parallel: BatchParallel | None = None,
):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    eval_batch_size = int(config.eval.batch_size)
    player_search_config = config.eval.player_search
    player_action_commitment_type = config.eval.player_action_commitment_type
    opponent_search_config = baseline_search_config(config)
    opponent_action_commitment_type = config.eval.baseline_action_commitment_type

    def search_player(model, search_config, action_commitment_type):
        return make_search_player(
            env,
            model,
            search_config,
            action_commitment_type,
            q_loss_weight_mode=str(config.training.losses.q_loss_weight_mode),
        )

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: nnx.Module):
        """MCTS evaluation with exact alternating candidate seats."""
        metrics = play_eval(
            env,
            search_player(model, player_search_config, player_action_commitment_type),
            search_player(
                baseline_model,
                opponent_search_config,
                opponent_action_commitment_type,
            ),
            rng_key,
            batch_size=eval_batch_size,
            # ``None`` means even rows take the environment's initial
            # player ID and odd rows take the other ID.  Hex randomizes
            # player IDs, so alternating IDs here would not balance seats.
            player_1_id=None,
            parallel=parallel,
        )
        return assert_batch_axis_sharded(
            metrics.returns,
            parallel,
            batch_axis=0,
            label="eval returns",
        )

    return evaluate
