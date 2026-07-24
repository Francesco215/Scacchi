from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pgx
import pytest

from scacchi.logger import training_metrics
from scacchi.loss import _zero_train_metrics_like
from scacchi.pipeline import _with_data_stats
from scacchi.play import TrainingSamples, play
from scacchi.play_search import (
    PlayerOutput,
    PosteriorPrediction,
    PosteriorTargets,
)


def _samples(
    *,
    terminated: jax.Array,
    reward: jax.Array,
    played_action: jax.Array,
    actor: jax.Array | None,
    num_actions: int,
) -> TrainingSamples:
    batch_size, num_steps = terminated.shape
    return TrainingSamples(
        obs=jnp.zeros((batch_size, num_steps, 1), dtype=jnp.float32),
        reward=reward,
        terminated=terminated,
        discount=jnp.where(terminated, 0.0, -1.0),
        posterior=PosteriorTargets(
            prediction=PosteriorPrediction(
                policy=jnp.zeros(
                    (batch_size, num_steps, num_actions),
                    dtype=jnp.float32,
                )
            )
        ),
        played_action=played_action,
        legal_action_mask=jnp.ones(
            (batch_size, num_steps, num_actions),
            dtype=bool,
        ),
        actor=actor,
    )


def _logged_data_metrics(samples: TrainingSamples, num_actions: int):
    metrics = _with_data_stats(
        _zero_train_metrics_like(jnp.asarray(0.0, dtype=jnp.float32)),
        samples,
        num_actions,
    )
    return training_metrics(
        metrics,
        seconds=1.0,
        hours=0.0,
        frames=samples.terminated.size,
        frames_this_iteration=samples.terminated.size,
    )


def test_trajectory_metrics_pool_auto_reset_games_and_attribute_seats() -> None:
    terminated = jnp.asarray(
        [
            [False, True, False, False, True, False, False, False],
            [False, False, True, False, True, False, False, True],
        ]
    )
    reward = jnp.where(terminated, 1.0, 0.0).at[1, 7].set(0.0)
    actor = jnp.asarray(
        [
            # The first terminal intentionally breaks parity: attribution
            # must use the recorded actor, not assume strict alternation.
            [0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 1, 0],
        ],
        dtype=jnp.int32,
    )
    played_action = jnp.asarray(
        [
            [0, 1, 0, 2, 3, 0, 2, 4],
            [0, 2, 1, 1, 4, 1, 3, 2],
        ],
        dtype=jnp.int32,
    )

    logged = _logged_data_metrics(
        _samples(
            terminated=terminated,
            reward=reward,
            played_action=played_action,
            actor=actor,
            num_actions=5,
        ),
        num_actions=5,
    )

    assert logged["data/selfplay_outcome_count"] == 5.0
    assert logged["data/first_player_win_rate"] == pytest.approx(0.6)
    assert logged["data/second_player_win_rate"] == pytest.approx(0.2)
    assert logged["data/selfplay_draw_rate"] == pytest.approx(0.2)
    assert logged["data/first_player_score"] == pytest.approx(0.7)

    assert logged["data/game_length_mean"] == pytest.approx(2.6)
    assert logged["data/game_length_std"] == pytest.approx(math.sqrt(0.24))
    assert logged["data/game_length_p10"] == 2.0
    assert logged["data/game_length_p50"] == 3.0
    assert logged["data/game_length_p90"] == 3.0

    opening_probability = np.asarray([4.0 / 6.0, 2.0 / 6.0])
    opening_entropy = -float(
        np.sum(opening_probability * np.log(opening_probability))
    )
    assert logged["data/opening_action_sample_count"] == 6.0
    assert logged["data/opening_action_entropy_nats"] == pytest.approx(
        opening_entropy
    )
    assert logged["data/opening_action_effective_support"] == pytest.approx(
        math.exp(opening_entropy)
    )
    assert logged["data/opening_action_max_share"] == pytest.approx(4.0 / 6.0)
    assert logged["data/opening_action_unique_count"] == 2
    assert logged["data/opening_action_space_coverage"] == pytest.approx(0.4)

    ply_one_probability = np.asarray([1.0 / 6.0, 3.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    ply_one_entropy = -float(
        np.sum(ply_one_probability * np.log(ply_one_probability))
    )
    ply_two_entropy = math.log(4.0)
    assert logged["data/early_ply_active_count"] == 3
    assert logged["data/early_ply_0_action_entropy_nats"] == pytest.approx(
        opening_entropy
    )
    assert logged["data/early_ply_1_action_entropy_nats"] == pytest.approx(
        ply_one_entropy
    )
    assert logged["data/early_ply_2_action_entropy_nats"] == pytest.approx(
        ply_two_entropy
    )
    assert logged["data/early_ply_3_action_entropy_nats"] == 0.0
    assert logged["data/early_ply_action_entropy_mean_nats"] == pytest.approx(
        (opening_entropy + ply_one_entropy + ply_two_entropy) / 3.0
    )


def test_empty_trajectory_populations_log_only_finite_zeros() -> None:
    samples = _samples(
        terminated=jnp.zeros((1, 2), dtype=bool),
        reward=jnp.asarray([[jnp.nan, jnp.inf]], dtype=jnp.float32),
        played_action=jnp.asarray([[0, 1]], dtype=jnp.int32),
        actor=None,
        num_actions=3,
    )

    logged = _logged_data_metrics(samples, num_actions=3)

    for key in (
        "data/selfplay_outcome_count",
        "data/first_player_win_rate",
        "data/second_player_win_rate",
        "data/selfplay_draw_rate",
        "data/first_player_score",
        "data/game_length_mean",
        "data/game_length_std",
        "data/game_length_p10",
        "data/game_length_p50",
        "data/game_length_p90",
    ):
        assert logged[key] == 0.0
    assert all(
        math.isfinite(float(value))
        for key, value in logged.items()
        if key.startswith("data/")
    )


def test_play_training_records_the_pre_transition_actor() -> None:
    env = pgx.make("tic_tac_toe")

    def player(env_state, key: jax.Array) -> PlayerOutput:
        del key
        policy = env_state.legal_action_mask.astype(jnp.float32)
        policy /= jnp.sum(policy, axis=-1, keepdims=True)
        return PlayerOutput(
            action=jnp.argmax(env_state.legal_action_mask, axis=-1),
            posterior=PosteriorTargets(
                prediction=PosteriorPrediction(policy=policy)
            ),
        )

    samples = play(
        env,
        player,
        player,
        jax.random.PRNGKey(0),
        mode="training",
        batch_size=1,
        max_num_steps=3,
    )

    assert isinstance(samples, TrainingSamples)
    assert samples.actor is not None
    assert jnp.array_equal(samples.actor, jnp.asarray([[0, 1, 0]]))
