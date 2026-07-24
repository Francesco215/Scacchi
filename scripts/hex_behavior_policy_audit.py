#!/usr/bin/env python3
"""Audit Hex behavior policies at one frozen checkpoint.

The audit deliberately does not train, modify a checkpoint, or contact an
external service.  It restores one exact checkpoint step and runs the same
weights in self-play under five completed-root commitment rules:

* guarded true-Q21 followed by a categorical posterior sample;
* guarded true-Q21 power-transformed by a configured temperature and sampled;
* guarded true-Q21 followed by the plurality of 32 categorical votes;
* guarded true-Q21 followed by posterior argmax;
* native M32 followed by posterior argmax.

Internal repair and the replay target remain guarded Q21 in all three modes.
The game-coordinate PRNG keys are identical across modes.  Once actions
diverge the resulting states are, correctly, no longer paired positions.

The output contains frequency summaries rather than raw trajectories.  State
and ordered-prefix diversity are measured at every ply, with a separate
0--10-ply summary.  For unresolved roots whose Q21 target passed its numeric
guards, the audit also records whether posterior sampling is calibrated to
the commitment population.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import tempfile
import time
from typing import Any, Mapping, NamedTuple, Sequence

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from scacchi.dirichlet_mctx.native_targets import TARGET_CATEGORICAL
from scacchi.envs import make_env
from scacchi.play_search import PlayerOutput, make_search_player
from scacchi.types import (
    ActionCommitmentType,
    PosteriorPolicyEstimator,
    SearchConfig,
    SearchKind,
)
from scripts import hex_balanced_eval as balanced
from scripts import hex_checkpoint_league as league


SCHEMA_VERSION = 3
KIND = "scacchi.hex_behavior_policy_audit"
EARLY_LAST_PLY = 10


@dataclass(frozen=True)
class ModeSpec:
    mode_id: str
    root_action_estimator: PosteriorPolicyEstimator
    action_commitment_type: ActionCommitmentType
    description: str
    uses_configured_sample_temperature: bool = False


MODE_SPECS = (
    ModeSpec(
        mode_id="q21_posterior_sample",
        root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
        action_commitment_type=ActionCommitmentType.posterior_sample,
        description=(
            "guarded Q21 unresolved-root population followed by one "
            "categorical action draw"
        ),
    ),
    ModeSpec(
        mode_id="q21_posterior_temperature",
        root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
        action_commitment_type=ActionCommitmentType.posterior_sample,
        description=(
            "guarded Q21 unresolved-root population power-transformed by "
            "the configured posterior-sample temperature, followed by one "
            "categorical action draw"
        ),
        uses_configured_sample_temperature=True,
    ),
    ModeSpec(
        mode_id="q21_posterior_plurality32",
        root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
        action_commitment_type=ActionCommitmentType.posterior_plurality,
        description=(
            "guarded Q21 unresolved-root population followed by the "
            "lowest-index plurality winner of 32 categorical votes"
        ),
    ),
    ModeSpec(
        mode_id="q21_posterior_argmax",
        root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
        action_commitment_type=ActionCommitmentType.posterior_argmax,
        description=(
            "guarded Q21 unresolved-root population followed by argmax"
        ),
    ),
    ModeSpec(
        mode_id="m32_posterior_argmax",
        root_action_estimator=PosteriorPolicyEstimator.winner_mc,
        action_commitment_type=ActionCommitmentType.posterior_argmax,
        description=(
            "native 32-sample posterior-best population followed by argmax"
        ),
    ),
)


class AuditChunk(NamedTuple):
    valid: jax.Array
    cells: jax.Array
    current_color: jax.Array
    action: jax.Array
    legal_action_mask: jax.Array
    q21_policy: jax.Array
    root_solved: jax.Array
    target_prefix_eligible: jax.Array
    target_prefix_accepted: jax.Array
    target_prefix_fallback: jax.Array
    action_prefix_eligible: jax.Array
    action_prefix_accepted: jax.Array
    action_prefix_fallback: jax.Array
    prefix_tail_clipped: jax.Array
    prefix_density_guard: jax.Array
    prefix_nonfinite: jax.Array
    first_player_return: jax.Array
    game_length: jax.Array
    completed: jax.Array


@dataclass(frozen=True)
class HostModeData:
    valid: np.ndarray
    cells: np.ndarray
    current_color: np.ndarray
    action: np.ndarray
    legal_action_mask: np.ndarray
    q21_policy: np.ndarray
    root_solved: np.ndarray
    target_prefix_eligible: np.ndarray
    target_prefix_accepted: np.ndarray
    target_prefix_fallback: np.ndarray
    action_prefix_eligible: np.ndarray
    action_prefix_accepted: np.ndarray
    action_prefix_fallback: np.ndarray
    prefix_tail_clipped: np.ndarray
    prefix_density_guard: np.ndarray
    prefix_nonfinite: np.ndarray
    first_player_return: np.ndarray
    game_length: np.ndarray
    completed: np.ndarray


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def build_mode_search(
    stored_search: SearchConfig,
    spec: ModeSpec,
    *,
    num_simulations: int = 32,
    policy_samples: int = 32,
    posterior_plurality_samples: int = 32,
    posterior_sample_temperature: float = 1.0 / 3.0,
    kappa: float = 3.0,
    prefix_cdf_half_width: int = 10,
) -> SearchConfig:
    """Return the common DT32/Q21 search with one commitment estimator."""

    if stored_search.kind != SearchKind.dirichlet_thompson:
        raise ValueError(
            "behavior audit requires selfplay.search.kind="
            "dirichlet_thompson"
        )
    if num_simulations < 1:
        raise ValueError("num_simulations must be positive")
    if policy_samples < 1:
        raise ValueError("policy_samples must be positive")
    if posterior_plurality_samples < 1:
        raise ValueError("posterior_plurality_samples must be positive")
    if (
        not math.isfinite(posterior_sample_temperature)
        or posterior_sample_temperature <= 0.0
    ):
        raise ValueError(
            "posterior_sample_temperature must be finite and positive"
        )
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    if prefix_cdf_half_width < 1:
        raise ValueError("prefix_cdf_half_width must be positive")

    active = stored_search.dirichlet_thompson
    effective_active = replace(
        active,
        num_simulations=int(num_simulations),
        max_depth=int(num_simulations),
        kappa=float(kappa),
        policy_samples=int(policy_samples),
        posterior_policy_estimator=PosteriorPolicyEstimator.prefix_cdf,
        root_policy_target_estimator=PosteriorPolicyEstimator.prefix_cdf,
        root_action_estimator=spec.root_action_estimator,
        prefix_cdf_half_width=int(prefix_cdf_half_width),
    )
    return replace(
        stored_search,
        posterior_plurality_samples=int(posterior_plurality_samples),
        posterior_sample_temperature=(
            float(posterior_sample_temperature)
            if spec.uses_configured_sample_temperature
            else 1.0
        ),
        dirichlet_thompson=effective_active,
    )


def _required_posterior(output: PlayerOutput) -> Any:
    if output.posterior is None:
        raise ValueError("audit player did not return posterior targets")
    if output.posterior.metadata is None:
        raise ValueError("audit player did not return target metadata")
    if output.posterior.diagnostics is None:
        raise ValueError("audit player did not return search diagnostics")
    if output.posterior.metadata.v_target_kind is None:
        raise ValueError("audit player did not return root target kind")
    return output.posterior


def make_audit_evaluator(
    env: Any,
    search: SearchConfig,
    commitment: ActionCommitmentType,
    *,
    batch_size: int,
    q_loss_weight_mode: str,
) -> Any:
    """Build one frozen-weight, same-policy self-play game evaluator."""

    board_size = int(env.size)
    board_cells = board_size * board_size
    num_actions = int(env.num_actions)
    # PGX Hex exposes one extra pie-rule/swap action.  A swapped game can take
    # one more move than there are board cells, while every recorded board
    # state still contains exactly ``size**2`` cells.
    max_plies = num_actions
    if num_actions != board_cells + 1:
        raise ValueError(
            "behavior audit expects PGX Hex to expose size**2 board actions "
            "plus one swap action"
        )

    @nnx.jit
    def evaluate_chunk(
        rng_key: jax.Array,
        model: nnx.Module,
    ) -> AuditChunk:
        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, batch_size)
        state = jax.vmap(env.init)(init_keys)
        first_player_id = state._player_order[:, 0]
        player = make_search_player(
            env,
            model,
            search,
            commitment,
            q_loss_weight_mode=q_loss_weight_mode,
        )

        trace = AuditChunk(
            valid=jnp.zeros((batch_size, max_plies), dtype=jnp.bool_),
            cells=jnp.zeros(
                (batch_size, max_plies, board_cells),
                dtype=jnp.int8,
            ),
            current_color=jnp.full(
                (batch_size, max_plies),
                -1,
                dtype=jnp.int8,
            ),
            action=jnp.full(
                (batch_size, max_plies),
                -1,
                dtype=jnp.int16,
            ),
            legal_action_mask=jnp.zeros(
                (batch_size, max_plies, num_actions),
                dtype=jnp.bool_,
            ),
            q21_policy=jnp.zeros(
                (batch_size, max_plies, num_actions),
                dtype=jnp.float32,
            ),
            root_solved=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            target_prefix_eligible=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            target_prefix_accepted=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            target_prefix_fallback=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            action_prefix_eligible=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            action_prefix_accepted=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            action_prefix_fallback=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            prefix_tail_clipped=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            prefix_density_guard=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            prefix_nonfinite=jnp.zeros(
                (batch_size, max_plies),
                dtype=jnp.bool_,
            ),
            first_player_return=jnp.zeros(
                (batch_size,),
                dtype=jnp.float32,
            ),
            game_length=jnp.zeros((batch_size,), dtype=jnp.int16),
            completed=jnp.zeros((batch_size,), dtype=jnp.bool_),
        )

        def body_fn(
            carry: tuple[jax.Array, Any, jax.Array, AuditChunk],
        ) -> tuple[jax.Array, Any, jax.Array, AuditChunk]:
            loop_key, loop_state, ply, loop_trace = carry
            loop_key, player_key = jax.random.split(loop_key)
            active = ~loop_state.terminated
            output = player(loop_state, player_key)
            posterior = _required_posterior(output)
            metadata = posterior.metadata
            diagnostics = posterior.diagnostics
            if metadata is None or diagnostics is None:
                raise AssertionError("required posterior fields disappeared")
            v_target_kind = metadata.v_target_kind
            if v_target_kind is None:
                raise AssertionError("required root target kind disappeared")

            raw_board = loop_state._x.board
            current_color = loop_state._x.color
            fixed_cells = jnp.where(
                raw_board == 0,
                0,
                jnp.where(
                    raw_board > 0,
                    current_color[:, None] + 1,
                    2 - current_color[:, None],
                ),
            ).astype(jnp.int8)
            policy = posterior.prediction.policy.astype(jnp.float32)

            def put(array: jax.Array, values: jax.Array) -> jax.Array:
                return array.at[:, ply].set(values)

            next_state = jax.vmap(env.step)(loop_state, output.action)
            reward = next_state.rewards[
                jnp.arange(batch_size),
                first_player_id,
            ]
            loop_trace = loop_trace._replace(
                valid=put(loop_trace.valid, active),
                cells=put(loop_trace.cells, fixed_cells),
                current_color=put(
                    loop_trace.current_color,
                    current_color.astype(jnp.int8),
                ),
                action=put(
                    loop_trace.action,
                    output.action.astype(jnp.int16),
                ),
                legal_action_mask=put(
                    loop_trace.legal_action_mask,
                    loop_state.legal_action_mask,
                ),
                q21_policy=put(loop_trace.q21_policy, policy),
                root_solved=put(
                    loop_trace.root_solved,
                    v_target_kind == int(TARGET_CATEGORICAL),
                ),
                target_prefix_eligible=put(
                    loop_trace.target_prefix_eligible,
                    diagnostics.search_root_policy_target_prefix_eligible_count
                    > 0,
                ),
                target_prefix_accepted=put(
                    loop_trace.target_prefix_accepted,
                    diagnostics.search_root_policy_target_prefix_accepted_count
                    > 0,
                ),
                target_prefix_fallback=put(
                    loop_trace.target_prefix_fallback,
                    diagnostics.search_root_policy_target_prefix_fallback_count
                    > 0,
                ),
                action_prefix_eligible=put(
                    loop_trace.action_prefix_eligible,
                    diagnostics.search_root_action_prefix_eligible_count > 0,
                ),
                action_prefix_accepted=put(
                    loop_trace.action_prefix_accepted,
                    diagnostics.search_root_action_prefix_accepted_count > 0,
                ),
                action_prefix_fallback=put(
                    loop_trace.action_prefix_fallback,
                    diagnostics.search_root_action_prefix_fallback_count > 0,
                ),
                prefix_tail_clipped=put(
                    loop_trace.prefix_tail_clipped,
                    diagnostics.search_root_policy_target_prefix_tail_clipped_count
                    > 0,
                ),
                prefix_density_guard=put(
                    loop_trace.prefix_density_guard,
                    diagnostics.search_root_policy_target_prefix_density_guard_count
                    > 0,
                ),
                prefix_nonfinite=put(
                    loop_trace.prefix_nonfinite,
                    diagnostics.search_root_policy_target_prefix_nonfinite_count
                    > 0,
                ),
                first_player_return=(
                    loop_trace.first_player_return
                    + jnp.where(active, reward, 0.0)
                ),
                game_length=(
                    loop_trace.game_length + active.astype(jnp.int16)
                ),
                completed=next_state.terminated,
            )
            return loop_key, next_state, ply + 1, loop_trace

        _, final_state, _, trace = nnx.while_loop(
            lambda carry: (
                (carry[2] < max_plies)
                & ~carry[1].terminated.all()
            ),
            body_fn,
            (
                key,
                state,
                jnp.asarray(0, dtype=jnp.int32),
                trace,
            ),
        )
        return trace._replace(completed=final_state.terminated)

    return evaluate_chunk


def _frequency_stats(keys: Sequence[Any]) -> dict[str, Any]:
    count = len(keys)
    if count == 0:
        return {
            "sample_count": 0,
            "unique_count": 0,
            "unique_rate": 0.0,
            "entropy_nats": 0.0,
            "effective_support": 0.0,
            "max_share": 0.0,
        }
    frequencies = np.asarray(
        list(Counter(keys).values()),
        dtype=np.float64,
    )
    probability = frequencies / float(count)
    entropy = float(-np.sum(probability * np.log(probability)))
    unique = int(frequencies.size)
    return {
        "sample_count": count,
        "unique_count": unique,
        "unique_rate": unique / count,
        "entropy_nats": entropy,
        "effective_support": float(np.exp(entropy)),
        "max_share": float(np.max(probability)),
    }


def _action_stats(actions: np.ndarray, num_actions: int) -> dict[str, Any]:
    flat = np.asarray(actions, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        result = _frequency_stats(())
        result["counts"] = [0] * num_actions
        return result
    if np.any((flat < 0) | (flat >= num_actions)):
        raise ValueError("valid audit actions contain an out-of-range value")
    counts = np.bincount(flat, minlength=num_actions)
    result = _frequency_stats(flat.tolist())
    result["counts"] = [int(value) for value in counts.tolist()]
    return result


def _state_key(cells: np.ndarray, current_color: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(cells, dtype=np.int8).tobytes())
    digest.update(int(current_color).to_bytes(1, "big", signed=False))
    return digest.hexdigest()


def _prefix_key(actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    values = np.asarray(actions, dtype=np.int16)
    digest.update(len(values).to_bytes(2, "big"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _per_ply_diversity(
    data: HostModeData,
    *,
    num_actions: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for ply in range(data.valid.shape[1]):
        rows = np.flatnonzero(data.valid[:, ply])
        state_keys = [
            _state_key(
                data.cells[row, ply],
                int(data.current_color[row, ply]),
            )
            for row in rows
        ]
        prefix_keys = [
            _prefix_key(data.action[row, : ply + 1])
            for row in rows
        ]
        results.append(
            {
                "ply": ply,
                "actions": _action_stats(
                    data.action[rows, ply],
                    num_actions,
                ),
                "states": _frequency_stats(state_keys),
                "ordered_prefixes": _frequency_stats(prefix_keys),
            }
        )
    return results


def _early_diversity(
    data: HostModeData,
    *,
    num_actions: int,
    last_ply: int = EARLY_LAST_PLY,
) -> dict[str, Any]:
    stop = min(last_ply + 1, data.valid.shape[1])
    coordinates = np.argwhere(data.valid[:, :stop])
    actions = np.asarray(
        [data.action[row, ply] for row, ply in coordinates],
        dtype=np.int64,
    )
    state_keys = [
        f"{int(ply)}:{_state_key(data.cells[row, ply], int(data.current_color[row, ply]))}"
        for row, ply in coordinates
    ]
    prefix_keys = [
        f"{int(ply)}:{_prefix_key(data.action[row, : ply + 1])}"
        for row, ply in coordinates
    ]
    return {
        "ply_range_inclusive": [0, stop - 1],
        "actions": _action_stats(actions, num_actions),
        "states": _frequency_stats(state_keys),
        "ordered_prefixes": _frequency_stats(prefix_keys),
    }


def _game_summary(data: HostModeData) -> dict[str, Any]:
    lengths = data.game_length.astype(np.int64)
    returns = np.rint(data.first_player_return).astype(np.int64)
    if not bool(np.all(data.completed)):
        raise ValueError("at least one Hex game did not complete")
    if np.any(lengths <= 0):
        raise ValueError("completed games must have positive lengths")
    if np.any(~np.isin(returns, (-1, 0, 1))):
        raise ValueError("Hex return is not encoded as -1, 0, or 1")
    games = int(lengths.size)
    first_wins = int(np.sum(returns > 0))
    second_wins = int(np.sum(returns < 0))
    draws = int(np.sum(returns == 0))
    quantiles = np.quantile(
        lengths,
        (0.10, 0.50, 0.90),
        method="inverted_cdf",
    )
    histogram = np.bincount(
        lengths,
        minlength=data.valid.shape[1] + 1,
    )
    total_frames = int(np.sum(lengths))
    return {
        "games": games,
        "first_player_wins": first_wins,
        "second_player_wins": second_wins,
        "draws": draws,
        "first_player_win_rate": first_wins / games,
        "second_player_win_rate": second_wins / games,
        "draw_rate": draws / games,
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "length_p10": int(quantiles[0]),
        "length_p50": int(quantiles[1]),
        "length_p90": int(quantiles[2]),
        "length_histogram": [
            int(value) for value in histogram.tolist()
        ],
        "terminal_events_per_1k_frames": 1000.0 * games / total_frames,
    }


def _guard_summary(data: HostModeData) -> dict[str, Any]:
    valid = data.valid

    def count(mask: np.ndarray) -> int:
        return int(np.sum(valid & mask))

    target_eligible = count(data.target_prefix_eligible)
    target_accepted = count(data.target_prefix_accepted)
    target_fallback = count(data.target_prefix_fallback)
    action_eligible = count(data.action_prefix_eligible)
    action_accepted = count(data.action_prefix_accepted)
    action_fallback = count(data.action_prefix_fallback)
    return {
        "target": {
            "eligible_count": target_eligible,
            "accepted_count": target_accepted,
            "fallback_count": target_fallback,
            "acceptance_fraction": (
                target_accepted / target_eligible
                if target_eligible
                else 0.0
            ),
            "fallback_fraction": (
                target_fallback / target_eligible
                if target_eligible
                else 0.0
            ),
        },
        "action": {
            "eligible_count": action_eligible,
            "accepted_count": action_accepted,
            "fallback_count": action_fallback,
            "acceptance_fraction": (
                action_accepted / action_eligible
                if action_eligible
                else 0.0
            ),
            "fallback_fraction": (
                action_fallback / action_eligible
                if action_eligible
                else 0.0
            ),
        },
        "tail_clipped_count": count(data.prefix_tail_clipped),
        "density_guard_count": count(data.prefix_density_guard),
        "nonfinite_count": count(data.prefix_nonfinite),
    }


def _power_temperature_policy(
    policies: np.ndarray,
    legal: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Mirror production posterior-sample logits on the host."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("sampling temperature must be finite and positive")
    policies = np.asarray(policies, dtype=np.float64)
    legal = np.asarray(legal, dtype=bool)
    if policies.shape != legal.shape:
        raise ValueError("policy/legal shape mismatch")
    logits = np.log(np.clip(policies, 1e-8, 1.0)) / temperature
    logits = np.where(legal, logits, -np.inf)
    row_max = np.max(logits, axis=-1, keepdims=True)
    probability = np.where(legal, np.exp(logits - row_max), 0.0)
    return probability / np.sum(probability, axis=-1, keepdims=True)


def _commitment_q_summary(
    data: HostModeData,
    *,
    num_actions: int,
    sampling_calibration_applicable: bool,
    sampling_temperature: float,
) -> dict[str, Any]:
    mask = (
        data.valid
        & ~data.root_solved
        & data.target_prefix_accepted
    )
    policies = data.q21_policy[mask].astype(np.float64)
    legal = data.legal_action_mask[mask].astype(bool)
    actions = data.action[mask].astype(np.int64)
    count = int(actions.size)
    if count == 0:
        return {
            "q21_accepted_unresolved_count": 0,
            "sampling_calibration_applicable": (
                sampling_calibration_applicable
            ),
            "sampling_temperature": sampling_temperature,
        }
    if not bool(np.all(np.isfinite(policies))):
        raise ValueError("accepted Q21 policies contain a nonfinite value")
    row_mass = np.sum(policies, axis=1)
    if not bool(np.allclose(row_mass, 1.0, atol=2e-5, rtol=2e-5)):
        raise ValueError("accepted Q21 policies are not normalized")
    if np.any((actions < 0) | (actions >= num_actions)):
        raise ValueError("commitment action is outside Q21 policy")
    if np.any(~legal[np.arange(count), actions]):
        raise ValueError("commitment action is illegal")

    clipped = np.clip(policies, 0.0, 1.0)
    sampling_policy = _power_temperature_policy(
        clipped,
        legal,
        sampling_temperature,
    )
    positive = clipped > 0.0
    entropy = -np.sum(
        np.where(positive, clipped * np.log(np.maximum(clipped, 1e-300)), 0.0),
        axis=1,
    )
    sampling_positive = sampling_policy > 0.0
    sampling_entropy = -np.sum(
        np.where(
            sampling_positive,
            sampling_policy
            * np.log(np.maximum(sampling_policy, 1e-300)),
            0.0,
        ),
        axis=1,
    )
    q_max = np.max(clipped, axis=1)
    sampling_max = np.max(sampling_policy, axis=1)
    q_argmax = np.argmax(clipped, axis=1)
    q_played = clipped[np.arange(count), actions]
    observed_argmax = actions == q_argmax
    expected_counts = np.sum(sampling_policy, axis=0)
    observed_counts = np.bincount(actions, minlength=num_actions).astype(
        np.float64
    )
    residual = observed_counts - expected_counts
    covariance = (
        np.diag(expected_counts)
        - sampling_policy.T @ sampling_policy
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = max(
        float(np.max(np.abs(eigenvalues))) * 1e-10,
        1e-10,
    )
    covariance_rank = int(np.sum(eigenvalues > tolerance))
    mahalanobis = float(
        residual @ np.linalg.pinv(covariance, rcond=1e-10) @ residual
    )
    expected_argmax_count = float(np.sum(sampling_max))
    argmax_variance = float(
        np.sum(sampling_max * (1.0 - sampling_max))
    )
    argmax_z = (
        (float(np.sum(observed_argmax)) - expected_argmax_count)
        / math.sqrt(argmax_variance)
        if argmax_variance > 0.0
        else 0.0
    )
    return {
        "q21_accepted_unresolved_count": count,
        "sampling_temperature": sampling_temperature,
        "policy_entropy_mean_nats": float(np.mean(entropy)),
        "policy_effective_support_mean": float(np.mean(np.exp(entropy))),
        "sampling_policy_entropy_mean_nats": float(
            np.mean(sampling_entropy)
        ),
        "sampling_policy_effective_support_mean": float(
            np.mean(np.exp(sampling_entropy))
        ),
        "policy_max_probability_mean": float(np.mean(q_max)),
        "sampling_policy_max_probability_mean": float(
            np.mean(sampling_max)
        ),
        "played_action_q21_mass_mean": float(np.mean(q_played)),
        "q21_max_minus_played_mass_mean": float(
            np.mean(q_max - q_played)
        ),
        "observed_q21_argmax_fraction": float(
            np.mean(observed_argmax)
        ),
        "expected_argmax_fraction_under_q21_sampling": (
            expected_argmax_count / count
        ),
        "expected_argmax_fraction_under_commitment_sampling": (
            expected_argmax_count / count
        ),
        "argmax_hit_calibration_z": argmax_z,
        "expected_action_counts_under_q21_sampling": [
            float(value) for value in expected_counts.tolist()
        ],
        "expected_action_counts_under_commitment_sampling": [
            float(value) for value in expected_counts.tolist()
        ],
        "observed_action_counts": [
            int(value) for value in observed_counts.astype(np.int64).tolist()
        ],
        "action_count_residuals_observed_minus_expected": [
            float(value) for value in residual.tolist()
        ],
        "multinomial_covariance_rank": covariance_rank,
        "multinomial_mahalanobis_statistic": mahalanobis,
        "sampling_calibration_applicable": (
            sampling_calibration_applicable
        ),
        "calibration_note": (
            "Under independent categorical draws from the varying effective "
            "commitment rows, the residual count covariance is "
            "sum(diag(q_T)-q_T q_T^T). The legacy fields named "
            "'under_q21_sampling' contain q_T when sampling_temperature is "
            "not one. The Mahalanobis statistic is descriptive when "
            "calibration is not applicable."
        ),
    }


def host_mode_data(chunks: Sequence[AuditChunk]) -> HostModeData:
    if not chunks:
        raise ValueError("cannot summarize zero audit chunks")

    def concatenate(field_name: str) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(
                    jax.device_get(getattr(chunk, field_name))
                )
                for chunk in chunks
            ],
            axis=0,
        )

    return HostModeData(
        **{
            field.name: concatenate(field.name)
            for field in fields(HostModeData)
        }
    )


def summarize_mode(
    data: HostModeData,
    *,
    num_actions: int,
    sampling_calibration_applicable: bool,
    sampling_temperature: float = 1.0,
) -> dict[str, Any]:
    """Reduce raw, fixed-checkpoint game traces into JSON-safe metrics."""

    return {
        "games": _game_summary(data),
        "early_plies_0_to_10": _early_diversity(
            data,
            num_actions=num_actions,
        ),
        "by_ply": _per_ply_diversity(
            data,
            num_actions=num_actions,
        ),
        "commitment_q21": _commitment_q_summary(
            data,
            num_actions=num_actions,
            sampling_calibration_applicable=(
                sampling_calibration_applicable
            ),
            sampling_temperature=sampling_temperature,
        ),
        "numeric_guards": _guard_summary(data),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _distance_fraction(
    candidate: float,
    reference: float,
    endpoint_a: float,
    endpoint_b: float,
) -> float | None:
    """Scale candidate error by the nearer failed-endpoint separation."""

    denominator = min(
        abs(reference - endpoint_a),
        abs(reference - endpoint_b),
    )
    if denominator <= 0.0:
        return None
    return abs(candidate - reference) / denominator


def compare_modes(mode_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return the predeclared early-diversity contrasts."""

    def contrast(
        treatment_id: str,
        control_id: str,
    ) -> dict[str, Any]:
        treatment = mode_results[treatment_id]["early_plies_0_to_10"]
        control = mode_results[control_id]["early_plies_0_to_10"]
        return {
            "state_effective_support_ratio": _safe_ratio(
                float(treatment["states"]["effective_support"]),
                float(control["states"]["effective_support"]),
            ),
            "state_unique_count_difference": (
                int(treatment["states"]["unique_count"])
                - int(control["states"]["unique_count"])
            ),
            "ordered_prefix_effective_support_ratio": _safe_ratio(
                float(treatment["ordered_prefixes"]["effective_support"]),
                float(control["ordered_prefixes"]["effective_support"]),
            ),
            "ordered_prefix_unique_count_difference": (
                int(treatment["ordered_prefixes"]["unique_count"])
                - int(control["ordered_prefixes"]["unique_count"])
            ),
            "action_effective_support_ratio": _safe_ratio(
                float(treatment["actions"]["effective_support"]),
                float(control["actions"]["effective_support"]),
            ),
        }

    comparisons: dict[str, Any] = {}
    for control_id in (
        "q21_posterior_argmax",
        "m32_posterior_argmax",
    ):
        comparisons[f"q21_posterior_sample_vs_{control_id}"] = contrast(
            "q21_posterior_sample",
            control_id,
        )
    for control_id in (
        "q21_posterior_sample",
        "q21_posterior_argmax",
        "m32_posterior_argmax",
    ):
        comparisons[
            f"q21_posterior_plurality32_vs_{control_id}"
        ] = contrast(
            "q21_posterior_plurality32",
            control_id,
        )
    for control_id in (
        "q21_posterior_sample",
        "q21_posterior_plurality32",
        "q21_posterior_argmax",
        "m32_posterior_argmax",
    ):
        comparisons[
            f"q21_posterior_temperature_vs_{control_id}"
        ] = contrast(
            "q21_posterior_temperature",
            control_id,
        )

    native = mode_results["m32_posterior_argmax"]
    sample = mode_results["q21_posterior_sample"]
    argmax = mode_results["q21_posterior_argmax"]
    native_early = native["early_plies_0_to_10"]
    sample_early = sample["early_plies_0_to_10"]
    argmax_early = argmax["early_plies_0_to_10"]
    native_games = native["games"]

    def equivalence(
        candidate_id: str,
        candidate_description: str,
    ) -> dict[str, Any]:
        candidate = mode_results[candidate_id]
        candidate_early = candidate["early_plies_0_to_10"]
        candidate_games = candidate["games"]
        metrics = {
            "opening_action_effective_support_ratio": _safe_ratio(
                float(
                    candidate["by_ply"][0]["actions"][
                        "effective_support"
                    ]
                ),
                float(
                    native["by_ply"][0]["actions"]["effective_support"]
                ),
            ),
            "plies_0_to_10_action_effective_support_ratio": _safe_ratio(
                float(candidate_early["actions"]["effective_support"]),
                float(native_early["actions"]["effective_support"]),
            ),
            "plies_0_to_10_state_effective_support_ratio": _safe_ratio(
                float(candidate_early["states"]["effective_support"]),
                float(native_early["states"]["effective_support"]),
            ),
            "plies_0_to_10_prefix_effective_support_ratio": _safe_ratio(
                float(
                    candidate_early["ordered_prefixes"][
                        "effective_support"
                    ]
                ),
                float(
                    native_early["ordered_prefixes"]["effective_support"]
                ),
            ),
            "terminal_event_rate_ratio": _safe_ratio(
                float(
                    candidate_games["terminal_events_per_1k_frames"]
                ),
                float(native_games["terminal_events_per_1k_frames"]),
            ),
            "mean_game_length_difference_plies": (
                float(candidate_games["length_mean"])
                - float(native_games["length_mean"])
            ),
            "first_player_win_rate_difference": (
                float(candidate_games["first_player_win_rate"])
                - float(native_games["first_player_win_rate"])
            ),
            "state_ess_error_fraction_of_nearer_endpoint_separation": (
                _distance_fraction(
                    float(candidate_early["states"]["effective_support"]),
                    float(native_early["states"]["effective_support"]),
                    float(sample_early["states"]["effective_support"]),
                    float(argmax_early["states"]["effective_support"]),
                )
            ),
            "prefix_ess_error_fraction_of_nearer_endpoint_separation": (
                _distance_fraction(
                    float(
                        candidate_early["ordered_prefixes"][
                            "effective_support"
                        ]
                    ),
                    float(
                        native_early["ordered_prefixes"][
                            "effective_support"
                        ]
                    ),
                    float(
                        sample_early["ordered_prefixes"][
                            "effective_support"
                        ]
                    ),
                    float(
                        argmax_early["ordered_prefixes"][
                            "effective_support"
                        ]
                    ),
                )
            ),
        }

        def in_interval(name: str, lower: float, upper: float) -> bool:
            value = metrics[name]
            return value is not None and lower <= value <= upper

        passed = {
            "opening_action_ess_ratio_in_0_95_to_1_05": in_interval(
                "opening_action_effective_support_ratio",
                0.95,
                1.05,
            ),
            "early_action_ess_ratio_in_0_95_to_1_05": in_interval(
                "plies_0_to_10_action_effective_support_ratio",
                0.95,
                1.05,
            ),
            "early_state_ess_ratio_in_0_90_to_1_10": in_interval(
                "plies_0_to_10_state_effective_support_ratio",
                0.90,
                1.10,
            ),
            "early_prefix_ess_ratio_in_0_90_to_1_10": in_interval(
                "plies_0_to_10_prefix_effective_support_ratio",
                0.90,
                1.10,
            ),
            "terminal_rate_ratio_in_0_95_to_1_05": in_interval(
                "terminal_event_rate_ratio",
                0.95,
                1.05,
            ),
            "mean_game_length_difference_at_most_0_5_ply": (
                abs(float(metrics["mean_game_length_difference_plies"]))
                <= 0.5
            ),
            "first_player_win_rate_difference_at_most_0_02": (
                abs(float(metrics["first_player_win_rate_difference"]))
                <= 0.02
            ),
            "state_ess_at_least_4x_closer_than_nearer_endpoint": (
                in_interval(
                    "state_ess_error_fraction_of_nearer_endpoint_separation",
                    0.0,
                    0.25,
                )
            ),
            "prefix_ess_at_least_4x_closer_than_nearer_endpoint": (
                in_interval(
                    "prefix_ess_error_fraction_of_nearer_endpoint_separation",
                    0.0,
                    0.25,
                )
            ),
        }
        return {
            "definitions": {
                "reference": "native M32 followed by lowest-index argmax",
                "candidate": candidate_description,
                "endpoint_separation": (
                    "the smaller absolute native-M32 distance to Q21 sample "
                    "or Q21 argmax on the same scalar ESS metric"
                ),
                "inference_limit": (
                    "single-block descriptive integration gate; independent "
                    "coordinate blocks are required for a boundary result"
                ),
            },
            "metrics": metrics,
            "pass": {
                **passed,
                "all_pass": bool(all(passed.values())),
            },
        }

    return {
        "scope": "pooled pre-action records at plies 0 through 10 inclusive",
        "inference_note": (
            "Common RNG coordinates identify matched game lanes, but states "
            "after the first action divergence are different estimands. "
            "These are descriptive frequency contrasts, not paired game "
            "confidence intervals."
        ),
        "contrasts": comparisons,
        "plurality32_native_m32_equivalence": equivalence(
            "q21_posterior_plurality32",
            "guarded Q21 followed by plurality of 32 votes",
        ),
        "temperature_native_m32_equivalence": equivalence(
            "q21_posterior_temperature",
            "guarded Q21 power-temperature followed by one draw",
        ),
    }


def _prng_key_data(key: jax.Array) -> list[int]:
    key_data = key
    if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        key_data = jax.random.key_data(key)
    return [
        int(value)
        for value in np.asarray(jax.device_get(key_data)).reshape(-1)
    ]


def common_coordinate_layout(
    *,
    seed: int,
    games: int,
    batch_size: int,
) -> tuple[jax.Array, dict[str, Any]]:
    if games <= 0:
        raise ValueError("--games must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if games % batch_size:
        raise ValueError("--games must be divisible by --batch-size")
    num_chunks = games // batch_size
    keys = jax.random.split(jax.random.PRNGKey(seed), num_chunks)
    key_rows = [
        {
            "chunk_index": index,
            "global_game_index_start": index * batch_size,
            "game_count": batch_size,
            "rng_key_data": _prng_key_data(keys[index]),
        }
        for index in range(num_chunks)
    ]
    material = {
        "seed": seed,
        "games": games,
        "batch_size": batch_size,
        "chunks": key_rows,
    }
    digest = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return keys, {
        **material,
        "layout_sha256": digest,
        "reuse": (
            "The exact same ordered chunk keys are reused for all four "
            "modes."
        ),
    }


def _write_json_create_once(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    """Atomically create an immutable JSON artifact."""

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-only Q21-sample/Q21-temperature/"
            "Q21-plurality32/Q21-argmax/M32-argmax behavior audit at one "
            "exact Hex checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--games", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=10175001)
    parser.add_argument("--num-simulations", type=int, default=32)
    parser.add_argument("--policy-samples", type=int, default=32)
    parser.add_argument(
        "--posterior-plurality-samples",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--posterior-sample-temperature",
        type=float,
        default=1.0 / 3.0,
        help=(
            "Power temperature used only by q21_posterior_temperature; "
            "q21_posterior_sample remains exactly temperature one."
        ),
    )
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--prefix-cdf-half-width", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print progress and the final path, but not the full JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    run_keys, coordinate_layout = common_coordinate_layout(
        seed=args.seed,
        games=args.games,
        batch_size=args.batch_size,
    )
    loaded = balanced.load_checkpoint_metadata(
        args.checkpoint,
        args.step,
    )
    if loaded.selection.selection_mode != "exact":
        raise AssertionError("behavior audit requires an exact checkpoint")
    if str(loaded.config.env.id) != "hex":
        raise ValueError("behavior audit currently supports Hex only")
    env = make_env(
        str(loaded.config.env.id),
        loaded.config.env.board_size,
    )
    model = balanced.load_model_at_step(loaded, env)
    q_loss_weight_mode = str(
        loaded.config.training.losses.q_loss_weight_mode
    )

    mode_results: dict[str, dict[str, Any]] = {}
    effective_modes: dict[str, dict[str, Any]] = {}
    runtime_modes: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for spec in MODE_SPECS:
        search = build_mode_search(
            loaded.config.selfplay.search,
            spec,
            num_simulations=args.num_simulations,
            policy_samples=args.policy_samples,
            posterior_plurality_samples=(
                args.posterior_plurality_samples
            ),
            posterior_sample_temperature=(
                args.posterior_sample_temperature
            ),
            kappa=args.kappa,
            prefix_cdf_half_width=args.prefix_cdf_half_width,
        )
        evaluator = make_audit_evaluator(
            env,
            search,
            spec.action_commitment_type,
            batch_size=args.batch_size,
            q_loss_weight_mode=q_loss_weight_mode,
        )
        mode_started = time.perf_counter()
        chunks: list[AuditChunk] = []
        for chunk_index, key in enumerate(run_keys):
            chunk = evaluator(key, model)
            jax.block_until_ready(chunk.valid)
            chunks.append(chunk)
            if not args.quiet:
                print(
                    f"{spec.mode_id}: chunk {chunk_index + 1}/"
                    f"{len(run_keys)}",
                    flush=True,
                )
        host = host_mode_data(chunks)
        mode_results[spec.mode_id] = summarize_mode(
            host,
            num_actions=int(env.num_actions),
            sampling_calibration_applicable=(
                spec.action_commitment_type
                == ActionCommitmentType.posterior_sample
                and spec.root_action_estimator
                == PosteriorPolicyEstimator.prefix_cdf
            ),
            sampling_temperature=(
                float(search.posterior_sample_temperature)
                if spec.action_commitment_type
                == ActionCommitmentType.posterior_sample
                else 1.0
            ),
        )
        effective_modes[spec.mode_id] = {
            "description": spec.description,
            "action_commitment_type": str(
                spec.action_commitment_type
            ),
            "search": _jsonable(search),
            "q_loss_weight_mode": q_loss_weight_mode,
            "same_checkpoint_for_both_players": True,
        }
        runtime_modes[spec.mode_id] = {
            "seconds": time.perf_counter() - mode_started,
            "games": args.games,
        }

    script_path = Path(__file__).resolve()
    step_root = (
        loaded.selection.directory
        / str(loaded.selection.selected_step)
    )
    metadata_path = step_root / "meta" / "metadata"
    reproduction_argv = list(
        sys.argv if argv is None else [sys.argv[0], *argv]
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
            "contains_raw_positions": False,
        },
        "reproduction": {
            "command": " ".join(
                shlex.quote(argument)
                for argument in reproduction_argv
            ),
            "working_directory": str(Path.cwd().resolve()),
            "script_path": str(script_path),
            "script_sha256": hashlib.sha256(
                script_path.read_bytes()
            ).hexdigest(),
        },
        "checkpoint": {
            **loaded.selection.provenance(),
            "metadata_sha256": league.file_sha256(metadata_path),
            "checkpoint_tree_sha256": league.tree_sha256(step_root),
            "stored_selfplay_search": _jsonable(
                loaded.config.selfplay.search
            ),
            "stored_selfplay_action_commitment_type": str(
                loaded.config.selfplay.action_commitment_type
            ),
        },
        "environment": {
            "id": str(loaded.config.env.id),
            "board_size": int(loaded.config.env.board_size),
            "num_actions": int(env.num_actions),
            "complete_game_max_plies": int(env.num_actions),
        },
        "coordinate_layout": coordinate_layout,
        "audit_contract": {
            "fixed_weights": True,
            "same_model_both_players": True,
            "common_coordinate_keys": True,
            "internal_repair_estimator": "prefix_cdf",
            "root_training_target_estimator": "prefix_cdf",
            "prefix_cdf_grid_points": (
                2 * int(args.prefix_cdf_half_width) + 1
            ),
            "native_root_population_samples": int(args.policy_samples),
            "posterior_plurality_samples": int(
                args.posterior_plurality_samples
            ),
            "posterior_sample_temperature": float(
                args.posterior_sample_temperature
            ),
            "early_ply_range_inclusive": [0, EARLY_LAST_PLY],
            "state_key": (
                "SHA-256 of fixed-colour board cells and current colour"
            ),
            "ordered_prefix_key": (
                "SHA-256 of prefix length and ordered played actions"
            ),
            "no_causal_strength_claim": (
                "Diversity and commitment calibration diagnose behavior; "
                "they do not establish useful weight transfer or game "
                "strength."
            ),
        },
        "effective_modes": effective_modes,
        "modes": mode_results,
        "comparisons": compare_modes(mode_results),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "by_mode": runtime_modes,
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "python": platform.python_version(),
            "jax": jax.__version__,
        },
    }
    digest = _write_json_create_once(args.output, result)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(
        f"Wrote {args.output.resolve()} sha256={digest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
