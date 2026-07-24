#!/usr/bin/env python3
"""Local-only, role-balanced evaluation of an exact Hex checkpoint.

The ordinary training monitor fixes the candidate as PGX logical player 0
while PGX independently randomizes which logical player receives the first
seat.  This harness evaluates equal-sized cells from the full product

    candidate logical player id in {0, 1}
        x candidate seat in {first, second}.

Unlike :func:`scacchi.checkpoint.from_pretrained`, ``--step`` restores the
requested retained Orbax step even when newer checkpoints exist.

Optionally, the harness can freeze a deterministic sample of non-terminal
pre-action Hex positions for later exact-oracle analysis.  Trace selection is
performed after the games finish and consumes no JAX PRNG keys, so enabling it
does not change evaluation actions or returns.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import platform
from pathlib import Path
import shlex
import sys
import time
from typing import Any, NamedTuple, Sequence

from flax import nnx
import jax
import jax.numpy as jnp

from scacchi import checkpoint as checkpoint_io
from scacchi.envs import make_env
from scacchi.evaluations import baseline_search_config
from scacchi.network import build_model
from scacchi.play_search import make_search_player
from scacchi.types import PosteriorPolicyEstimator, SearchKind


MAX_TRACE_EMPTY_COUNT = 15
TRACE_SCHEMA_VERSION = 1
GAME_RETURNS_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class Stratum:
    candidate_player_id: int
    candidate_first: bool
    candidate_seat: str


STRATUM_SPEC = (
    Stratum(0, True, "first"),
    Stratum(0, False, "second"),
    Stratum(1, True, "first"),
    Stratum(1, False, "second"),
)


@dataclass(frozen=True)
class CheckpointSelection:
    directory: Path
    requested_step: int | None
    selected_step: int
    available_steps: tuple[int, ...]

    @property
    def selection_mode(self) -> str:
        return "latest" if self.requested_step is None else "exact"

    def provenance(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "requested_step": self.requested_step,
            "selected_step": self.selected_step,
            "selection_mode": self.selection_mode,
            "available_steps": list(self.available_steps),
        }


@dataclass(frozen=True)
class LoadedCheckpointMetadata:
    selection: CheckpointSelection
    metadata: dict[str, Any]
    config: Any


class ChunkTrace(NamedTuple):
    valid: jax.Array
    cells: jax.Array
    action: jax.Array
    current_color: jax.Array
    actor_player_id: jax.Array


class BalancedChunkOutput(NamedTuple):
    returns: jax.Array
    trace: ChunkTrace


@dataclass(frozen=True)
class TraceRecord:
    empty_count: int
    cells: tuple[int, ...]
    current_color: int
    action: int
    actor_player_id: int
    actor_agent: str
    candidate_player_id: int
    candidate_seat: str
    final_candidate_return: int
    stratum_index: int
    chunk_index: int
    row_index: int

    @property
    def state_key(self) -> str:
        encoded = json.dumps(
            {
                "cells": self.cells,
                "current_color": self.current_color,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def sampling_priority(self, seed: int) -> str:
        source = (
            f"{seed}:{self.empty_count}:{self.stratum_index}:"
            f"{self.chunk_index}:{self.row_index}:{self.state_key}"
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def to_json(self, *, seed: int, board_size: int) -> dict[str, Any]:
        priority = self.sampling_priority(seed)
        payload = asdict(self)
        payload.update(
            {
                "trace_id": hashlib.sha256(
                    (
                        f"{priority}:{self.action}:{self.actor_player_id}"
                    ).encode("utf-8")
                ).hexdigest(),
                "position_id": self.state_key,
                "sample_priority": priority,
                "ply_index": board_size * board_size - self.empty_count,
                "candidate_won": self.final_candidate_return > 0,
            }
        )
        payload["cells"] = list(self.cells)
        return payload


def wilson_interval(
    wins: int,
    games: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if games <= 0:
        return (math.nan, math.nan)
    probability = wins / games
    z_squared = z * z
    denominator = 1.0 + z_squared / games
    centre = (
        probability + z_squared / (2.0 * games)
    ) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / games
            + z_squared / (4.0 * games * games)
        )
        / denominator
    )
    return (centre - radius, centre + radius)


def summarize_returns(
    returns: Sequence[int],
    *,
    candidate_id: int | None = None,
    seat: str | None = None,
) -> dict[str, Any]:
    games = len(returns)
    if games == 0:
        raise ValueError("cannot summarize zero games")
    wins = sum(value > 0 for value in returns)
    draws = sum(value == 0 for value in returns)
    losses = sum(value < 0 for value in returns)
    ci_low, ci_high = wilson_interval(wins, games)
    result: dict[str, Any] = {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / games,
        "avg_return": sum(returns) / games,
        "wilson_95": [ci_low, ci_high],
    }
    if candidate_id is not None:
        result["candidate_player_id"] = candidate_id
    if seat is not None:
        result["candidate_seat"] = seat
    return result


def _prng_key_data(key: jax.Array) -> list[int]:
    key_data = key
    if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        key_data = jax.random.key_data(key)
    return [
        int(value)
        for value in jax.device_get(key_data).reshape(-1).tolist()
    ]


def flatten_game_returns(payload: dict[str, Any]) -> list[int]:
    """Flatten a serialized return block in its declared pairing order."""

    returns: list[int] = []
    expected_global_index = 0
    for expected_stratum_index, stratum in enumerate(payload["strata"]):
        if int(stratum["stratum_index"]) != expected_stratum_index:
            raise ValueError("game-return strata are not in canonical order")
        for expected_chunk_index, chunk in enumerate(stratum["chunks"]):
            if int(chunk["chunk_index"]) != expected_chunk_index:
                raise ValueError(
                    "game-return chunks are not in canonical order"
                )
            if int(chunk["global_game_index_start"]) != expected_global_index:
                raise ValueError(
                    "game-return global game indices are not contiguous"
                )
            chunk_returns = [
                int(value) for value in chunk["returns"]
            ]
            if len(chunk_returns) != int(chunk["game_count"]):
                raise ValueError(
                    "game-return chunk length disagrees with game_count"
                )
            returns.extend(chunk_returns)
            expected_global_index += len(chunk_returns)
    if expected_global_index != int(payload["games"]):
        raise ValueError(
            "serialized game-return count disagrees with games"
        )
    return returns


def build_game_returns_payload(
    returns_by_chunk: Sequence[Sequence[Sequence[int]]],
    *,
    run_keys: Any,
    seed: int,
    games: int,
    games_per_stratum: int,
    batch_size: int,
) -> dict[str, Any]:
    """Serialize paired per-game outcomes without changing evaluation order."""

    if len(returns_by_chunk) != len(STRATUM_SPEC):
        raise ValueError("raw returns must contain exactly four strata")
    num_chunks = games_per_stratum // batch_size
    if games != games_per_stratum * len(STRATUM_SPEC):
        raise ValueError("games disagrees with games_per_stratum")
    if len(run_keys) != len(STRATUM_SPEC) * num_chunks:
        raise ValueError("run-key count disagrees with the chunk layout")

    serialized_strata: list[dict[str, Any]] = []
    pairing_layout: list[dict[str, Any]] = []
    for stratum_index, (stratum, stratum_chunks) in enumerate(
        zip(STRATUM_SPEC, returns_by_chunk, strict=True)
    ):
        if len(stratum_chunks) != num_chunks:
            raise ValueError(
                f"stratum {stratum_index} has {len(stratum_chunks)} chunks; "
                f"expected {num_chunks}"
            )
        serialized_chunks: list[dict[str, Any]] = []
        stratum_returns: list[int] = []
        for chunk_index, chunk_values in enumerate(stratum_chunks):
            chunk_returns = [int(value) for value in chunk_values]
            if len(chunk_returns) != batch_size:
                raise ValueError(
                    f"stratum {stratum_index} chunk {chunk_index} has "
                    f"{len(chunk_returns)} games; expected {batch_size}"
                )
            if any(value not in (-1, 0, 1) for value in chunk_returns):
                raise ValueError(
                    "candidate returns must be encoded as -1, 0, or 1"
                )
            run_key_index = stratum_index * num_chunks + chunk_index
            key_data = _prng_key_data(run_keys[run_key_index])
            global_start = (
                stratum_index * games_per_stratum
                + chunk_index * batch_size
            )
            serialized_chunks.append(
                {
                    "chunk_index": chunk_index,
                    "run_key_index": run_key_index,
                    "rng_key_data": key_data,
                    "global_game_index_start": global_start,
                    "game_count": batch_size,
                    "returns": chunk_returns,
                }
            )
            pairing_layout.append(
                {
                    "stratum_index": stratum_index,
                    "chunk_index": chunk_index,
                    "run_key_index": run_key_index,
                    "rng_key_data": key_data,
                    "global_game_index_start": global_start,
                    "game_count": batch_size,
                }
            )
            stratum_returns.extend(chunk_returns)
        serialized_strata.append(
            {
                "stratum_index": stratum_index,
                **asdict(stratum),
                "summary": summarize_returns(
                    stratum_returns,
                    candidate_id=stratum.candidate_player_id,
                    seat=stratum.candidate_seat,
                ),
                "chunks": serialized_chunks,
            }
        )

    layout_material = {
        "seed": seed,
        "games": games,
        "games_per_stratum": games_per_stratum,
        "batch_size": batch_size,
        "stratum_order": [
            asdict(stratum) for stratum in STRATUM_SPEC
        ],
        "chunks": pairing_layout,
    }
    layout_sha256 = hashlib.sha256(
        json.dumps(
            layout_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": GAME_RETURNS_SCHEMA_VERSION,
        "kind": "scacchi.hex_balanced_eval_game_returns",
        "seed": seed,
        "games": games,
        "games_per_stratum": games_per_stratum,
        "batch_size": batch_size,
        "num_chunks_per_stratum": num_chunks,
        "returns_encoding": {
            "-1": "candidate loss",
            "0": "draw",
            "1": "candidate win",
        },
        "pairing_layout_sha256": layout_sha256,
        "pairing_contract": {
            "canonical_order": (
                "stratum_index, then chunk_index, then row_index"
            ),
            "stratum_index": (
                "index into the result's fixed stratum_order"
            ),
            "run_key_index_formula": (
                "stratum_index * num_chunks_per_stratum + chunk_index"
            ),
            "global_game_index_formula": (
                "stratum_index * games_per_stratum + "
                "chunk_index * batch_size + row_index"
            ),
            "row_rng_semantics": (
                "Each chunk uses the recorded rng_key_data. evaluate_chunk "
                "splits it into loop/init keys; env.init row order is the "
                "order returned by split(init_key, batch_size)."
            ),
            "valid_pair_requirement": (
                "pairing_layout_sha256 must match, and candidate/baseline "
                "checkpoints plus every search/evaluation setting other than "
                "the deliberately tested factor must be identical"
            ),
            "paired_analysis": (
                "Align by global_game_index. For McNemar use return>0; for "
                "paired deltas subtract aligned returns or win indicators; "
                "for bootstrap resample aligned game indices within each "
                "stratum."
            ),
        },
        "strata": serialized_strata,
    }
    flat_returns = flatten_game_returns(payload)
    payload["overall"] = summarize_returns(flat_returns)
    payload["returns_sha256"] = hashlib.sha256(
        json.dumps(
            flat_returns,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def validate_evaluation_shape(games: int, batch_size: int) -> int:
    if games <= 0:
        raise ValueError("--games must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if games % len(STRATUM_SPEC):
        raise ValueError("--games must be divisible by four")
    games_per_stratum = games // len(STRATUM_SPEC)
    if games_per_stratum % batch_size:
        raise ValueError(
            "games per stratum must be divisible by --batch-size"
        )
    return games_per_stratum


def stratum_player_order(stratum: Stratum) -> tuple[int, int]:
    """Return PGX's ``colour -> logical player id`` mapping."""

    if stratum.candidate_player_id not in (0, 1):
        raise ValueError("candidate player id must be zero or one")
    expected_seat = "first" if stratum.candidate_first else "second"
    if stratum.candidate_seat != expected_seat:
        raise ValueError("candidate_first and candidate_seat disagree")
    candidate_id = stratum.candidate_player_id
    if stratum.candidate_first:
        return (candidate_id, 1 - candidate_id)
    return (1 - candidate_id, candidate_id)


def select_checkpoint_step(
    checkpoint_dir: Path,
    available_steps: Sequence[int],
    requested_step: int | None,
) -> CheckpointSelection:
    resolved = checkpoint_dir.resolve()
    available = tuple(sorted({int(step) for step in available_steps}))
    if not available:
        raise FileNotFoundError(f"No checkpoint found in {resolved}")
    selected = available[-1] if requested_step is None else int(requested_step)
    if selected not in set(available):
        raise FileNotFoundError(
            f"checkpoint step {selected} not found in {resolved}; "
            f"available steps: {list(available)}"
        )
    return CheckpointSelection(
        directory=resolved,
        requested_step=requested_step,
        selected_step=selected,
        available_steps=available,
    )


def checkpoint_steps(checkpoint_dir: Path) -> tuple[int, ...]:
    """List complete-looking retained Orbax steps without choosing a latest."""

    resolved = checkpoint_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {resolved}")
    return tuple(
        sorted(
            int(child.name)
            for child in resolved.iterdir()
            if (
                child.is_dir()
                and child.name.isdigit()
                and (child / "_CHECKPOINT_METADATA").is_file()
                and (child / "meta" / "metadata").is_file()
            )
        )
    )


def load_checkpoint_metadata(
    checkpoint_dir: Path,
    requested_step: int | None,
) -> LoadedCheckpointMetadata:
    """Load metadata from exactly ``requested_step`` (or the latest if None)."""

    resolved = checkpoint_dir.resolve()
    selection = select_checkpoint_step(
        resolved,
        checkpoint_steps(resolved),
        requested_step,
    )
    metadata_path = (
        resolved / str(selection.selected_step) / "meta" / "metadata"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stored_step = int(metadata.get("step", selection.selected_step))
    if stored_step != selection.selected_step:
        raise ValueError(
            f"checkpoint directory step {selection.selected_step} stores "
            f"metadata step {stored_step}"
        )
    config = checkpoint_io._load_checkpoint_config(metadata["config"])
    return LoadedCheckpointMetadata(
        selection=selection,
        metadata=metadata,
        config=config,
    )


def load_model_at_step(
    loaded: LoadedCheckpointMetadata,
    env: Any,
    *,
    rng_seed: int = 0,
) -> nnx.Module:
    """Build and restore weights from the already-validated exact step."""

    model = build_model(
        loaded.config,
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        rngs=nnx.Rngs(rng_seed),
    )
    options = checkpoint_io._checkpoint_manager_options(read_only=True)
    selection = loaded.selection
    with checkpoint_io.ocp.CheckpointManager(
        str(selection.directory),
        options=options,
    ) as manager:
        available = set(int(step) for step in manager.all_steps())
        if selection.selected_step not in available:
            raise FileNotFoundError(
                f"checkpoint step {selection.selected_step} disappeared "
                f"from {selection.directory}"
            )
        restored = manager.restore(
            selection.selected_step,
            args=checkpoint_io.ocp.args.Composite(
                model=checkpoint_io.ocp.args.StandardRestore(
                    nnx.state(model)
                )
            ),
        )
    nnx.update(model, restored["model"])
    return model


def override_candidate_root_action_estimator(
    config: Any,
    requested_estimator: str | None,
) -> tuple[Any, dict[str, Any]]:
    """Optionally replace only the candidate evaluation root action readout.

    The override is deliberately separate from
    ``root_policy_target_estimator``: that field changes a loss target while
    retaining native winner-MC commitment, and therefore cannot implement a
    paired action/readout evaluation.
    """

    search = config.eval.player_search
    if search.kind != SearchKind.dirichlet_thompson:
        if requested_estimator is not None:
            raise ValueError(
                "--candidate-root-action-estimator requires candidate "
                "eval.player_search.kind=dirichlet_thompson"
            )
        return config, {
            "requested_override": None,
            "field_present": False,
            "stored": None,
            "effective": None,
            "checkpoint_exact": True,
            "note": "candidate evaluation search is not Dirichlet-Thompson",
        }

    active = search.dirichlet_thompson
    field_present = hasattr(active, "root_action_estimator")
    stored = (
        str(getattr(active, "root_action_estimator"))
        if field_present
        else None
    )
    if requested_estimator is None:
        return config, {
            "requested_override": None,
            "field_present": field_present,
            "stored": stored,
            "effective": (
                stored if field_present else "winner_mc (implicit)"
            ),
            "checkpoint_exact": True,
        }
    if not field_present:
        raise ValueError(
            "--candidate-root-action-estimator was requested, but this "
            "Scacchi build does not expose "
            "DirichletThompsonSearchConfig.root_action_estimator. "
            "root_policy_target_estimator is intentionally not substituted."
        )

    estimator = PosteriorPolicyEstimator(requested_estimator)
    effective_active = replace(
        active,
        root_action_estimator=estimator,
    )
    effective_search = replace(
        search,
        dirichlet_thompson=effective_active,
    )
    effective_config = replace(
        config,
        eval=replace(
            config.eval,
            player_search=effective_search,
        ),
    )
    return effective_config, {
        "requested_override": str(estimator),
        "field_present": True,
        "stored": stored,
        "effective": str(estimator),
        "checkpoint_exact": str(estimator) == stored,
        "scope": "candidate eval.player_search root action readout only",
    }


def override_candidate_kappa(
    config: Any,
    requested_kappa: float | None,
) -> tuple[Any, dict[str, Any]]:
    """Optionally replace only the candidate evaluation repair prior mass."""

    search = config.eval.player_search
    if search.kind != SearchKind.dirichlet_thompson:
        if requested_kappa is not None:
            raise ValueError(
                "--candidate-kappa requires candidate "
                "eval.player_search.kind=dirichlet_thompson"
            )
        return config, {
            "requested_override": None,
            "field_present": False,
            "stored": None,
            "effective": None,
            "checkpoint_exact": True,
            "note": "candidate evaluation search is not Dirichlet-Thompson",
        }

    active = search.dirichlet_thompson
    field_present = hasattr(active, "kappa")
    stored = float(getattr(active, "kappa")) if field_present else None
    if requested_kappa is None:
        return config, {
            "requested_override": None,
            "field_present": field_present,
            "stored": stored,
            "effective": stored,
            "checkpoint_exact": True,
        }
    if not field_present:
        raise ValueError(
            "--candidate-kappa was requested, but this Scacchi build does "
            "not expose DirichletThompsonSearchConfig.kappa"
        )
    if not math.isfinite(requested_kappa) or requested_kappa <= 0.0:
        raise ValueError(
            "--candidate-kappa must be finite and positive; "
            f"got {requested_kappa}"
        )

    effective_active = replace(active, kappa=float(requested_kappa))
    effective_search = replace(
        search,
        dirichlet_thompson=effective_active,
    )
    effective_config = replace(
        config,
        eval=replace(
            config.eval,
            player_search=effective_search,
        ),
    )
    return effective_config, {
        "requested_override": float(requested_kappa),
        "field_present": True,
        "stored": stored,
        "effective": float(requested_kappa),
        "checkpoint_exact": float(requested_kappa) == stored,
        "scope": (
            "candidate eval.player_search.dirichlet_thompson.kappa only"
        ),
        "meaning": "prior mass in gamma = n / (kappa + n)",
    }


def override_candidate_prefix_cdf_half_width(
    config: Any,
    requested_half_width: int | None,
) -> tuple[Any, dict[str, Any]]:
    """Optionally replace only the candidate evaluation prefix-CDF grid."""

    search = config.eval.player_search
    if search.kind != SearchKind.dirichlet_thompson:
        if requested_half_width is not None:
            raise ValueError(
                "--candidate-prefix-cdf-half-width requires candidate "
                "eval.player_search.kind=dirichlet_thompson"
            )
        return config, {
            "requested_override": None,
            "field_present": False,
            "stored": None,
            "effective": None,
            "effective_grid_points": None,
            "checkpoint_exact": True,
            "note": "candidate evaluation search is not Dirichlet-Thompson",
        }

    active = search.dirichlet_thompson
    field_present = hasattr(active, "prefix_cdf_half_width")
    stored = (
        int(getattr(active, "prefix_cdf_half_width"))
        if field_present
        else None
    )
    if requested_half_width is None:
        return config, {
            "requested_override": None,
            "field_present": field_present,
            "stored": stored,
            "effective": stored,
            "effective_grid_points": (
                2 * stored + 1 if stored is not None else None
            ),
            "checkpoint_exact": True,
        }
    if not field_present:
        raise ValueError(
            "--candidate-prefix-cdf-half-width was requested, but this "
            "Scacchi build does not expose "
            "DirichletThompsonSearchConfig.prefix_cdf_half_width"
        )
    if requested_half_width < 1:
        raise ValueError(
            "--candidate-prefix-cdf-half-width must be positive; "
            f"got {requested_half_width}"
        )

    effective_active = replace(
        active,
        prefix_cdf_half_width=int(requested_half_width),
    )
    effective_search = replace(
        search,
        dirichlet_thompson=effective_active,
    )
    effective_config = replace(
        config,
        eval=replace(
            config.eval,
            player_search=effective_search,
        ),
    )
    return effective_config, {
        "requested_override": int(requested_half_width),
        "field_present": True,
        "stored": stored,
        "effective": int(requested_half_width),
        "effective_grid_points": 2 * int(requested_half_width) + 1,
        "checkpoint_exact": int(requested_half_width) == stored,
        "scope": (
            "candidate eval.player_search.dirichlet_thompson."
            "prefix_cdf_half_width only"
        ),
    }


def make_balanced_evaluator(
    env: Any,
    config: Any,
    baseline_model: nnx.Module,
    batch_size: int,
    *,
    trace_empty_counts: Sequence[int] = (),
):
    candidate_search_config = config.eval.player_search
    candidate_commitment = config.eval.player_action_commitment_type
    opponent_search_config = baseline_search_config(config)
    opponent_commitment = config.eval.baseline_action_commitment_type
    q_weight_mode = str(config.training.losses.q_loss_weight_mode)
    trace_counts = tuple(int(count) for count in trace_empty_counts)
    trace_counts_array = jnp.asarray(trace_counts, dtype=jnp.int32)
    trace_count = len(trace_counts)
    board_cells = int(config.env.board_size) ** 2

    def search_player(model: Any, search_config: Any, commitment: Any):
        return make_search_player(
            env,
            model,
            search_config,
            commitment,
            q_loss_weight_mode=q_weight_mode,
        )

    @nnx.jit
    def evaluate_chunk(
        rng_key: jax.Array,
        candidate_model: nnx.Module,
        candidate_id: jax.Array,
        candidate_first: jax.Array,
    ) -> BalancedChunkOutput:
        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, batch_size)
        state = jax.vmap(env.init)(init_keys)

        # PGX Hex uses _player_order[colour] -> logical player id and colour 0
        # moves first.  Override env.init's random permutation so each call is
        # one exact identity/seat stratum.
        candidate_id = jnp.asarray(candidate_id, dtype=jnp.int32)
        first_order = jnp.stack((candidate_id, 1 - candidate_id))
        second_order = jnp.stack((1 - candidate_id, candidate_id))
        player_order = jnp.where(
            candidate_first,
            first_order,
            second_order,
        )
        player_order = jnp.broadcast_to(player_order, (batch_size, 2))
        state = state.replace(
            _player_order=player_order,
            current_player=player_order[:, 0],
        )

        candidate_player = search_player(
            candidate_model,
            candidate_search_config,
            candidate_commitment,
        )
        opponent_player = search_player(
            baseline_model,
            opponent_search_config,
            opponent_commitment,
        )
        returns = jnp.zeros((batch_size,), dtype=jnp.float32)
        trace = ChunkTrace(
            valid=jnp.zeros((batch_size, trace_count), dtype=jnp.bool_),
            cells=jnp.zeros(
                (batch_size, trace_count, board_cells),
                dtype=jnp.int8,
            ),
            action=jnp.full(
                (batch_size, trace_count),
                -1,
                dtype=jnp.int32,
            ),
            current_color=jnp.full(
                (batch_size, trace_count),
                -1,
                dtype=jnp.int8,
            ),
            actor_player_id=jnp.full(
                (batch_size, trace_count),
                -1,
                dtype=jnp.int8,
            ),
        )

        def body_fn(
            carry: tuple[jax.Array, Any, jax.Array, ChunkTrace],
        ):
            loop_key, loop_state, loop_returns, loop_trace = carry
            loop_key, candidate_key, opponent_key = jax.random.split(
                loop_key,
                3,
            )
            candidate_output = candidate_player(
                loop_state,
                candidate_key,
            )
            opponent_output = opponent_player(
                loop_state,
                opponent_key,
            )
            action = jnp.where(
                loop_state.current_player == candidate_id,
                candidate_output.action,
                opponent_output.action,
            )

            if trace_count:
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
                empty_count = jnp.sum(raw_board == 0, axis=-1)
                capture = (
                    (empty_count[:, None] == trace_counts_array[None, :])
                    & ~loop_state.terminated[:, None]
                    & ~loop_trace.valid
                )
                loop_trace = ChunkTrace(
                    valid=loop_trace.valid | capture,
                    cells=jnp.where(
                        capture[..., None],
                        fixed_cells[:, None, :],
                        loop_trace.cells,
                    ),
                    action=jnp.where(
                        capture,
                        action[:, None],
                        loop_trace.action,
                    ),
                    current_color=jnp.where(
                        capture,
                        current_color[:, None],
                        loop_trace.current_color,
                    ).astype(jnp.int8),
                    actor_player_id=jnp.where(
                        capture,
                        loop_state.current_player[:, None],
                        loop_trace.actor_player_id,
                    ).astype(jnp.int8),
                )

            loop_state = jax.vmap(env.step)(loop_state, action)
            reward = loop_state.rewards[
                jnp.arange(batch_size),
                candidate_id,
            ]
            return (
                loop_key,
                loop_state,
                loop_returns + reward,
                loop_trace,
            )

        _, _, returns, trace = nnx.while_loop(
            lambda carry: ~carry[1].terminated.all(),
            body_fn,
            (key, state, returns, trace),
        )
        return BalancedChunkOutput(returns=returns, trace=trace)

    return evaluate_chunk


def trace_records_from_chunk(
    trace: ChunkTrace,
    returns: Sequence[int],
    *,
    trace_empty_counts: Sequence[int],
    stratum: Stratum,
    stratum_index: int,
    chunk_index: int,
) -> list[TraceRecord]:
    host_trace = jax.device_get(trace)
    valid = host_trace.valid.tolist()
    cells = host_trace.cells.tolist()
    actions = host_trace.action.tolist()
    current_colors = host_trace.current_color.tolist()
    actor_ids = host_trace.actor_player_id.tolist()
    records: list[TraceRecord] = []
    for row_index, final_return in enumerate(returns):
        for target_index, empty_count in enumerate(trace_empty_counts):
            if not valid[row_index][target_index]:
                continue
            record_cells = tuple(
                int(value)
                for value in cells[row_index][target_index]
            )
            action = int(actions[row_index][target_index])
            if record_cells.count(0) != int(empty_count):
                raise ValueError(
                    "captured board has the wrong number of empty cells"
                )
            if not 0 <= action < len(record_cells):
                raise ValueError(
                    f"captured Hex action {action} is outside the board"
                )
            if record_cells[action] != 0:
                raise ValueError(
                    f"captured Hex action {action} is not legal"
                )
            actor_id = int(actor_ids[row_index][target_index])
            records.append(
                TraceRecord(
                    empty_count=int(empty_count),
                    cells=record_cells,
                    current_color=int(
                        current_colors[row_index][target_index]
                    ),
                    action=action,
                    actor_player_id=actor_id,
                    actor_agent=(
                        "candidate"
                        if actor_id == stratum.candidate_player_id
                        else "baseline"
                    ),
                    candidate_player_id=stratum.candidate_player_id,
                    candidate_seat=stratum.candidate_seat,
                    final_candidate_return=int(final_return),
                    stratum_index=stratum_index,
                    chunk_index=chunk_index,
                    row_index=row_index,
                )
            )
    return records


def select_trace_records(
    records: Sequence[TraceRecord],
    *,
    empty_counts: Sequence[int],
    per_empty_limit: int,
    seed: int,
) -> tuple[list[TraceRecord], dict[str, dict[str, int]]]:
    """Deterministically select unique positions by SHA-256 priority."""

    if per_empty_limit <= 0:
        raise ValueError("--trace-per-empty must be positive")
    selected: list[TraceRecord] = []
    statistics: dict[str, dict[str, int]] = {}
    for empty_count in empty_counts:
        eligible = [
            record
            for record in records
            if record.empty_count == int(empty_count)
        ]
        ordered = sorted(
            eligible,
            key=lambda record: (
                record.sampling_priority(seed),
                record.stratum_index,
                record.chunk_index,
                record.row_index,
            ),
        )
        unique: list[TraceRecord] = []
        seen_states: set[str] = set()
        for record in ordered:
            if record.state_key in seen_states:
                continue
            seen_states.add(record.state_key)
            unique.append(record)
        retained = unique[:per_empty_limit]
        selected.extend(retained)
        statistics[str(int(empty_count))] = {
            "eligible": len(eligible),
            "unique_positions": len(unique),
            "selected": len(retained),
        }
    selected.sort(
        key=lambda record: (
            record.empty_count,
            record.sampling_priority(seed),
        )
    )
    return selected, statistics


def build_trace_payload(
    records: Sequence[TraceRecord],
    *,
    empty_counts: Sequence[int],
    per_empty_limit: int,
    seed: int,
    board_size: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    selected, statistics = select_trace_records(
        records,
        empty_counts=empty_counts,
        per_empty_limit=per_empty_limit,
        seed=seed,
    )
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "kind": "scacchi.hex_balanced_eval_trace",
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
        },
        "source": source,
        "board_encoding": {
            "layout": "row-major",
            "cell_values": {
                "0": "empty",
                "1": "Hex colour 0 (first seat)",
                "2": "Hex colour 1 (second seat)",
            },
            "current_color": (
                "Hex colour to move; 0 is first and 1 is second"
            ),
            "action": "zero-based row-major cell index",
            "capture_point": (
                "non-terminal state immediately before the recorded action"
            ),
        },
        "sampling": {
            "seed": seed,
            "requested_empty_counts": [
                int(count) for count in empty_counts
            ],
            "per_empty_limit": per_empty_limit,
            "maximum_supported_empty_count": MAX_TRACE_EMPTY_COUNT,
            "method": (
                "Within each empty count, deduplicate (cells,current_color), "
                "then retain the lexicographically smallest SHA-256 priorities "
                "derived from seed and source coordinates."
            ),
            "counts": statistics,
        },
        "positions": [
            record.to_json(seed=seed, board_size=board_size)
            for record in selected
        ],
    }


def _parse_empty_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(
            sorted(
                {
                    int(part.strip())
                    for part in value.split(",")
                    if part.strip()
                }
            )
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "empty counts must be comma-separated integers"
        ) from error
    if not counts:
        raise argparse.ArgumentTypeError(
            "expected at least one empty count"
        )
    if any(
        count < 1 or count > MAX_TRACE_EMPTY_COUNT
        for count in counts
    ):
        raise argparse.ArgumentTypeError(
            "trace empty counts must be between 1 and "
            f"{MAX_TRACE_EMPTY_COUNT}"
        )
    return counts


def _parse_positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a floating-point number"
        ) from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be finite and positive"
        )
    return parsed


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one exact Hex checkpoint step in four equally weighted "
            "logical-id x seat strata, without network logging."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("checkpoints/6_solved"),
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="exact retained candidate checkpoint step",
    )
    parser.add_argument(
        "--baseline-step",
        type=int,
        default=None,
        help="exact baseline step (default: latest, recorded in provenance)",
    )
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=5504096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write the full artifact but print only its path and digest",
    )
    parser.add_argument(
        "--include-game-returns",
        action="store_true",
        help=(
            "include ordered per-game returns and RNG pairing provenance "
            "for paired statistical analysis"
        ),
    )
    parser.add_argument(
        "--candidate-root-action-estimator",
        choices=tuple(str(value) for value in PosteriorPolicyEstimator),
        default=None,
        help=(
            "optional candidate DT root action-readout override; omitted "
            "means use the checkpoint setting exactly"
        ),
    )
    parser.add_argument(
        "--candidate-kappa",
        type=_parse_positive_finite_float,
        default=None,
        help=(
            "optional candidate DT repair-prior mass override; omitted "
            "means use the checkpoint setting exactly"
        ),
    )
    parser.add_argument(
        "--candidate-prefix-cdf-half-width",
        type=int,
        default=None,
        help=(
            "optional candidate DT prefix-CDF half-width override; Q uses "
            "2*half_width+1 grid points (Q21 is half-width 10)"
        ),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help="optional local JSON path for sampled pre-action positions",
    )
    parser.add_argument(
        "--trace-empty-counts",
        type=_parse_empty_counts,
        default=None,
        help=(
            "comma-separated empty-cell counts in [1,15]; required with "
            "--trace-output"
        ),
    )
    parser.add_argument(
        "--trace-per-empty",
        type=int,
        default=32,
        help="maximum unique trace positions retained per empty count",
    )
    parser.add_argument(
        "--trace-seed",
        type=int,
        default=None,
        help="sampling seed (default: --seed); does not affect game RNG",
    )
    return parser


def _search_summary(search: Any) -> dict[str, Any]:
    active = search.active()
    summary: dict[str, Any] = {
        "kind": str(search.kind),
        "num_simulations": int(active.num_simulations),
    }
    policy_samples = getattr(active, "policy_samples", None)
    if policy_samples is not None:
        summary["policy_samples"] = int(policy_samples)
    kappa = getattr(active, "kappa", None)
    if kappa is not None:
        summary["kappa"] = float(kappa)
    prefix_cdf_half_width = getattr(
        active,
        "prefix_cdf_half_width",
        None,
    )
    if prefix_cdf_half_width is not None:
        summary["prefix_cdf_half_width"] = int(prefix_cdf_half_width)
        summary["prefix_cdf_grid_points"] = (
            2 * int(prefix_cdf_half_width) + 1
        )
    for field_name in (
        "posterior_policy_estimator",
        "root_policy_target_estimator",
        "root_action_estimator",
    ):
        value = getattr(active, field_name, None)
        if value is not None:
            summary[field_name] = str(value)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    games_per_stratum = validate_evaluation_shape(
        args.games,
        args.batch_size,
    )
    if args.trace_output is None and args.trace_empty_counts is not None:
        parser.error("--trace-empty-counts requires --trace-output")
    if args.trace_output is not None and args.trace_empty_counts is None:
        parser.error("--trace-output requires --trace-empty-counts")
    if args.trace_per_empty <= 0:
        parser.error("--trace-per-empty must be positive")
    if (
        args.candidate_prefix_cdf_half_width is not None
        and args.candidate_prefix_cdf_half_width <= 0
    ):
        parser.error(
            "--candidate-prefix-cdf-half-width must be positive"
        )
    if (
        args.trace_output is not None
        and args.trace_output.resolve() == args.output.resolve()
    ):
        parser.error("--trace-output and --output must be different paths")

    candidate = load_checkpoint_metadata(args.candidate, args.step)
    baseline = load_checkpoint_metadata(
        args.baseline,
        args.baseline_step,
    )
    config, root_action_override = (
        override_candidate_root_action_estimator(
            candidate.config,
            args.candidate_root_action_estimator,
        )
    )
    config, kappa_override = override_candidate_kappa(
        config,
        args.candidate_kappa,
    )
    config, prefix_cdf_grid_override = (
        override_candidate_prefix_cdf_half_width(
            config,
            args.candidate_prefix_cdf_half_width,
        )
    )
    effective_candidate = replace(candidate, config=config)
    if config.env.id != "hex" or config.env.board_size is None:
        raise ValueError(
            "candidate checkpoint must describe a fixed-size Hex environment"
        )
    if (
        baseline.config.env.id != config.env.id
        or baseline.config.env.board_size != config.env.board_size
    ):
        raise ValueError(
            "candidate/baseline environment mismatch: "
            f"{config.env.id}/{config.env.board_size} != "
            f"{baseline.config.env.id}/{baseline.config.env.board_size}"
        )

    env = make_env(config.env.id, config.env.board_size)
    candidate_model = load_model_at_step(effective_candidate, env)
    baseline_model = load_model_at_step(baseline, env)
    trace_empty_counts = args.trace_empty_counts or ()
    evaluate_chunk = make_balanced_evaluator(
        env,
        config,
        baseline_model,
        args.batch_size,
        trace_empty_counts=trace_empty_counts,
    )

    num_chunks = games_per_stratum // args.batch_size
    master_key = jax.random.PRNGKey(args.seed)
    run_keys = jax.random.split(
        master_key,
        len(STRATUM_SPEC) * num_chunks,
    )
    key_index = 0
    all_returns: list[int] = []
    returns_by_stratum: dict[tuple[int, str], list[int]] = {}
    returns_by_chunk: list[list[list[int]]] = []
    strata: list[dict[str, Any]] = []
    trace_records: list[TraceRecord] = []
    started = time.perf_counter()

    for stratum_index, stratum in enumerate(STRATUM_SPEC):
        stratum_returns: list[int] = []
        stratum_chunk_returns: list[list[int]] = []
        stratum_started = time.perf_counter()
        for chunk_index in range(num_chunks):
            chunk_started = time.perf_counter()
            output = evaluate_chunk(
                run_keys[key_index],
                candidate_model,
                jnp.asarray(
                    stratum.candidate_player_id,
                    dtype=jnp.int32,
                ),
                jnp.asarray(stratum.candidate_first),
            )
            host_returns = (
                jax.device_get(output.returns).astype(int).tolist()
            )
            key_index += 1
            stratum_returns.extend(host_returns)
            stratum_chunk_returns.append(host_returns)
            if trace_empty_counts:
                trace_records.extend(
                    trace_records_from_chunk(
                        output.trace,
                        host_returns,
                        trace_empty_counts=trace_empty_counts,
                        stratum=stratum,
                        stratum_index=stratum_index,
                        chunk_index=chunk_index,
                    )
                )
            elapsed = time.perf_counter() - chunk_started
            print(
                f"id={stratum.candidate_player_id} "
                f"seat={stratum.candidate_seat} "
                f"chunk={chunk_index + 1}/{num_chunks} "
                f"wins={sum(value > 0 for value in host_returns)}/"
                f"{len(host_returns)} seconds={elapsed:.3f}",
                flush=True,
            )
        summary = summarize_returns(
            stratum_returns,
            candidate_id=stratum.candidate_player_id,
            seat=stratum.candidate_seat,
        )
        summary["seconds"] = time.perf_counter() - stratum_started
        strata.append(summary)
        returns_by_stratum[
            (stratum.candidate_player_id, stratum.candidate_seat)
        ] = stratum_returns
        returns_by_chunk.append(stratum_chunk_returns)
        all_returns.extend(stratum_returns)

    total_seconds = time.perf_counter() - started
    overall = summarize_returns(all_returns)
    game_returns_payload = (
        build_game_returns_payload(
            returns_by_chunk,
            run_keys=run_keys,
            seed=args.seed,
            games=args.games,
            games_per_stratum=games_per_stratum,
            batch_size=args.batch_size,
        )
        if args.include_game_returns
        else None
    )
    if (
        game_returns_payload is not None
        and game_returns_payload["overall"] != overall
    ):
        raise AssertionError(
            "serialized per-game returns disagree with overall summary"
        )
    marginals = {
        "candidate_player_id": [
            summarize_returns(
                returns_by_stratum[(candidate_id, "first")]
                + returns_by_stratum[(candidate_id, "second")],
                candidate_id=candidate_id,
            )
            for candidate_id in (0, 1)
        ],
        "candidate_seat": [
            summarize_returns(
                returns_by_stratum[(0, seat)]
                + returns_by_stratum[(1, seat)],
                seat=seat,
            )
            for seat in ("first", "second")
        ],
    }

    script_path = Path(__file__).resolve()
    script_sha256 = hashlib.sha256(
        script_path.read_bytes()
    ).hexdigest()
    reproduction_argv = list(sys.argv if argv is None else [sys.argv[0], *argv])
    trace_seed = args.seed if args.trace_seed is None else args.trace_seed
    trace_capture: dict[str, Any] = {
        "enabled": bool(trace_empty_counts),
        "game_rng_impact": (
            "none; capture consumes no PRNG keys and sampling occurs "
            "after evaluation"
        ),
    }
    if trace_empty_counts:
        trace_source = {
            "candidate_checkpoint": candidate.selection.provenance(),
            "baseline_checkpoint": baseline.selection.provenance(),
            "candidate_root_action_estimator": root_action_override,
            "candidate_kappa": kappa_override,
            "candidate_prefix_cdf_grid": prefix_cdf_grid_override,
            "evaluation_seed": args.seed,
            "games": args.games,
            "batch_size": args.batch_size,
            "strata": [asdict(stratum) for stratum in STRATUM_SPEC],
            "script_path": str(script_path),
            "script_sha256": script_sha256,
        }
        trace_payload = build_trace_payload(
            trace_records,
            empty_counts=trace_empty_counts,
            per_empty_limit=args.trace_per_empty,
            seed=trace_seed,
            board_size=config.env.board_size,
            source=trace_source,
        )
        trace_sha256 = _write_json(args.trace_output, trace_payload)
        trace_capture.update(
            {
                "output": str(args.trace_output.resolve()),
                "sha256": trace_sha256,
                "empty_counts": list(trace_empty_counts),
                "per_empty_limit": args.trace_per_empty,
                "seed": trace_seed,
                "eligible_records": len(trace_records),
                "selected_records": len(trace_payload["positions"]),
                "counts": trace_payload["sampling"]["counts"],
            }
        )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
        },
        "reproduction": {
            "command": " ".join(
                shlex.quote(argument)
                for argument in reproduction_argv
            ),
            "working_directory": str(Path.cwd().resolve()),
            "script_path": str(script_path),
            "script_sha256": script_sha256,
        },
        # Preserve the temporary v2 harness fields while adding unambiguous
        # selection provenance for non-latest restores.
        "candidate_checkpoint": str(candidate.selection.directory),
        "candidate_step": candidate.selection.selected_step,
        "candidate_available_steps": list(
            candidate.selection.available_steps
        ),
        "candidate_checkpoint_metadata": candidate.metadata,
        "candidate_checkpoint_selection": (
            candidate.selection.provenance()
        ),
        "candidate_root_action_estimator": root_action_override,
        "candidate_kappa": kappa_override,
        "candidate_prefix_cdf_grid": prefix_cdf_grid_override,
        "baseline_checkpoint": str(baseline.selection.directory),
        "baseline_step": baseline.selection.selected_step,
        "baseline_available_steps": list(
            baseline.selection.available_steps
        ),
        "baseline_checkpoint_metadata": baseline.metadata,
        "baseline_checkpoint_selection": (
            baseline.selection.provenance()
        ),
        "games": args.games,
        "games_per_stratum": games_per_stratum,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "environment": {
            "id": config.env.id,
            "board_size": config.env.board_size,
            "pgx_player_order_semantics": (
                "_player_order[colour] maps colour/seat to logical player id; "
                "colour 0 moves first"
            ),
        },
        "role_and_rng_construction": {
            "stratum_order": [
                asdict(stratum) for stratum in STRATUM_SPEC
            ],
            "player_order_by_stratum": [
                list(stratum_player_order(stratum))
                for stratum in STRATUM_SPEC
            ],
            "role_balance": (
                "Exactly equal cells for candidate logical PGX id {0,1} x "
                "candidate colour/seat {first=0,second=1}."
            ),
            "role_forcing": (
                "After env.init, overwrite _player_order so "
                "_player_order[candidate colour] equals candidate_player_id; "
                "set current_player=_player_order[0]."
            ),
            "search_key_assignment": (
                "Candidate always receives the candidate/player_1 split key "
                "and the solved opponent always receives the opponent/player_2 "
                "split key, independent of logical id and seat."
            ),
            "rng": (
                "jax.random.split(PRNGKey(seed), 4*num_chunks) supplies one "
                "independent key per chunk in stratum_order. Each chunk mirrors "
                "play_eval: split once into loop/init keys, then each ply splits "
                "loop key into next/candidate/opponent keys."
            ),
        },
        "monitor": {
            "candidate_search": _search_summary(
                config.eval.player_search
            ),
            "candidate_action_commitment": str(
                config.eval.player_action_commitment_type
            ),
            "opponent_search": _search_summary(
                baseline_search_config(config)
            ),
            "opponent_action_commitment": str(
                config.eval.baseline_action_commitment_type
            ),
            "equivalence_to_training_monitor": (
                "Same models, search configs, action commitments, q-loss mode, "
                "and per-ply PRNG split as scacchi.play.play_eval; only PGX "
                "logical-id/seat assignment is fixed instead of randomized."
            ),
        },
        "confidence_interval": {
            "method": "Wilson score interval",
            "nominal_coverage": 0.95,
            "sidedness": "two-sided",
            "z": 1.959963984540054,
            "success_definition": "candidate return > 0",
            "note": (
                "Reported separately for every fixed stratum and for the pooled "
                "exactly balanced sample. With equal stratum sizes, the pooled "
                "win rate is the intended four-cell balanced estimand."
            ),
        },
        "trace_capture": trace_capture,
        "runtime": {
            "seconds": total_seconds,
            "games_per_second": args.games / total_seconds,
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "python": platform.python_version(),
            "jax": jax.__version__,
        },
        "strata": strata,
        "marginals": marginals,
        "overall": overall,
    }
    if game_returns_payload is not None:
        result["game_returns"] = game_returns_payload
    result_sha256 = _write_json(args.output, result)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(
        f"Wrote {args.output.resolve()} sha256={result_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
