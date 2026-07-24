#!/usr/bin/env python3
"""Reproducible, local-only checkpoint cross-play for fixed-size Hex.

One pair evaluation fixes competitor A in each cell of the product

    A logical PGX player id in {0, 1}
        x A Hex seat in {first, second}.

All four cells have equal size.  Competitor B therefore receives every
logical id and seat equally often as well.  Search RNG streams are assigned
to competitor identity rather than PGX id or seat: A always receives the
first per-ply search key and B the second.

The ``matrix`` command consumes a JSON manifest, loads each needed model at
most once, and creates one immutable result per unordered pair.  A resumed
run reuses an existing artifact only after its canonical job-spec hash has
been validated.  No network or external logger is used.

Manifest schema (version 1)::

    {
      "schema_version": 1,
      "kind": "scacchi.hex_checkpoint_league_manifest",
      "output_directory": "artifacts",
      "games": 4096,
      "batch_size": 256,
      "seed": 5504096,
      "include_game_returns": false,
      "competitors": [
        {
          "id": "e8-q21",
          "checkpoint": "../../checkpoints/e8",
          "step": 75,
          "root_action_estimator": "prefix_cdf",
          "prefix_cdf_half_width": 10,
          "kappa": 3.0
        }
      ]
    }

When ``pairs`` is omitted, all unordered pairs are evaluated in roster
order.  An explicit ``pairs`` list contains objects with ``a`` and ``b``
competitor ids and may override ``games``, ``batch_size``, ``seed``,
``include_game_returns``, and ``output``.  Paths in the manifest are
resolved relative to the manifest file; explicit pair outputs are resolved
relative to ``output_directory``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import sys
import tempfile
import time
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp

from scripts import hex_balanced_eval as balanced
from scacchi.envs import make_env
from scacchi.play_search import make_search_player
from scacchi.types import PosteriorPolicyEstimator, SearchKind


MANIFEST_SCHEMA_VERSION = 1
PAIR_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
MANIFEST_KIND = "scacchi.hex_checkpoint_league_manifest"
PAIR_KIND = "scacchi.hex_checkpoint_league_pair"
RUN_KIND = "scacchi.hex_checkpoint_league_run"
GAME_RETURNS_KIND = "scacchi.hex_checkpoint_league_game_returns"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CompetitorSpec:
    competitor_id: str
    checkpoint: Path
    step: int
    root_action_estimator: str | None = None
    prefix_cdf_half_width: int | None = None
    kappa: float | None = None


@dataclass(frozen=True)
class PairSpec:
    competitor_a: str
    competitor_b: str
    games: int
    batch_size: int
    seed: int
    include_game_returns: bool
    output: Path | None = None


@dataclass(frozen=True)
class LeagueManifest:
    path: Path
    file_sha256: str
    output_directory: Path
    competitors: tuple[CompetitorSpec, ...]
    pairs: tuple[PairSpec, ...]


@dataclass(frozen=True)
class PreparedCompetitor:
    spec: CompetitorSpec
    loaded: balanced.LoadedCheckpointMetadata
    effective_config: Any
    overrides: dict[str, Any]
    metadata_sha256: str
    checkpoint_tree_sha256: str
    effective_eval_sha256: str

    @property
    def model_cache_key(self) -> tuple[str, int]:
        return (
            str(self.loaded.selection.directory),
            self.loaded.selection.selected_step,
        )


class LeagueChunkOutput(NamedTuple):
    competitor_a_returns: jax.Array


def canonical_sha256(payload: Any) -> str:
    """Hash JSON-compatible data using a single canonical encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_provenance() -> dict[str, Any]:
    """Hash the local source files that define league game semantics."""

    workspace = Path(__file__).resolve().parent.parent
    paths = (
        Path(__file__).resolve(),
        Path(balanced.__file__).resolve(),
        workspace / "scacchi" / "envs.py",
        workspace / "scacchi" / "play_search.py",
        workspace / "scacchi" / "dirichlet_mctx" / "action_selection.py",
    )
    files = {
        str(path): file_sha256(path)
        for path in paths
    }
    return {
        "files": files,
        "bundle_sha256": canonical_sha256(files),
    }


def tree_sha256(root: Path) -> str:
    """Content-address a directory, including relative file/link names."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"checkpoint step directory not found: {resolved}")
    digest = hashlib.sha256()
    entries = sorted(
        resolved.rglob("*"),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )
    for path in entries:
        relative = path.relative_to(resolved).as_posix()
        if path.is_symlink():
            kind = b"L"
            content = os.readlink(path).encode("utf-8")
        elif path.is_file():
            kind = b"F"
            content = None
        elif path.is_dir():
            kind = b"D"
            content = b""
        else:
            raise ValueError(f"unsupported checkpoint tree entry: {path}")
        encoded_name = relative.encode("utf-8")
        digest.update(kind)
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        if content is not None:
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
            continue
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


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
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _require_int(value: Any, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{location} must be >= {minimum}")
    return value


def _require_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be boolean")
    return value


def _require_identifier(value: Any, location: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{location} must match {IDENTIFIER_PATTERN.pattern!r}"
        )
    return value


def _optional_positive_float(value: Any, location: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{location} must be a finite positive number")
    return result


def _resolve_path(value: Any, base: Path, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_root_estimator(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string or null")
    try:
        return str(PosteriorPolicyEstimator(value))
    except ValueError as error:
        choices = [str(item) for item in PosteriorPolicyEstimator]
        raise ValueError(f"{location} must be one of {choices}") from error


def override_evaluation_search(
    config: Any,
    *,
    root_action_estimator: str | None = None,
    prefix_cdf_half_width: int | None = None,
    kappa: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Override only one competitor's test-time DT readout/repair settings."""

    search = config.eval.player_search
    active = search.active()
    stored_root = getattr(active, "root_action_estimator", None)
    stored_half_width = getattr(active, "prefix_cdf_half_width", None)
    stored_kappa = getattr(active, "kappa", None)
    requested = {
        "root_action_estimator": root_action_estimator,
        "prefix_cdf_half_width": prefix_cdf_half_width,
        "kappa": kappa,
    }
    if (
        root_action_estimator is None
        and prefix_cdf_half_width is None
        and kappa is None
    ):
        return config, {
            "requested": requested,
            "stored": {
                "root_action_estimator": (
                    str(stored_root) if stored_root is not None else None
                ),
                "prefix_cdf_half_width": (
                    int(stored_half_width)
                    if stored_half_width is not None
                    else None
                ),
                "kappa": (
                    float(stored_kappa) if stored_kappa is not None else None
                ),
            },
            "effective": {
                "root_action_estimator": (
                    str(stored_root) if stored_root is not None else None
                ),
                "prefix_cdf_half_width": (
                    int(stored_half_width)
                    if stored_half_width is not None
                    else None
                ),
                "kappa": (
                    float(stored_kappa) if stored_kappa is not None else None
                ),
            },
            "checkpoint_exact": True,
        }
    if search.kind != SearchKind.dirichlet_thompson:
        raise ValueError(
            "root-action-estimator/prefix-grid/kappa overrides require "
            "eval.player_search.kind=dirichlet_thompson"
        )
    if root_action_estimator is not None and not hasattr(
        active,
        "root_action_estimator",
    ):
        raise ValueError(
            "this Scacchi build does not expose "
            "DirichletThompsonSearchConfig.root_action_estimator"
        )
    if kappa is not None and not hasattr(active, "kappa"):
        raise ValueError(
            "this Scacchi build does not expose "
            "DirichletThompsonSearchConfig.kappa"
        )
    if prefix_cdf_half_width is not None and not hasattr(
        active,
        "prefix_cdf_half_width",
    ):
        raise ValueError(
            "this Scacchi build does not expose "
            "DirichletThompsonSearchConfig.prefix_cdf_half_width"
        )

    changes: dict[str, Any] = {}
    if root_action_estimator is not None:
        changes["root_action_estimator"] = PosteriorPolicyEstimator(
            root_action_estimator
        )
    if prefix_cdf_half_width is not None:
        if prefix_cdf_half_width < 1:
            raise ValueError(
                "prefix_cdf_half_width override must be positive"
            )
        changes["prefix_cdf_half_width"] = int(prefix_cdf_half_width)
    if kappa is not None:
        if not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError("kappa override must be finite and positive")
        changes["kappa"] = float(kappa)
    effective_active = replace(active, **changes)
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
    effective_root = getattr(
        effective_active,
        "root_action_estimator",
        None,
    )
    effective_half_width = getattr(
        effective_active,
        "prefix_cdf_half_width",
        None,
    )
    effective_kappa = getattr(effective_active, "kappa", None)
    return effective_config, {
        "requested": requested,
        "stored": {
            "root_action_estimator": (
                str(stored_root) if stored_root is not None else None
            ),
            "prefix_cdf_half_width": (
                int(stored_half_width)
                if stored_half_width is not None
                else None
            ),
            "kappa": (
                float(stored_kappa) if stored_kappa is not None else None
            ),
        },
        "effective": {
            "root_action_estimator": (
                str(effective_root) if effective_root is not None else None
            ),
            "prefix_cdf_half_width": (
                int(effective_half_width)
                if effective_half_width is not None
                else None
            ),
            "prefix_cdf_grid_points": (
                2 * int(effective_half_width) + 1
                if effective_half_width is not None
                else None
            ),
            "kappa": (
                float(effective_kappa) if effective_kappa is not None else None
            ),
        },
        "checkpoint_exact": (
            (
                root_action_estimator is None
                or str(stored_root) == str(effective_root)
            )
            and (
                prefix_cdf_half_width is None
                or (
                    stored_half_width is not None
                    and effective_half_width is not None
                    and int(stored_half_width)
                    == int(effective_half_width)
                )
            )
            and (
                kappa is None
                or (
                    stored_kappa is not None
                    and effective_kappa is not None
                    and float(stored_kappa) == float(effective_kappa)
                )
            )
        ),
        "scope": "this competitor's eval.player_search only",
    }


def prepare_competitor(spec: CompetitorSpec) -> PreparedCompetitor:
    """Validate and hash an exact retained checkpoint without loading weights."""

    loaded = balanced.load_checkpoint_metadata(spec.checkpoint, spec.step)
    if loaded.selection.selection_mode != "exact":
        raise AssertionError("league competitors require exact checkpoint steps")
    effective_config, overrides = override_evaluation_search(
        loaded.config,
        root_action_estimator=spec.root_action_estimator,
        prefix_cdf_half_width=spec.prefix_cdf_half_width,
        kappa=spec.kappa,
    )
    step_root = (
        loaded.selection.directory / str(loaded.selection.selected_step)
    )
    metadata_path = step_root / "meta" / "metadata"
    return PreparedCompetitor(
        spec=spec,
        loaded=loaded,
        effective_config=effective_config,
        overrides=overrides,
        metadata_sha256=file_sha256(metadata_path),
        checkpoint_tree_sha256=tree_sha256(step_root),
        effective_eval_sha256=canonical_sha256(
            {
                "eval": _jsonable(effective_config.eval),
                "q_loss_weight_mode": str(
                    effective_config.training.losses.q_loss_weight_mode
                ),
            }
        ),
    )


def load_prepared_model(
    competitor: PreparedCompetitor,
    env: Any,
) -> nnx.Module:
    effective = replace(
        competitor.loaded,
        config=competitor.effective_config,
    )
    return balanced.load_model_at_step(effective, env)


def make_league_evaluator(
    env: Any,
    config_a: Any,
    config_b: Any,
    batch_size: int,
) -> Callable[..., LeagueChunkOutput]:
    """Build a jitted A-vs-B evaluator with identity-stable RNG assignment."""

    search_a = config_a.eval.player_search
    search_b = config_b.eval.player_search
    commitment_a = config_a.eval.player_action_commitment_type
    commitment_b = config_b.eval.player_action_commitment_type
    q_mode_a = str(config_a.training.losses.q_loss_weight_mode)
    q_mode_b = str(config_b.training.losses.q_loss_weight_mode)

    @nnx.jit
    def evaluate_chunk(
        rng_key: jax.Array,
        model_a: nnx.Module,
        model_b: nnx.Module,
        competitor_a_player_id: jax.Array,
        competitor_a_first: jax.Array,
    ) -> LeagueChunkOutput:
        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, batch_size)
        state = jax.vmap(env.init)(init_keys)

        competitor_a_player_id = jnp.asarray(
            competitor_a_player_id,
            dtype=jnp.int32,
        )
        first_order = jnp.stack(
            (competitor_a_player_id, 1 - competitor_a_player_id)
        )
        second_order = jnp.stack(
            (1 - competitor_a_player_id, competitor_a_player_id)
        )
        player_order = jnp.where(
            competitor_a_first,
            first_order,
            second_order,
        )
        player_order = jnp.broadcast_to(player_order, (batch_size, 2))
        state = state.replace(
            _player_order=player_order,
            current_player=player_order[:, 0],
        )

        player_a = make_search_player(
            env,
            model_a,
            search_a,
            commitment_a,
            q_loss_weight_mode=q_mode_a,
        )
        player_b = make_search_player(
            env,
            model_b,
            search_b,
            commitment_b,
            q_loss_weight_mode=q_mode_b,
        )
        returns = jnp.zeros((batch_size,), dtype=jnp.float32)

        def body_fn(carry: tuple[jax.Array, Any, jax.Array]):
            loop_key, loop_state, loop_returns = carry
            loop_key, key_a, key_b = jax.random.split(loop_key, 3)
            output_a = player_a(loop_state, key_a)
            output_b = player_b(loop_state, key_b)
            action = jnp.where(
                loop_state.current_player == competitor_a_player_id,
                output_a.action,
                output_b.action,
            )
            loop_state = jax.vmap(env.step)(loop_state, action)
            reward = loop_state.rewards[
                jnp.arange(batch_size),
                competitor_a_player_id,
            ]
            return loop_key, loop_state, loop_returns + reward

        _, _, returns = nnx.while_loop(
            lambda carry: ~carry[1].terminated.all(),
            body_fn,
            (key, state, returns),
        )
        return LeagueChunkOutput(competitor_a_returns=returns)

    return evaluate_chunk


def _summary(
    returns: Sequence[int],
    *,
    competitor_id: str,
    logical_player_id: int | None = None,
    seat: str | None = None,
) -> dict[str, Any]:
    result = balanced.summarize_returns(returns)
    result["competitor_id"] = competitor_id
    if logical_player_id is not None:
        result["logical_player_id"] = logical_player_id
    if seat is not None:
        result["seat"] = seat
    return result


def summarize_pair_returns(
    returns_by_stratum: Sequence[Sequence[int]],
    *,
    competitor_a: str,
    competitor_b: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize the canonical A-id x A-seat strata from both perspectives."""

    if len(returns_by_stratum) != len(balanced.STRATUM_SPEC):
        raise ValueError("pair returns must contain exactly four strata")
    sizes = {len(values) for values in returns_by_stratum}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("all four pair strata must be non-empty and equal")
    if any(
        value not in (-1, 0, 1)
        for values in returns_by_stratum
        for value in values
    ):
        raise ValueError("pair returns must be encoded as -1, 0, or 1")

    strata: list[dict[str, Any]] = []
    by_a_key: dict[tuple[int, str], list[int]] = {}
    for index, (stratum, values_raw) in enumerate(
        zip(balanced.STRATUM_SPEC, returns_by_stratum, strict=True)
    ):
        values = [int(value) for value in values_raw]
        b_values = [-value for value in values]
        b_id = 1 - stratum.candidate_player_id
        b_seat = "second" if stratum.candidate_first else "first"
        by_a_key[
            (stratum.candidate_player_id, stratum.candidate_seat)
        ] = values
        strata.append(
            {
                "stratum_index": index,
                "competitor_a_logical_player_id": (
                    stratum.candidate_player_id
                ),
                "competitor_a_first": stratum.candidate_first,
                "competitor_a_seat": stratum.candidate_seat,
                "competitor_b_logical_player_id": b_id,
                "competitor_b_first": not stratum.candidate_first,
                "competitor_b_seat": b_seat,
                "competitor_a": _summary(
                    values,
                    competitor_id=competitor_a,
                    logical_player_id=stratum.candidate_player_id,
                    seat=stratum.candidate_seat,
                ),
                "competitor_b": _summary(
                    b_values,
                    competitor_id=competitor_b,
                    logical_player_id=b_id,
                    seat=b_seat,
                ),
            }
        )

    all_a = [
        value
        for stratum_values in returns_by_stratum
        for value in stratum_values
    ]
    all_b = [-value for value in all_a]
    a_first = by_a_key[(0, "first")] + by_a_key[(1, "first")]
    a_second = by_a_key[(0, "second")] + by_a_key[(1, "second")]
    b_first = [-value for value in a_second]
    b_second = [-value for value in a_first]

    pairwise = {
        "overall": {
            "competitor_a": _summary(
                all_a,
                competitor_id=competitor_a,
            ),
            "competitor_b": _summary(
                all_b,
                competitor_id=competitor_b,
            ),
        },
        "by_seat": {
            "first": {
                "competitor_a": _summary(
                    a_first,
                    competitor_id=competitor_a,
                    seat="first",
                ),
                "competitor_b": _summary(
                    b_first,
                    competitor_id=competitor_b,
                    seat="first",
                ),
            },
            "second": {
                "competitor_a": _summary(
                    a_second,
                    competitor_id=competitor_a,
                    seat="second",
                ),
                "competitor_b": _summary(
                    b_second,
                    competitor_id=competitor_b,
                    seat="second",
                ),
            },
        },
        "by_logical_player_id": {
            str(player_id): {
                "competitor_a": _summary(
                    by_a_key[(player_id, "first")]
                    + by_a_key[(player_id, "second")],
                    competitor_id=competitor_a,
                    logical_player_id=player_id,
                ),
                "competitor_b": _summary(
                    [
                        -value
                        for value in (
                            by_a_key[(1 - player_id, "first")]
                            + by_a_key[(1 - player_id, "second")]
                        )
                    ],
                    competitor_id=competitor_b,
                    logical_player_id=player_id,
                ),
            }
            for player_id in (0, 1)
        },
    }
    pairwise["competitor_a"] = {
        "overall": pairwise["overall"]["competitor_a"],
        "by_seat": {
            seat: pairwise["by_seat"][seat]["competitor_a"]
            for seat in ("first", "second")
        },
        "by_logical_player_id": {
            player_id: pairwise["by_logical_player_id"][player_id][
                "competitor_a"
            ]
            for player_id in ("0", "1")
        },
    }
    pairwise["competitor_b"] = {
        "overall": pairwise["overall"]["competitor_b"],
        "by_seat": {
            seat: pairwise["by_seat"][seat]["competitor_b"]
            for seat in ("first", "second")
        },
        "by_logical_player_id": {
            player_id: pairwise["by_logical_player_id"][player_id][
                "competitor_b"
            ]
            for player_id in ("0", "1")
        },
    }
    return strata, pairwise


def build_league_game_returns(
    returns_by_chunk: Sequence[Sequence[Sequence[int]]],
    *,
    run_keys: Any,
    pair: PairSpec,
) -> dict[str, Any]:
    """Use the balanced evaluator's tested coordinate/pairing serialization."""

    games_per_stratum = pair.games // len(balanced.STRATUM_SPEC)
    payload = balanced.build_game_returns_payload(
        returns_by_chunk,
        run_keys=run_keys,
        seed=pair.seed,
        games=pair.games,
        games_per_stratum=games_per_stratum,
        batch_size=pair.batch_size,
    )
    payload["kind"] = GAME_RETURNS_KIND
    payload["perspective"] = {
        "returns": "competitor_a",
        "coordinate_field_alias": {
            "candidate_player_id": "competitor_a_logical_player_id",
            "candidate_first": "competitor_a_first",
            "candidate_seat": "competitor_a_seat",
        },
    }
    payload["returns_encoding"] = {
        "-1": "competitor A loss / competitor B win",
        "0": "draw",
        "1": "competitor A win / competitor B loss",
    }
    return payload


def _competitor_identity(
    competitor: PreparedCompetitor,
) -> dict[str, Any]:
    selection = competitor.loaded.selection
    return {
        "id": competitor.spec.competitor_id,
        "checkpoint_directory": str(selection.directory),
        "step": selection.selected_step,
        "checkpoint_metadata_sha256": competitor.metadata_sha256,
        "selected_step_tree_sha256": competitor.checkpoint_tree_sha256,
        "effective_eval_sha256": competitor.effective_eval_sha256,
        "overrides": competitor.overrides,
    }


def build_job_spec(
    pair: PairSpec,
    competitor_a: PreparedCompetitor,
    competitor_b: PreparedCompetitor,
) -> dict[str, Any]:
    return {
        "implementation": implementation_provenance(),
        "competitor_a": _competitor_identity(competitor_a),
        "competitor_b": _competitor_identity(competitor_b),
        "games": pair.games,
        "batch_size": pair.batch_size,
        "seed": pair.seed,
        "include_game_returns": pair.include_game_returns,
        "stratum_order": [
            {
                "competitor_a_logical_player_id": stratum.candidate_player_id,
                "competitor_a_first": stratum.candidate_first,
                "competitor_a_seat": stratum.candidate_seat,
            }
            for stratum in balanced.STRATUM_SPEC
        ],
    }


def _search_summary(config: Any) -> dict[str, Any]:
    return {
        "search": balanced._search_summary(config.eval.player_search),
        "action_commitment": str(
            config.eval.player_action_commitment_type
        ),
        "q_loss_weight_mode": str(
            config.training.losses.q_loss_weight_mode
        ),
    }


def _checkpoint_provenance(
    competitor: PreparedCompetitor,
) -> dict[str, Any]:
    return {
        "id": competitor.spec.competitor_id,
        "checkpoint_selection": (
            competitor.loaded.selection.provenance()
        ),
        "checkpoint_metadata_sha256": competitor.metadata_sha256,
        "selected_step_tree_sha256": competitor.checkpoint_tree_sha256,
        "selected_step_tree_hash_contract": (
            "SHA-256 over sorted relative directory/file/symlink names, "
            "entry kinds, file sizes, and file/link contents"
        ),
        "effective_eval_sha256": competitor.effective_eval_sha256,
        "overrides": competitor.overrides,
        "evaluation_behavior": _search_summary(
            competitor.effective_config
        ),
    }


def _validate_environment(
    competitor_a: PreparedCompetitor,
    competitor_b: PreparedCompetitor,
) -> tuple[str, int]:
    env_a = competitor_a.effective_config.env
    env_b = competitor_b.effective_config.env
    if env_a.id != "hex" or env_a.board_size is None:
        raise ValueError("competitor A must use a fixed-size Hex environment")
    if env_b.id != env_a.id or env_b.board_size != env_a.board_size:
        raise ValueError(
            "competitor environment mismatch: "
            f"{env_a.id}/{env_a.board_size} != "
            f"{env_b.id}/{env_b.board_size}"
        )
    return str(env_a.id), int(env_a.board_size)


def _validate_summary_counts(
    summary: Mapping[str, Any],
    expected: Sequence[int],
    location: str,
) -> None:
    recomputed = balanced.summarize_returns(expected)
    for field in ("games", "wins", "draws", "losses"):
        if summary.get(field) != recomputed[field]:
            raise ValueError(
                f"{location}.{field} disagrees with pair returns"
            )
    for field in ("win_rate", "avg_return"):
        value = summary.get(field)
        if not isinstance(value, (int, float)) or not math.isclose(
            float(value),
            float(recomputed[field]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"{location}.{field} disagrees with pair returns"
            )


def validate_pair_payload(
    payload: Mapping[str, Any],
    *,
    expected_job_spec_sha256: str | None = None,
) -> None:
    if payload.get("kind") != PAIR_KIND:
        raise ValueError(f"pair artifact kind must be {PAIR_KIND!r}")
    if payload.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise ValueError("unsupported pair artifact schema version")
    job_spec = _require_mapping(payload.get("job_spec"), "job_spec")
    stored_hash = payload.get("job_spec_sha256")
    recomputed_hash = canonical_sha256(job_spec)
    if stored_hash != recomputed_hash:
        raise ValueError("pair artifact job_spec_sha256 is invalid")
    if (
        expected_job_spec_sha256 is not None
        and stored_hash != expected_job_spec_sha256
    ):
        raise ValueError(
            "existing pair artifact belongs to a different job spec"
        )

    strata_raw = payload.get("strata")
    if not isinstance(strata_raw, list) or len(strata_raw) != 4:
        raise ValueError("pair artifact must contain four strata")
    a_returns: list[int] = []
    for index, stratum_raw in enumerate(strata_raw):
        stratum = _require_mapping(stratum_raw, f"strata[{index}]")
        if stratum.get("stratum_index") != index:
            raise ValueError("pair strata are not in canonical order")
        summary_a = _require_mapping(
            stratum.get("competitor_a"),
            f"strata[{index}].competitor_a",
        )
        summary_b = _require_mapping(
            stratum.get("competitor_b"),
            f"strata[{index}].competitor_b",
        )
        games = _require_int(
            summary_a.get("games"),
            f"strata[{index}].competitor_a.games",
            minimum=1,
        )
        if summary_b.get("games") != games:
            raise ValueError("A/B stratum game counts disagree")
        if (
            summary_a.get("wins") != summary_b.get("losses")
            or summary_a.get("losses") != summary_b.get("wins")
            or summary_a.get("draws") != summary_b.get("draws")
        ):
            raise ValueError("A/B stratum outcomes are not inverse")
        a_returns.extend(
            [1] * int(summary_a["wins"])
            + [0] * int(summary_a["draws"])
            + [-1] * int(summary_a["losses"])
        )
    pairwise = _require_mapping(payload.get("pairwise"), "pairwise")
    overall = _require_mapping(pairwise.get("overall"), "pairwise.overall")
    overall_a = _require_mapping(
        overall.get("competitor_a"),
        "pairwise.overall.competitor_a",
    )
    overall_b = _require_mapping(
        overall.get("competitor_b"),
        "pairwise.overall.competitor_b",
    )
    _validate_summary_counts(
        overall_a,
        a_returns,
        "pairwise.overall.competitor_a",
    )
    _validate_summary_counts(
        overall_b,
        [-value for value in a_returns],
        "pairwise.overall.competitor_b",
    )


def _write_json_create_once(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    """Atomically create ``path`` without ever replacing an existing file."""

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def reuse_pair_artifact(
    path: Path,
    expected_job_spec_sha256: str,
) -> tuple[dict[str, Any], str] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"pair output exists but is not a file: {path}")
    payload_raw = json.loads(path.read_text(encoding="utf-8"))
    payload = _require_mapping(payload_raw, str(path))
    validate_pair_payload(
        payload,
        expected_job_spec_sha256=expected_job_spec_sha256,
    )
    return payload, file_sha256(path)


def evaluate_pair(
    pair: PairSpec,
    competitor_a: PreparedCompetitor,
    competitor_b: PreparedCompetitor,
    model_a: nnx.Module,
    model_b: nnx.Module,
    evaluator: Callable[..., LeagueChunkOutput],
    *,
    reproduction: dict[str, Any],
) -> dict[str, Any]:
    env_id, board_size = _validate_environment(
        competitor_a,
        competitor_b,
    )
    games_per_stratum = balanced.validate_evaluation_shape(
        pair.games,
        pair.batch_size,
    )
    num_chunks = games_per_stratum // pair.batch_size
    run_keys = jax.random.split(
        jax.random.PRNGKey(pair.seed),
        len(balanced.STRATUM_SPEC) * num_chunks,
    )
    key_index = 0
    returns_by_stratum: list[list[int]] = []
    returns_by_chunk: list[list[list[int]]] = []
    runtime_strata: list[float] = []
    started = time.perf_counter()

    for stratum_index, stratum in enumerate(balanced.STRATUM_SPEC):
        stratum_started = time.perf_counter()
        stratum_returns: list[int] = []
        chunks: list[list[int]] = []
        for chunk_index in range(num_chunks):
            chunk_started = time.perf_counter()
            output = evaluator(
                run_keys[key_index],
                model_a,
                model_b,
                jnp.asarray(
                    stratum.candidate_player_id,
                    dtype=jnp.int32,
                ),
                jnp.asarray(stratum.candidate_first),
            )
            values = (
                jax.device_get(output.competitor_a_returns)
                .astype(int)
                .tolist()
            )
            if any(value not in (-1, 0, 1) for value in values):
                raise ValueError("Hex evaluator returned a non-outcome value")
            key_index += 1
            stratum_returns.extend(values)
            chunks.append(values)
            print(
                f"a={pair.competitor_a} b={pair.competitor_b} "
                f"a_id={stratum.candidate_player_id} "
                f"a_seat={stratum.candidate_seat} "
                f"chunk={chunk_index + 1}/{num_chunks} "
                f"a_wins={sum(value > 0 for value in values)}/"
                f"{len(values)} "
                f"seconds={time.perf_counter() - chunk_started:.3f}",
                flush=True,
            )
        returns_by_stratum.append(stratum_returns)
        returns_by_chunk.append(chunks)
        runtime_strata.append(time.perf_counter() - stratum_started)

    total_seconds = time.perf_counter() - started
    strata, pairwise = summarize_pair_returns(
        returns_by_stratum,
        competitor_a=pair.competitor_a,
        competitor_b=pair.competitor_b,
    )
    for stratum, seconds in zip(strata, runtime_strata, strict=True):
        stratum["seconds"] = seconds
    job_spec = build_job_spec(pair, competitor_a, competitor_b)
    result: dict[str, Any] = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "kind": PAIR_KIND,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
        },
        "job_spec": job_spec,
        "job_spec_sha256": canonical_sha256(job_spec),
        "reproduction": reproduction,
        "competitors": {
            "a": _checkpoint_provenance(competitor_a),
            "b": _checkpoint_provenance(competitor_b),
        },
        "environment": {
            "id": env_id,
            "board_size": board_size,
            "pgx_player_order_semantics": (
                "_player_order[colour] maps Hex seat/colour to logical "
                "player id; colour 0 moves first"
            ),
        },
        "evaluation": {
            "games": pair.games,
            "games_per_stratum": games_per_stratum,
            "batch_size": pair.batch_size,
            "seed": pair.seed,
            "num_chunks_per_stratum": num_chunks,
            "role_balance": (
                "Equal cells for competitor A logical PGX id {0,1} x "
                "competitor A Hex seat {first,second}; competitor B is the "
                "complement in every game."
            ),
            "rng": (
                "split(PRNGKey(seed), 4*num_chunks) in canonical stratum/"
                "chunk order; each chunk splits loop/init once and each ply "
                "splits loop into next/A/B keys. A and B keep those search "
                "key streams independent of logical id and seat."
            ),
        },
        "confidence_interval": {
            "method": "Wilson score interval",
            "nominal_coverage": 0.95,
            "sidedness": "two-sided",
            "z": 1.959963984540054,
            "success_definition": "the named competitor's return > 0",
            "scope": (
                "Every stratum, pooled competitor, and seat-specific "
                "competitor summary."
            ),
        },
        "runtime": {
            "seconds": total_seconds,
            "games_per_second": pair.games / total_seconds,
            "jax_default_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "python": platform.python_version(),
            "jax": jax.__version__,
        },
        "strata": strata,
        "pairwise": pairwise,
    }
    if pair.include_game_returns:
        result["game_returns"] = build_league_game_returns(
            returns_by_chunk,
            run_keys=run_keys,
            pair=pair,
        )
    validate_pair_payload(result)
    return result


def _default_pair_output(
    output_directory: Path,
    pair: PairSpec,
    job_spec_sha256: str,
) -> Path:
    name = (
        f"{pair.competitor_a}__vs__{pair.competitor_b}"
        f"__{job_spec_sha256[:16]}.json"
    )
    return output_directory / name


def _parse_competitor(
    value: Any,
    *,
    base: Path,
    index: int,
) -> CompetitorSpec:
    location = f"competitors[{index}]"
    item = _require_mapping(value, location)
    competitor_id = _require_identifier(item.get("id"), f"{location}.id")
    checkpoint = _resolve_path(
        item.get("checkpoint"),
        base,
        f"{location}.checkpoint",
    )
    step = _require_int(item.get("step"), f"{location}.step", minimum=0)
    root_estimator = _parse_root_estimator(
        item.get("root_action_estimator"),
        f"{location}.root_action_estimator",
    )
    kappa = _optional_positive_float(item.get("kappa"), f"{location}.kappa")
    half_width_raw = item.get("prefix_cdf_half_width")
    half_width = (
        None
        if half_width_raw is None
        else _require_int(
            half_width_raw,
            f"{location}.prefix_cdf_half_width",
            minimum=1,
        )
    )
    known = {
        "id",
        "checkpoint",
        "step",
        "root_action_estimator",
        "prefix_cdf_half_width",
        "kappa",
    }
    unknown = sorted(set(item) - known)
    if unknown:
        raise ValueError(f"{location} has unknown fields: {unknown}")
    return CompetitorSpec(
        competitor_id=competitor_id,
        checkpoint=checkpoint,
        step=step,
        root_action_estimator=root_estimator,
        prefix_cdf_half_width=half_width,
        kappa=kappa,
    )


def load_manifest(path: Path) -> LeagueManifest:
    resolved = path.resolve()
    encoded = resolved.read_bytes()
    raw = json.loads(encoded)
    payload = _require_mapping(raw, "manifest")
    if payload.get("kind") != MANIFEST_KIND:
        raise ValueError(f"manifest kind must be {MANIFEST_KIND!r}")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported league manifest schema version")
    base = resolved.parent
    output_directory = _resolve_path(
        payload.get("output_directory"),
        base,
        "output_directory",
    )

    competitors_raw = payload.get("competitors")
    if not isinstance(competitors_raw, list) or not competitors_raw:
        raise ValueError("competitors must be a non-empty JSON array")
    competitors = tuple(
        _parse_competitor(item, base=base, index=index)
        for index, item in enumerate(competitors_raw)
    )
    roster = {item.competitor_id: item for item in competitors}
    if len(roster) != len(competitors):
        raise ValueError("competitor ids must be unique")

    global_games = _require_int(
        payload.get("games"),
        "games",
        minimum=1,
    )
    global_batch = _require_int(
        payload.get("batch_size"),
        "batch_size",
        minimum=1,
    )
    global_seed = _require_int(payload.get("seed"), "seed")
    global_raw = _require_bool(
        payload.get("include_game_returns", False),
        "include_game_returns",
    )
    balanced.validate_evaluation_shape(global_games, global_batch)

    pairs_raw = payload.get("pairs")
    include_self_play = _require_bool(
        payload.get("include_self_play", False),
        "include_self_play",
    )
    pair_items: list[dict[str, Any]]
    if pairs_raw is None:
        pair_items = [
            {"a": item_a.competitor_id, "b": item_b.competitor_id}
            for index_a, item_a in enumerate(competitors)
            for index_b, item_b in enumerate(competitors)
            if (
                index_b > index_a
                or (include_self_play and index_b == index_a)
            )
        ]
    else:
        if not isinstance(pairs_raw, list) or not pairs_raw:
            raise ValueError("pairs must be a non-empty JSON array when set")
        pair_items = [
            _require_mapping(item, f"pairs[{index}]")
            for index, item in enumerate(pairs_raw)
        ]

    pairs: list[PairSpec] = []
    unordered_seen: set[tuple[str, str]] = set()
    outputs_seen: set[Path] = set()
    for index, item in enumerate(pair_items):
        location = f"pairs[{index}]"
        a = _require_identifier(item.get("a"), f"{location}.a")
        b = _require_identifier(item.get("b"), f"{location}.b")
        if a not in roster or b not in roster:
            raise ValueError(f"{location} references an unknown competitor")
        if a == b and not include_self_play:
            raise ValueError(
                f"{location} is self-play but include_self_play is false"
            )
        unordered = (a, b) if a <= b else (b, a)
        if unordered in unordered_seen:
            raise ValueError(
                f"{location} duplicates unordered pair {unordered}"
            )
        unordered_seen.add(unordered)
        games = _require_int(
            item.get("games", global_games),
            f"{location}.games",
            minimum=1,
        )
        batch = _require_int(
            item.get("batch_size", global_batch),
            f"{location}.batch_size",
            minimum=1,
        )
        seed = _require_int(
            item.get("seed", global_seed),
            f"{location}.seed",
        )
        include_raw = _require_bool(
            item.get("include_game_returns", global_raw),
            f"{location}.include_game_returns",
        )
        balanced.validate_evaluation_shape(games, batch)
        output_value = item.get("output")
        output = (
            _resolve_path(
                output_value,
                output_directory,
                f"{location}.output",
            )
            if output_value is not None
            else None
        )
        if output is not None:
            if output in outputs_seen:
                raise ValueError("explicit pair output paths must be unique")
            outputs_seen.add(output)
        known = {
            "a",
            "b",
            "games",
            "batch_size",
            "seed",
            "include_game_returns",
            "output",
        }
        unknown = sorted(set(item) - known)
        if unknown:
            raise ValueError(f"{location} has unknown fields: {unknown}")
        pairs.append(
            PairSpec(
                competitor_a=a,
                competitor_b=b,
                games=games,
                batch_size=batch,
                seed=seed,
                include_game_returns=include_raw,
                output=output,
            )
        )
    if not pairs:
        raise ValueError(
            "manifest expands to no pairs; add another competitor or enable "
            "self-play"
        )
    known_top = {
        "schema_version",
        "kind",
        "output_directory",
        "games",
        "batch_size",
        "seed",
        "include_game_returns",
        "include_self_play",
        "competitors",
        "pairs",
    }
    unknown_top = sorted(set(payload) - known_top)
    if unknown_top:
        raise ValueError(f"manifest has unknown fields: {unknown_top}")
    return LeagueManifest(
        path=resolved,
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        output_directory=output_directory,
        competitors=competitors,
        pairs=tuple(pairs),
    )


def _evaluator_cache_key(
    competitor_a: PreparedCompetitor,
    competitor_b: PreparedCompetitor,
    batch_size: int,
) -> tuple[str, str, int]:
    return (
        competitor_a.effective_eval_sha256,
        competitor_b.effective_eval_sha256,
        batch_size,
    )


def _script_reproduction(argv: Sequence[str]) -> dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "command": " ".join(shlex.quote(argument) for argument in argv),
        "working_directory": str(Path.cwd().resolve()),
        "script_path": str(script),
        "script_sha256": file_sha256(script),
    }


def run_manifest(
    manifest: LeagueManifest,
    *,
    summary_output: Path | None,
    reproduction_argv: Sequence[str],
) -> dict[str, Any]:
    """Run pending matrix cells while reusing validated immutable artifacts."""

    prepared = {
        spec.competitor_id: prepare_competitor(spec)
        for spec in manifest.competitors
    }
    planned: list[
        tuple[PairSpec, PreparedCompetitor, PreparedCompetitor, Path, str]
    ] = []
    resolved_outputs: set[Path] = set()
    for pair in manifest.pairs:
        competitor_a = prepared[pair.competitor_a]
        competitor_b = prepared[pair.competitor_b]
        job_hash = canonical_sha256(
            build_job_spec(pair, competitor_a, competitor_b)
        )
        output = (
            pair.output
            if pair.output is not None
            else _default_pair_output(
                manifest.output_directory,
                pair,
                job_hash,
            )
        ).resolve()
        if output in resolved_outputs:
            raise ValueError(f"multiple pairs resolve to output {output}")
        resolved_outputs.add(output)
        planned.append(
            (pair, competitor_a, competitor_b, output, job_hash)
        )
    if (
        summary_output is not None
        and summary_output.resolve() in resolved_outputs
    ):
        raise ValueError(
            "matrix summary output must not overlap a create-once pair artifact"
        )

    reusable: dict[Path, tuple[dict[str, Any], str]] = {}
    pending: list[
        tuple[PairSpec, PreparedCompetitor, PreparedCompetitor, Path, str]
    ] = []
    for job in planned:
        existing = reuse_pair_artifact(job[3], job[4])
        if existing is None:
            pending.append(job)
        else:
            reusable[job[3]] = existing

    environments: dict[tuple[str, int], Any] = {}
    models: dict[tuple[str, int], nnx.Module] = {}
    evaluators: dict[tuple[str, str, int], Callable[..., LeagueChunkOutput]] = {}
    entries: list[dict[str, Any]] = []
    reproduction = _script_reproduction(reproduction_argv)
    started = time.perf_counter()

    for pair, competitor_a, competitor_b, output, job_hash in planned:
        existing = reusable.get(output)
        if existing is not None:
            payload, digest = existing
            entries.append(
                {
                    "competitor_a": pair.competitor_a,
                    "competitor_b": pair.competitor_b,
                    "status": "reused",
                    "artifact": str(output),
                    "sha256": digest,
                    "job_spec_sha256": job_hash,
                    "a_win_rate": payload["pairwise"]["overall"][
                        "competitor_a"
                    ]["win_rate"],
                }
            )
            print(f"Reused {output} sha256={digest}", flush=True)
            continue

        env_id, board_size = _validate_environment(
            competitor_a,
            competitor_b,
        )
        env_key = (env_id, board_size)
        if env_key not in environments:
            environments[env_key] = make_env(env_id, board_size)
        env = environments[env_key]
        for competitor in (competitor_a, competitor_b):
            if competitor.model_cache_key not in models:
                models[competitor.model_cache_key] = load_prepared_model(
                    competitor,
                    env,
                )
        evaluator_key = _evaluator_cache_key(
            competitor_a,
            competitor_b,
            pair.batch_size,
        )
        if evaluator_key not in evaluators:
            evaluators[evaluator_key] = make_league_evaluator(
                env,
                competitor_a.effective_config,
                competitor_b.effective_config,
                pair.batch_size,
            )
        payload = evaluate_pair(
            pair,
            competitor_a,
            competitor_b,
            models[competitor_a.model_cache_key],
            models[competitor_b.model_cache_key],
            evaluators[evaluator_key],
            reproduction=reproduction,
        )
        if payload["job_spec_sha256"] != job_hash:
            raise AssertionError("evaluated pair job hash changed after planning")
        try:
            digest = _write_json_create_once(output, payload)
            status = "created"
        except FileExistsError:
            raced = reuse_pair_artifact(output, job_hash)
            if raced is None:
                raise AssertionError("pair artifact disappeared after race")
            payload, digest = raced
            status = "reused_after_race"
        entries.append(
            {
                "competitor_a": pair.competitor_a,
                "competitor_b": pair.competitor_b,
                "status": status,
                "artifact": str(output),
                "sha256": digest,
                "job_spec_sha256": job_hash,
                "a_win_rate": payload["pairwise"]["overall"][
                    "competitor_a"
                ]["win_rate"],
            }
        )
        print(f"{status.capitalize()} {output} sha256={digest}", flush=True)

    summary: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "kind": RUN_KIND,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
        },
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.file_sha256,
        },
        "output_directory": str(manifest.output_directory),
        "runtime": {
            "seconds": time.perf_counter() - started,
            "created": sum(item["status"] == "created" for item in entries),
            "reused": sum(item["status"] != "created" for item in entries),
            "models_loaded": len(models),
            "evaluator_variants": len(evaluators),
        },
        "pairs": entries,
        "pair_artifacts": [
            {
                "path": item["artifact"],
                "sha256": item["sha256"],
                "job_spec_sha256": item["job_spec_sha256"],
            }
            for item in entries
        ],
    }
    if summary_output is not None:
        digest = balanced._write_json(summary_output, summary)
        print(
            f"Wrote matrix summary {summary_output.resolve()} "
            f"sha256={digest}",
            flush=True,
        )
    return summary


def _pair_from_args(args: argparse.Namespace) -> tuple[
    PairSpec,
    CompetitorSpec,
    CompetitorSpec,
]:
    games = int(args.games)
    batch_size = int(args.batch_size)
    balanced.validate_evaluation_shape(games, batch_size)
    competitor_a = CompetitorSpec(
        competitor_id=_require_identifier(args.a_id, "--a-id"),
        checkpoint=args.a_checkpoint.resolve(),
        step=int(args.a_step),
        root_action_estimator=args.a_root_action_estimator,
        prefix_cdf_half_width=args.a_prefix_cdf_half_width,
        kappa=args.a_kappa,
    )
    competitor_b = CompetitorSpec(
        competitor_id=_require_identifier(args.b_id, "--b-id"),
        checkpoint=args.b_checkpoint.resolve(),
        step=int(args.b_step),
        root_action_estimator=args.b_root_action_estimator,
        prefix_cdf_half_width=args.b_prefix_cdf_half_width,
        kappa=args.b_kappa,
    )
    pair = PairSpec(
        competitor_a=competitor_a.competitor_id,
        competitor_b=competitor_b.competitor_id,
        games=games,
        batch_size=batch_size,
        seed=int(args.seed),
        include_game_returns=bool(args.include_game_returns),
        output=args.output.resolve(),
    )
    return pair, competitor_a, competitor_b


def run_single_pair(
    pair: PairSpec,
    spec_a: CompetitorSpec,
    spec_b: CompetitorSpec,
    *,
    reproduction_argv: Sequence[str],
) -> tuple[dict[str, Any], str, str]:
    competitor_a = prepare_competitor(spec_a)
    competitor_b = prepare_competitor(spec_b)
    job_hash = canonical_sha256(
        build_job_spec(pair, competitor_a, competitor_b)
    )
    if pair.output is None:
        raise ValueError("single pair execution requires an output path")
    existing = reuse_pair_artifact(pair.output, job_hash)
    if existing is not None:
        payload, digest = existing
        return payload, digest, "reused"
    env_id, board_size = _validate_environment(competitor_a, competitor_b)
    env = make_env(env_id, board_size)
    model_a = load_prepared_model(competitor_a, env)
    model_b = load_prepared_model(competitor_b, env)
    evaluator = make_league_evaluator(
        env,
        competitor_a.effective_config,
        competitor_b.effective_config,
        pair.batch_size,
    )
    payload = evaluate_pair(
        pair,
        competitor_a,
        competitor_b,
        model_a,
        model_b,
        evaluator,
        reproduction=_script_reproduction(reproduction_argv),
    )
    try:
        digest = _write_json_create_once(pair.output, payload)
        status = "created"
    except FileExistsError:
        raced = reuse_pair_artifact(pair.output, job_hash)
        if raced is None:
            raise AssertionError("pair artifact disappeared after race")
        payload, digest = raced
        status = "reused_after_race"
    return payload, digest, status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run local-only, exact-step, role-balanced Hex checkpoint "
            "cross-play."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pair = commands.add_parser("pair", help="evaluate one exact pair")
    pair.add_argument("--a-id", required=True)
    pair.add_argument("--a-checkpoint", type=Path, required=True)
    pair.add_argument("--a-step", type=int, required=True)
    pair.add_argument(
        "--a-root-action-estimator",
        choices=tuple(str(item) for item in PosteriorPolicyEstimator),
        default=None,
    )
    pair.add_argument("--a-kappa", type=float, default=None)
    pair.add_argument(
        "--a-prefix-cdf-half-width",
        type=int,
        default=None,
    )
    pair.add_argument("--b-id", required=True)
    pair.add_argument("--b-checkpoint", type=Path, required=True)
    pair.add_argument("--b-step", type=int, required=True)
    pair.add_argument(
        "--b-root-action-estimator",
        choices=tuple(str(item) for item in PosteriorPolicyEstimator),
        default=None,
    )
    pair.add_argument("--b-kappa", type=float, default=None)
    pair.add_argument(
        "--b-prefix-cdf-half-width",
        type=int,
        default=None,
    )
    pair.add_argument("--games", type=int, default=4096)
    pair.add_argument("--batch-size", type=int, default=256)
    pair.add_argument("--seed", type=int, default=5504096)
    pair.add_argument("--include-game-returns", action="store_true")
    pair.add_argument("--output", type=Path, required=True)

    matrix = commands.add_parser(
        "matrix",
        help="run/resume all pairs in a manifest",
    )
    matrix.add_argument("--manifest", type=Path, required=True)
    matrix.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "mutable run summary path (default: "
            "<output_directory>/latest_run.json)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    reproduction_argv = list(
        sys.argv if argv is None else [sys.argv[0], *argv]
    )
    if args.command == "pair":
        if args.a_step < 0 or args.b_step < 0:
            parser.error("checkpoint steps must be non-negative")
        for flag, value in (
            ("--a-kappa", args.a_kappa),
            ("--b-kappa", args.b_kappa),
        ):
            if value is not None and (
                not math.isfinite(value) or value <= 0.0
            ):
                parser.error(f"{flag} must be finite and positive")
        for flag, value in (
            (
                "--a-prefix-cdf-half-width",
                args.a_prefix_cdf_half_width,
            ),
            (
                "--b-prefix-cdf-half-width",
                args.b_prefix_cdf_half_width,
            ),
        ):
            if value is not None and value < 1:
                parser.error(f"{flag} must be positive")
        pair, spec_a, spec_b = _pair_from_args(args)
        payload, digest, status = run_single_pair(
            pair,
            spec_a,
            spec_b,
            reproduction_argv=reproduction_argv,
        )
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        print(
            f"{status.capitalize()} {pair.output} sha256={digest}",
            flush=True,
        )
        return

    manifest = load_manifest(args.manifest)
    summary_output = (
        args.summary_output.resolve()
        if args.summary_output is not None
        else manifest.output_directory / "latest_run.json"
    )
    summary = run_manifest(
        manifest,
        summary_output=summary_output,
        reproduction_argv=reproduction_argv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
