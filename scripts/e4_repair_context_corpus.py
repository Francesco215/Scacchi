#!/usr/bin/env python3
"""Collect and analyze actual posterior-repair estimator contexts on Hex6.

This is an offline E4 diagnostic.  It does not change traversal, posterior
repair, public root targets, action commitment, training, or configuration.

Examples
--------
Collect a checkpoint/stage-balanced corpus from the instrumented E0 run::

    JAX_PLATFORMS=cuda,cpu uv run python scripts/e4_repair_context_corpus.py \
      collect \
      --checkpoint checkpoints/hex6_info_baseline_s0 \
      --steps 0,50,100 \
      --output experiments/e4/hex6_repair_contexts_v1

Verify stored invariants and hashes::

    uv run python scripts/e4_repair_context_corpus.py verify \
      --corpus experiments/e4/hex6_repair_contexts_v1

Measure exact-Beta quadrature M32 cache noise::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu \
      uv run python scripts/e4_repair_context_corpus.py analyze \
      --corpus experiments/e4/hex6_repair_contexts_v1

Benchmark lower-cost deterministic and Rao--Blackwellized estimators::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu \
      uv run python scripts/e4_repair_context_corpus.py benchmark \
      --corpus experiments/e4/hex6_repair_contexts_v1 \
      --quadrature-grids 20:0.3,40:0.2,60:0.15,80:0.1 \
      --rao-blackwell-samples 8,16,32,64 \
      --winner-mc-samples 128

Add ``--complete-search-roots 8192`` on a GPU for the authoritative
end-to-end runtime gate.  The frozen roots are stage-interleaved and tiled to
that production batch shape; baseline and candidates receive identical keys
in an interleaved timing protocol.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, NamedTuple

import numpy as np


STAGES: tuple[tuple[str, int, int], ...] = (
    ("early", 0, 11),
    ("mid", 12, 23),
    ("late", 24, 35),
)
CORPUS_VERSION = 1


class _PolicyEstimate(NamedTuple):
    """Common JAX result surface for ordinary winner-count candidates."""

    policy: Any
    normalization_error: Any
    finite: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    relative_paths = (
        "scacchi/dirichlet_mctx/action_selection.py",
        "scacchi/dirichlet_mctx/posterior_updates.py",
        "scacchi/dirichlet_mctx/search.py",
        "scacchi/dirichlet_mctx/tree.py",
        "scripts/e4_repair_context_corpus.py",
    )
    for relative in relative_paths:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _parse_steps(encoded: str) -> tuple[int, ...]:
    steps = tuple(int(value.strip()) for value in encoded.split(","))
    if not steps or any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError(
            f"steps must be comma-separated nonnegative integers, got {encoded}"
        )
    if len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError(f"steps must be unique, got {encoded}")
    return steps


def _parse_positive_ints(encoded: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in encoded.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated positive integers, got {encoded}"
        ) from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated positive integers, got {encoded}"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            f"values must be unique, got {encoded}"
        )
    return values


def _parse_optional_positive_ints(encoded: str) -> tuple[int, ...]:
    if encoded.strip().lower() in {"", "none"}:
        return ()
    return _parse_positive_ints(encoded)


def _parse_quadrature_grids(
    encoded: str,
) -> tuple[tuple[int, float], ...]:
    grids: list[tuple[int, float]] = []
    for item in encoded.split(","):
        pieces = item.strip().split(":")
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(
                "quadrature grids must be HALF_WIDTH:STEP pairs; "
                f"got {item!r}"
            )
        try:
            half_width = int(pieces[0])
            step = float(pieces[1])
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid quadrature grid {item!r}"
            ) from error
        if half_width < 1 or not np.isfinite(step) or step <= 0:
            raise argparse.ArgumentTypeError(
                "quadrature half-width and step must be positive; "
                f"got {item!r}"
            )
        grids.append((half_width, step))
    if not grids or len(set(grids)) != len(grids):
        raise argparse.ArgumentTypeError(
            f"quadrature grids must be nonempty and unique, got {encoded}"
        )
    return tuple(grids)


def _parse_optional_quadrature_grids(
    encoded: str,
) -> tuple[tuple[int, float], ...]:
    if encoded.strip().lower() in {"", "none"}:
        return ()
    return _parse_quadrature_grids(encoded)


def _load_checkpoint_config_and_model(
    checkpoint_path: Path,
    step: int,
):
    # The exact-step loader already backs the frozen oracle.  Importing from
    # the neighboring script avoids a second checkpoint interpretation.
    from hex_oracle_harness import (
        _load_checkpoint_config_and_step,
        _load_checkpoint_model_at_step,
    )

    from scacchi.envs import make_env

    config, restored_step, progress = _load_checkpoint_config_and_step(
        checkpoint_path,
        step,
    )
    if restored_step != step:
        raise ValueError(
            f"requested checkpoint {step}, restored {restored_step}"
        )
    if config.env.id != "hex" or config.env.board_size != 6:
        raise ValueError(
            "E4 corpus collection currently requires Hex6; "
            f"got {config.env.id!s} size={config.env.board_size}"
        )
    if config.env.num_outcomes not in (None, 2):
        raise ValueError(
            "E4 binary reference requires two outcomes; "
            f"got {config.env.num_outcomes}"
        )
    env = make_env(config.env.id, config.env.board_size)
    model = _load_checkpoint_model_at_step(
        checkpoint_path,
        step=step,
        config=config,
        env=env,
    )
    return config, env, model, progress


def _make_on_policy_rollout(
    env: Any,
    config: Any,
    *,
    batch_size: int,
    max_steps: int,
):
    from flax import nnx
    import jax

    from scacchi.play_search import make_search_player

    @nnx.jit
    def rollout(model, rng_key):
        rollout_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, batch_size)
        state = jax.vmap(env.init)(init_keys)
        player = make_search_player(
            env,
            model,
            config.selfplay.search,
            config.selfplay.action_commitment_type,
            q_loss_weight_mode=str(
                config.training.losses.q_loss_weight_mode
            ),
        )

        def step(current_state, key):
            analyzed_state = current_state
            output = player(current_state, key)
            next_state = jax.vmap(env.step)(current_state, output.action)
            return next_state, (analyzed_state, output.action)

        step_keys = jax.random.split(rollout_key, max_steps)
        _, (states, actions) = jax.lax.scan(step, state, step_keys)
        return init_keys, states, actions

    return rollout


def _select_roots(
    states: Any,
    actions: Any,
    init_keys: Any,
    *,
    checkpoint_step: int,
    roots_per_stage: int,
    rng: np.random.Generator,
    first_root_id: int,
):
    import jax
    import jax.numpy as jnp

    terminated = np.asarray(jax.device_get(states.terminated), dtype=bool)
    plys = np.asarray(jax.device_get(states._step_count), dtype=np.int32)
    actions_host = np.asarray(jax.device_get(actions), dtype=np.int32)
    init_keys_host = np.asarray(
        jax.device_get(jax.random.key_data(init_keys)),
        dtype=np.uint32,
    )
    num_steps, batch_size = terminated.shape

    selected_time: list[int] = []
    selected_game: list[int] = []
    stage_ids: list[int] = []
    root_weights: list[float] = []
    stage_population: dict[str, int] = {}

    for stage_id, (stage_name, lower, upper) in enumerate(STAGES):
        candidates = [
            (time_index, game_index)
            for time_index in range(num_steps)
            for game_index in range(batch_size)
            if (
                lower <= int(plys[time_index, game_index]) <= upper
                and not terminated[time_index, game_index]
            )
        ]
        stage_population[stage_name] = len(candidates)
        order = rng.permutation(len(candidates))
        chosen: list[tuple[int, int]] = []
        used_games: set[int] = set()
        for index in order:
            time_index, game_index = candidates[int(index)]
            if game_index in used_games:
                continue
            chosen.append((time_index, game_index))
            used_games.add(game_index)
            if len(chosen) == roots_per_stage:
                break
        if len(chosen) < roots_per_stage:
            raise RuntimeError(
                f"checkpoint {checkpoint_step} has only {len(chosen)} "
                f"independent {stage_name} roots; increase --rollout-batch-size"
            )
        inclusion_weight = len(candidates) / len(chosen)
        for time_index, game_index in chosen:
            selected_time.append(time_index)
            selected_game.append(game_index)
            stage_ids.append(stage_id)
            root_weights.append(inclusion_weight)

    time_index = jnp.asarray(selected_time, dtype=jnp.int32)
    game_index = jnp.asarray(selected_game, dtype=jnp.int32)
    selected_states = jax.tree.map(
        lambda leaf: leaf[time_index, game_index],
        states,
    )
    count = len(selected_time)
    root_ids = np.arange(
        first_root_id,
        first_root_id + count,
        dtype=np.int32,
    )
    root_ply = plys[np.asarray(selected_time), np.asarray(selected_game)]
    padded_actions = np.zeros(
        (count, num_steps),
        dtype=np.int32,
    )
    root_init_keys = np.empty((count, 2), dtype=np.uint32)
    state_hashes = np.empty((count,), dtype="S64")
    boards = np.asarray(
        jax.device_get(selected_states._x.board),
        dtype=np.int32,
    )
    player_orders = np.asarray(
        jax.device_get(selected_states._player_order),
        dtype=np.int32,
    )
    current_players = np.asarray(
        jax.device_get(selected_states.current_player),
        dtype=np.int32,
    )
    for row, (time_value, game_value) in enumerate(
        zip(selected_time, selected_game, strict=True)
    ):
        action_count = int(root_ply[row])
        if action_count:
            padded_actions[row, :action_count] = actions_host[
                :action_count,
                game_value,
            ]
        root_init_keys[row] = init_keys_host[game_value]
        digest = hashlib.sha256()
        digest.update(boards[row].tobytes())
        digest.update(player_orders[row].tobytes())
        digest.update(current_players[row].tobytes())
        state_hashes[row] = digest.hexdigest().encode("ascii")

    root_table = {
        "root_id": root_ids,
        "game_cluster_id": (
            checkpoint_step * 1_000_000
            + np.asarray(selected_game, dtype=np.int32)
        ),
        "rollout_game_index": np.asarray(selected_game, dtype=np.int32),
        "checkpoint_step": np.full(
            count,
            checkpoint_step,
            dtype=np.int32,
        ),
        "stage_id": np.asarray(stage_ids, dtype=np.int8),
        "root_ply": root_ply.astype(np.int32),
        "root_weight": np.asarray(root_weights, dtype=np.float64),
        "init_key": root_init_keys,
        "padded_actions": padded_actions,
        "action_count": root_ply.astype(np.int32),
        "state_sha256": state_hashes,
    }
    return selected_states, root_table, stage_population


def _replay_roots(env: Any, root_table: dict[str, np.ndarray]):
    import jax
    import jax.numpy as jnp

    init_key_data = jnp.asarray(root_table["init_key"], dtype=jnp.uint32)
    init_keys = jax.vmap(jax.random.wrap_key_data)(init_key_data)
    padded_actions = jnp.asarray(
        root_table["padded_actions"],
        dtype=jnp.int32,
    )
    action_count = jnp.asarray(root_table["action_count"], dtype=jnp.int32)
    num_steps = padded_actions.shape[1]

    @jax.jit
    def replay(keys, actions, counts):
        state = jax.vmap(env.init)(keys)

        def step(current_state, indexed_actions):
            index, action = indexed_actions
            stepped = jax.vmap(env.step)(current_state, action)
            active = index < counts
            current_state = jax.tree.map(
                lambda old, new: jnp.where(
                    active.reshape(
                        (active.shape[0],)
                        + (1,) * (old.ndim - 1)
                    ),
                    new,
                    old,
                ),
                current_state,
                stepped,
            )
            return current_state, None

        state, _ = jax.lax.scan(
            step,
            state,
            (
                jnp.arange(num_steps, dtype=jnp.int32),
                jnp.swapaxes(actions, 0, 1),
            ),
        )
        return state

    return replay(init_keys, padded_actions, action_count)


def _assert_same_states(expected: Any, actual: Any) -> None:
    import jax

    expected_leaves, expected_tree = jax.tree.flatten(expected)
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    if expected_tree != actual_tree or len(expected_leaves) != len(actual_leaves):
        raise AssertionError("replayed PGX state structure differs")
    for index, (lhs, rhs) in enumerate(
        zip(expected_leaves, actual_leaves, strict=True)
    ):
        if not np.array_equal(jax.device_get(lhs), jax.device_get(rhs)):
            raise AssertionError(
                f"replayed PGX state leaf {index} differs"
            )


class _ContextCollector:
    """Host sink for occurrence-weighted active repair snapshots."""

    def __init__(self) -> None:
        self._chunks: dict[str, list[np.ndarray]] = {}
        self.callback_count = 0

    def _append(self, name: str, value: np.ndarray) -> None:
        self._chunks.setdefault(name, []).append(np.asarray(value))

    def __call__(
        self,
        effective_alpha,
        cache_alpha,
        invalid_actions,
        categorical_outcome,
        n_down,
        gamma,
        value_prior,
        previous_value_alpha,
        active,
        root_id,
        game_cluster_id,
        root_ply,
        root_weight,
        stage_id,
        checkpoint_step,
        node_index,
        node_ply,
        update_key,
    ):
        active = np.asarray(active, dtype=bool)
        selected = np.flatnonzero(active)
        callback_ordinal = self.callback_count
        self.callback_count += 1
        if selected.size == 0:
            return np.int32(callback_ordinal)

        rows = selected.size
        self._append("effective_alpha", np.asarray(effective_alpha)[selected])
        self._append("cache_alpha", np.asarray(cache_alpha)[selected])
        self._append(
            "invalid_actions",
            np.asarray(invalid_actions, dtype=bool)[selected],
        )
        self._append(
            "categorical_outcome",
            np.asarray(categorical_outcome, dtype=np.int8)[selected],
        )
        self._append("n_down", np.asarray(n_down, dtype=np.int32)[selected])
        self._append("gamma", np.asarray(gamma)[selected])
        self._append("value_prior", np.asarray(value_prior)[selected])
        self._append(
            "previous_value_alpha",
            np.asarray(previous_value_alpha)[selected],
        )
        self._append("active", np.ones((rows,), dtype=bool))
        self._append("root_id", np.asarray(root_id, dtype=np.int32)[selected])
        self._append(
            "game_cluster_id",
            np.asarray(game_cluster_id, dtype=np.int32)[selected],
        )
        self._append("root_ply", np.asarray(root_ply, dtype=np.int32)[selected])
        self._append(
            "root_weight",
            np.asarray(root_weight, dtype=np.float64)[selected],
        )
        self._append("stage_id", np.asarray(stage_id, dtype=np.int8)[selected])
        self._append(
            "checkpoint_step",
            np.asarray(checkpoint_step, dtype=np.int32)[selected],
        )
        self._append(
            "node_index",
            np.asarray(node_index, dtype=np.int32)[selected],
        )
        self._append(
            "node_ply",
            np.asarray(node_ply, dtype=np.int32)[selected],
        )
        key = np.asarray(update_key, dtype=np.uint32)
        self._append("update_key", np.broadcast_to(key, (rows, *key.shape)))
        self._append(
            "callback_ordinal",
            np.full(rows, callback_ordinal, dtype=np.int32),
        )
        return np.int32(callback_ordinal)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._chunks:
            raise RuntimeError("no posterior-repair contexts were captured")
        return {
            name: np.concatenate(chunks, axis=0)
            for name, chunks in self._chunks.items()
        }


def _run_tree(
    env: Any,
    model: Any,
    state: Any,
    rng_key: Any,
    search_config: Any,
    posterior_update: Callable[..., Any],
    *,
    public_policy_samples: int = 1,
    public_policy_sample_chunk_size: int = 1,
):
    import functools

    import jax
    import jax.numpy as jnp

    from scacchi import dirichlet_mctx
    from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
    from scacchi.dirichlet_q_search import (
        make_dirichlet_expand_fn,
        terminal_outcome_from_reward,
    )
    from scacchi.play_search import make_evaluator

    evaluator = make_evaluator(model)
    prediction = evaluator(state.observation)
    if prediction.alpha_v is None or prediction.alpha_q is None:
        raise ValueError("checkpoint must expose native V/Q alphas")
    root_reward = state.rewards[
        jnp.arange(state.rewards.shape[0]),
        state.current_player,
    ]
    terminal_outcome = jnp.where(
        state.terminated,
        terminal_outcome_from_reward(
            root_reward,
            prediction.alpha_v.shape[-1],
        ),
        jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
    )
    root = dirichlet_mctx.RootFnOutput(
        prior_logits=prediction.logits,
        value=prediction.alpha_v,
        action_values=prediction.alpha_q,
        embedding=state,
        terminal_outcome=terminal_outcome,
        to_play=state.current_player,
    )
    expand_fn = make_dirichlet_expand_fn(env, evaluator)
    output = dirichlet_mctx.dirichlet_thompson_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=expand_fn,
        num_simulations=int(search_config.num_simulations),
        invalid_actions=~state.legal_action_mask,
        posterior_update=posterior_update,
        max_depth=search_config.max_depth,
        # Corpus tracing uses one public sample, while end-to-end timing passes
        # the production readout budget.  The wrapper splits a distinct policy
        # key before traversal, so this does not change internal repair.
        policy_samples=public_policy_samples,
        policy_sample_chunk_size=public_policy_sample_chunk_size,
    )
    tree = output.search_tree
    return (
        tree.children_index,
        tree.node_categorical_outcome,
        tree.node_payload,
        tree.edge_categorical_outcome,
        tree.edge_payload,
        tree.node_value_alpha,
        tree.edge_alpha,
        output.action_weights,
        tree.simulation_active_count,
        tree.executed_simulation_call_count,
    )


def _make_trace_functions(
    env: Any,
    config: Any,
    collector: _ContextCollector,
):
    import functools

    from flax import nnx
    import jax
    import jax.numpy as jnp
    from jax.experimental import io_callback

    from scacchi.dirichlet_mctx.posterior_updates import (
        posterior_estimator_snapshot,
        update_posterior,
    )

    search_config = config.selfplay.search.dirichlet_thompson
    posterior_samples = (
        int(search_config.policy_samples)
        if search_config.posterior_policy_samples is None
        else int(search_config.posterior_policy_samples)
    )
    chunk_size = (
        4
        if search_config.policy_sample_chunk_size is None
        else int(search_config.policy_sample_chunk_size)
    )
    base_update = functools.partial(
        update_posterior,
        kappa=float(search_config.kappa),
        policy_samples=max(1, posterior_samples),
        policy_sample_chunk_size=max(1, chunk_size),
    )

    @nnx.jit
    def untraced(model, state, rng_key):
        return _run_tree(
            env,
            model,
            state,
            rng_key,
            search_config,
            base_update,
        )

    @nnx.jit
    def traced(
        model,
        state,
        rng_key,
        root_id,
        game_cluster_id,
        root_ply,
        root_weight,
        stage_id,
        checkpoint_step,
    ):
        def traced_update(update_key, context):
            snapshot = posterior_estimator_snapshot(
                context,
                kappa=float(search_config.kappa),
            )
            node_ply = context.node.embedding._step_count
            _ = io_callback(
                collector,
                jax.ShapeDtypeStruct((), jnp.int32),
                snapshot.effective_alpha,
                snapshot.cache_alpha,
                snapshot.invalid_actions,
                snapshot.categorical_outcome,
                snapshot.n_down,
                snapshot.gamma,
                snapshot.value_prior,
                snapshot.previous_value_alpha,
                snapshot.active,
                root_id,
                game_cluster_id,
                root_ply,
                root_weight,
                stage_id,
                checkpoint_step,
                context.node.index,
                node_ply,
                jax.random.key_data(update_key),
                ordered=True,
            )
            return base_update(update_key, context)

        return _run_tree(
            env,
            model,
            state,
            rng_key,
            search_config,
            traced_update,
        )

    return untraced, traced


def _tree_batch(tree: Any, start: int, stop: int) -> Any:
    import jax

    return jax.tree.map(lambda leaf: leaf[start:stop], tree)


def _assert_same_tree_outputs(lhs: Any, rhs: Any) -> None:
    import jax

    lhs_leaves, lhs_tree = jax.tree.flatten(lhs)
    rhs_leaves, rhs_tree = jax.tree.flatten(rhs)
    if lhs_tree != rhs_tree:
        raise AssertionError("traced/untraced output structures differ")
    for index, (left, right) in enumerate(
        zip(lhs_leaves, rhs_leaves, strict=True)
    ):
        left = np.asarray(jax.device_get(left))
        right = np.asarray(jax.device_get(right))
        if not np.array_equal(left, right):
            maximum = float(np.max(np.abs(left.astype(float) - right.astype(float))))
            raise AssertionError(
                f"traced/untraced tree field {index} differs; max_abs={maximum}"
            )


def _trace_selected_roots(
    env: Any,
    config: Any,
    model: Any,
    states: Any,
    root_table: dict[str, np.ndarray],
    collector: _ContextCollector,
    *,
    trace_batch_size: int,
    seed: int,
    verify_first_batch: bool,
) -> None:
    import jax
    import jax.numpy as jnp

    untraced, traced = _make_trace_functions(env, config, collector)
    count = len(root_table["root_id"])
    if count % trace_batch_size:
        raise ValueError(
            f"selected root count {count} must be divisible by "
            f"trace batch size {trace_batch_size}"
        )
    for batch_index, start in enumerate(range(0, count, trace_batch_size)):
        stop = start + trace_batch_size
        batch_state = _tree_batch(states, start, stop)
        key = jax.random.fold_in(jax.random.PRNGKey(seed), batch_index)
        metadata = {
            name: jnp.asarray(root_table[name][start:stop])
            for name in (
                "root_id",
                "game_cluster_id",
                "root_ply",
                "root_weight",
                "stage_id",
                "checkpoint_step",
            )
        }
        if verify_first_batch and batch_index == 0:
            expected = jax.block_until_ready(
                untraced(model, batch_state, key)
            )
        actual = jax.block_until_ready(
            traced(
                model,
                batch_state,
                key,
                metadata["root_id"],
                metadata["game_cluster_id"],
                metadata["root_ply"],
                metadata["root_weight"],
                metadata["stage_id"],
                metadata["checkpoint_step"],
            )
        )
        if verify_first_batch and batch_index == 0:
            _assert_same_tree_outputs(expected, actual)


def _merge_tables(
    tables: Iterable[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    tables = tuple(tables)
    if not tables:
        raise ValueError("cannot merge an empty table sequence")
    keys = set(tables[0])
    if any(set(table) != keys for table in tables[1:]):
        raise ValueError("table schemas differ")
    return {
        key: np.concatenate([table[key] for table in tables], axis=0)
        for key in sorted(keys)
    }


def _collect(args: argparse.Namespace) -> None:
    import jax

    from scacchi.types import config_to_dict

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    steps = _parse_steps(args.steps)
    if args.roots_per_stage % args.trace_batch_size:
        raise ValueError(
            "--roots-per-stage must be divisible by --trace-batch-size"
        )

    collector = _ContextCollector()
    root_tables: list[dict[str, np.ndarray]] = []
    checkpoint_metadata: list[dict[str, Any]] = []
    first_root_id = 0
    corpus_config: dict[str, Any] | None = None
    started = time.perf_counter()

    for step in steps:
        config, env, model, progress = _load_checkpoint_config_and_model(
            args.checkpoint,
            step,
        )
        encoded_config = config_to_dict(config)
        if corpus_config is None:
            corpus_config = encoded_config
        else:
            # run.max_num_iters and checkpoint directory can differ after a
            # resume; estimator-relevant selfplay/model fields must not.
            for section in ("env", "model", "selfplay"):
                if corpus_config[section] != encoded_config[section]:
                    raise ValueError(
                        f"checkpoint {step} changed {section} configuration"
                    )

        rollout = _make_on_policy_rollout(
            env,
            config,
            batch_size=args.rollout_batch_size,
            max_steps=args.max_steps,
        )
        rollout_key = jax.random.PRNGKey(args.seed)
        init_keys, states, actions = jax.block_until_ready(
            rollout(model, rollout_key)
        )
        selected_states, root_table, stage_population = _select_roots(
            states,
            actions,
            init_keys,
            checkpoint_step=step,
            roots_per_stage=args.roots_per_stage,
            rng=np.random.default_rng(args.seed + 1_000_003 * step),
            first_root_id=first_root_id,
        )
        replayed_states = jax.block_until_ready(
            _replay_roots(env, root_table)
        )
        _assert_same_states(selected_states, replayed_states)
        _trace_selected_roots(
            env,
            config,
            model,
            replayed_states,
            root_table,
            collector,
            trace_batch_size=args.trace_batch_size,
            seed=args.seed + 10_000_019 * step,
            verify_first_batch=True,
        )
        first_root_id += len(root_table["root_id"])
        root_tables.append(root_table)
        checkpoint_metadata.append(
            {
                "step": step,
                "progress": progress,
                "stage_candidate_frames": stage_population,
                "roots": len(root_table["root_id"]),
            }
        )

    roots = _merge_tables(root_tables)
    contexts = collector.arrays()
    roots_path = output / "roots.npz"
    contexts_path = output / "contexts.npz"
    # NumPy's stub reserves an ``allow_pickle`` keyword, so a dynamic mapping
    # of homogeneous array payloads is conservatively rejected by the type
    # checker even though the runtime API accepts it.
    np.savez_compressed(roots_path, **roots)  # ty: ignore[invalid-argument-type]
    np.savez_compressed(contexts_path, **contexts)  # ty: ignore[invalid-argument-type]
    elapsed = time.perf_counter() - started
    repository_root = Path(__file__).resolve().parents[1]
    manifest = {
        "version": CORPUS_VERSION,
        "created_unix_seconds": time.time(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoints": checkpoint_metadata,
        "config": corpus_config,
        "seed": args.seed,
        "rollout_batch_size": args.rollout_batch_size,
        "max_steps": args.max_steps,
        "roots_per_stage": args.roots_per_stage,
        "trace_batch_size": args.trace_batch_size,
        "stages": [
            {"id": index, "name": name, "lower": lower, "upper": upper}
            for index, (name, lower, upper) in enumerate(STAGES)
        ],
        "root_count": int(len(roots["root_id"])),
        "context_count": int(len(contexts["root_id"])),
        "callback_count": collector.callback_count,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "seconds": elapsed,
        "source_sha256": _source_digest(repository_root),
        "artifacts": {
            "roots.npz": _sha256(roots_path),
            "contexts.npz": _sha256(contexts_path),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_corpus(path: Path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = dict(np.load(path / "roots.npz", allow_pickle=False))
    contexts = dict(np.load(path / "contexts.npz", allow_pickle=False))
    return manifest, roots, contexts


def _verify(args: argparse.Namespace) -> None:
    manifest, roots, contexts = _load_corpus(args.corpus.resolve())
    if int(manifest["version"]) != CORPUS_VERSION:
        raise ValueError(
            f"unsupported corpus version {manifest['version']}"
        )
    for filename, expected in manifest["artifacts"].items():
        actual = _sha256(args.corpus.resolve() / filename)
        if actual != expected:
            raise ValueError(
                f"{filename} hash mismatch: expected {expected}, got {actual}"
            )
    root_count = len(roots["root_id"])
    context_count = len(contexts["root_id"])
    if root_count != int(manifest["root_count"]):
        raise ValueError("root count disagrees with manifest")
    if context_count != int(manifest["context_count"]):
        raise ValueError("context count disagrees with manifest")
    if not np.all(contexts["active"]):
        raise ValueError("stored corpus contains inactive repair rows")
    if not np.all(np.isfinite(contexts["effective_alpha"])):
        raise ValueError("effective alpha contains non-finite values")
    unresolved_legal = (
        ~contexts["invalid_actions"]
        & (contexts["categorical_outcome"] == -1)
    )
    if not np.all(
        contexts["effective_alpha"][unresolved_legal] > 0
    ):
        raise ValueError("unresolved legal alpha is not strictly positive")
    if not np.all(np.isfinite(contexts["cache_alpha"])):
        raise ValueError("cache alpha contains non-finite values")
    if not np.all(np.sum(contexts["cache_alpha"], axis=-1) > 0):
        raise ValueError("cache action alpha has nonpositive mass")
    kappa = float(
        manifest["config"]["selfplay"]["search"]["dirichlet_thompson"][
            "kappa"
        ]
    )
    expected_gamma = contexts["n_down"] / (kappa + contexts["n_down"])
    np.testing.assert_allclose(
        contexts["gamma"],
        expected_gamma,
        rtol=2e-6,
        atol=2e-7,
    )
    if not np.all(contexts["node_ply"] >= contexts["root_ply"]):
        raise ValueError("node ply precedes its root ply")
    root_ids = set(int(value) for value in roots["root_id"])
    if not set(int(value) for value in contexts["root_id"]) <= root_ids:
        raise ValueError("context references an unknown root")

    summary = {
        "root_count": root_count,
        "context_count": context_count,
        "live_message_fraction": float(
            np.mean(contexts["n_down"] > 0)
        ),
        "root_context_fraction": float(
            np.mean(contexts["node_index"] == 0)
        ),
        "max_node_depth": int(
            np.max(contexts["node_ply"] - contexts["root_ply"])
        ),
        "checkpoint_steps": sorted(
            int(value) for value in np.unique(contexts["checkpoint_step"])
        ),
        "verified": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _weighted_group_summary(
    noise: np.ndarray,
    repair_signal: np.ndarray,
    prior_signal: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    noise_total = float(np.sum(weights * noise))
    repair_total = float(np.sum(weights * repair_signal))
    prior_total = float(np.sum(weights * prior_signal))
    denominator = noise_total + repair_total
    return {
        "weighted_noise_energy": noise_total,
        "weighted_incremental_signal_energy": repair_total,
        "weighted_prior_signal_energy": prior_total,
        "rho_incremental": (
            float(np.sqrt(noise_total / repair_total))
            if repair_total > 0
            else (float("inf") if noise_total > 0 else 0.0)
        ),
        "eta_incremental": (
            noise_total / denominator if denominator > 0 else 0.0
        ),
        "rho_prior": (
            float(np.sqrt(noise_total / prior_total))
            if prior_total > 0
            else (float("inf") if noise_total > 0 else 0.0)
        ),
    }


def _cluster_bootstrap(
    *,
    root_id: np.ndarray,
    stratum: np.ndarray,
    noise: np.ndarray,
    signal: np.ndarray,
    weights: np.ndarray,
    tail: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique_roots = np.unique(root_id)
    root_rows: list[tuple[int, int, float, float, float, float]] = []
    for root in unique_roots:
        mask = root_id == root
        root_strata = np.unique(stratum[mask])
        if len(root_strata) != 1:
            raise ValueError(f"root {root} spans multiple strata")
        root_rows.append(
            (
                int(root),
                int(root_strata[0]),
                float(np.sum(weights[mask] * noise[mask])),
                float(np.sum(weights[mask] * signal[mask])),
                float(np.sum(weights[mask] * tail[mask])),
                float(np.sum(weights[mask])),
            )
        )
    rows = np.asarray(root_rows, dtype=np.float64)
    strata = np.unique(rows[:, 1].astype(np.int64))
    rho_samples = np.empty(repetitions, dtype=np.float64)
    tail_samples = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled_parts: list[np.ndarray] = []
        for value in strata:
            part = rows[rows[:, 1] == value]
            sampled_parts.append(
                part[rng.integers(0, len(part), size=len(part))]
            )
        sample = np.concatenate(sampled_parts, axis=0)
        noise_total = float(np.sum(sample[:, 2]))
        signal_total = float(np.sum(sample[:, 3]))
        rho_samples[repetition] = (
            np.sqrt(noise_total / signal_total)
            if signal_total > 0
            else np.inf
        )
        tail_samples[repetition] = (
            float(np.sum(sample[:, 4]) / np.sum(sample[:, 5]))
            if np.sum(sample[:, 5]) > 0
            else 0.0
        )
    return {
        "rho_incremental_lcb95": float(np.quantile(rho_samples, 0.025)),
        "rho_incremental_ucb95": float(np.quantile(rho_samples, 0.975)),
        "tail_fraction_lcb95": float(np.quantile(tail_samples, 0.025)),
        "tail_fraction_ucb95": float(np.quantile(tail_samples, 0.975)),
        "repetitions": repetitions,
    }


def _analyze(args: argparse.Namespace) -> None:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    import functools

    import jax
    import jax.numpy as jnp

    from scacchi.dirichlet_mctx.estimator_diagnostics import (
        analytic_cache_noise,
        binary_posterior_best_policy_quadrature,
    )

    manifest, roots, contexts = _load_corpus(args.corpus.resolve())
    config = manifest["config"]
    settings = config["selfplay"]["search"]["dirichlet_thompson"]
    kappa = float(settings["kappa"])
    posterior_samples = settings.get("posterior_policy_samples")
    if posterior_samples is None:
        posterior_samples = settings["policy_samples"]
    posterior_samples = int(posterior_samples)

    legal = ~contexts["invalid_actions"]
    unresolved = legal & (contexts["categorical_outcome"] == -1)
    certified_win = legal & (contexts["categorical_outcome"] == 1)
    has_unresolved = np.any(unresolved, axis=-1)
    imminent_categorical = np.any(certified_win, axis=-1) | ~has_unresolved
    live = (
        contexts["active"]
        & (contexts["n_down"] > 0)
        & ~imminent_categorical
    )
    live_indices = np.flatnonzero(live)
    if live_indices.size == 0:
        raise RuntimeError("corpus has no live unresolved repair contexts")

    reference_function = jax.jit(
        functools.partial(
            binary_posterior_best_policy_quadrature,
            half_width=args.quadrature_half_width,
            step=args.quadrature_step,
        )
    )
    noise_function = jax.jit(
        functools.partial(
            analytic_cache_noise,
            kappa=kappa,
            num_samples=posterior_samples,
        )
    )
    metric_chunks: dict[str, list[np.ndarray]] = {}
    reference_mass: list[np.ndarray] = []
    reference_error: list[np.ndarray] = []
    reference_finite: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(live_indices), args.batch_size):
        index = live_indices[start : start + args.batch_size]
        alpha = jnp.asarray(
            contexts["effective_alpha"][index],
            dtype=jnp.float64,
        )
        invalid = jnp.asarray(contexts["invalid_actions"][index])
        categorical = jnp.asarray(contexts["categorical_outcome"][index])
        reference = jax.block_until_ready(
            reference_function(alpha, invalid, categorical)
        )
        diagnostics = jax.block_until_ready(
            noise_function(
                reference.policy,
                jnp.asarray(
                    contexts["cache_alpha"][index],
                    dtype=jnp.float64,
                ),
                jnp.asarray(
                    contexts["value_prior"][index],
                    dtype=jnp.float64,
                ),
                jnp.asarray(contexts["n_down"][index]),
                previous_value_alpha=jnp.asarray(
                    contexts["previous_value_alpha"][index],
                    dtype=jnp.float64,
                ),
            )
        )
        reference_mass.append(np.asarray(reference.raw_mass))
        reference_error.append(np.asarray(reference.normalization_error))
        reference_finite.append(np.asarray(reference.finite))
        for name, value in diagnostics._asdict().items():
            if value.ndim != 1:
                continue
            metric_chunks.setdefault(name, []).append(np.asarray(value))

    metrics = {
        name: np.concatenate(values)
        for name, values in metric_chunks.items()
    }
    raw_mass = np.concatenate(reference_mass)
    normalization_error = np.concatenate(reference_error)
    finite = np.concatenate(reference_finite)
    elapsed = time.perf_counter() - started
    if not np.all(finite):
        raise FloatingPointError(
            f"quadrature produced {np.sum(~finite)} non-finite rows"
        )
    if float(np.max(normalization_error)) > args.max_normalization_error:
        raise FloatingPointError(
            "quadrature normalization error exceeds gate: "
            f"{np.max(normalization_error)} > "
            f"{args.max_normalization_error}"
        )

    noise = np.maximum(metrics["raw_alpha_mse"], 0.0)
    repair_signal = np.maximum(metrics["repair_squared_l2"], 0.0)
    prior_signal = np.maximum(metrics["raw_update_squared_l2"], 0.0)
    weights = contexts["root_weight"][live_indices].astype(np.float64)
    rms_ratio = np.divide(
        np.sqrt(noise),
        np.sqrt(repair_signal),
        out=np.full_like(noise, np.inf),
        where=repair_signal > 0,
    )
    tail = rms_ratio >= args.tail_rms_ratio
    overall = _weighted_group_summary(
        noise,
        repair_signal,
        prior_signal,
        weights,
    )
    overall["tail_fraction"] = float(
        np.sum(weights * tail) / np.sum(weights)
    )

    stage = contexts["stage_id"][live_indices].astype(np.int64)
    checkpoint = contexts["checkpoint_step"][live_indices].astype(np.int64)
    # Resample whole on-policy games, including every early/mid/late root
    # selected from one trajectory.  Checkpoint remains the bootstrap stratum.
    stratum = checkpoint
    bootstrap = _cluster_bootstrap(
        root_id=contexts["game_cluster_id"][live_indices],
        stratum=stratum,
        noise=noise,
        signal=repair_signal,
        weights=weights,
        tail=tail,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    materiality_pass = (
        bootstrap["rho_incremental_lcb95"] >= args.minimum_rho
        or bootstrap["tail_fraction_lcb95"] >= args.minimum_tail_fraction
    )

    groups: dict[str, Any] = {}
    for step in sorted(np.unique(checkpoint)):
        mask = checkpoint == step
        groups[f"checkpoint_{step}"] = _weighted_group_summary(
            noise[mask],
            repair_signal[mask],
            prior_signal[mask],
            weights[mask],
        )
    for stage_id, (name, _, _) in enumerate(STAGES):
        mask = stage == stage_id
        groups[f"stage_{name}"] = _weighted_group_summary(
            noise[mask],
            repair_signal[mask],
            prior_signal[mask],
            weights[mask],
        )

    output = {
        "corpus": str(args.corpus.resolve()),
        "reference": {
            "kind": "fixed_sinh_logit_exact_beta",
            "half_width": args.quadrature_half_width,
            "step": args.quadrature_step,
            "raw_mass_mean": float(np.mean(raw_mass)),
            "normalization_error_p95": float(
                np.quantile(normalization_error, 0.95)
            ),
            "normalization_error_max": float(
                np.max(normalization_error)
            ),
            "finite_fraction": float(np.mean(finite)),
        },
        "population": {
            "all_active_contexts": int(len(contexts["root_id"])),
            "live_unresolved_contexts": int(len(live_indices)),
            "imminent_categorical_contexts": int(
                np.sum(contexts["active"] & imminent_categorical)
            ),
            "zero_message_contexts": int(
                np.sum(contexts["active"] & (contexts["n_down"] <= 0))
            ),
            "unique_live_roots": int(
                len(np.unique(contexts["root_id"][live_indices]))
            ),
        },
        "m32": {
            "population_samples": posterior_samples,
            "overall": overall,
            "bootstrap": bootstrap,
            "groups": groups,
            "semantic_delta_mse_weighted_mean": float(
                np.average(
                    np.maximum(metrics["semantic_mean_delta_mse"], 0.0),
                    weights=weights,
                )
            ),
            "concentration_mse_weighted_mean": float(
                np.average(
                    np.maximum(metrics["concentration_mse"], 0.0),
                    weights=weights,
                )
            ),
        },
        "materiality_gate": {
            "minimum_rho_lcb95": args.minimum_rho,
            "tail_rms_ratio": args.tail_rms_ratio,
            "minimum_tail_fraction_lcb95": args.minimum_tail_fraction,
            "passed": bool(materiality_pass),
            "decision": (
                "benchmark replacement estimators"
                if materiality_pass
                else "stop E4: current M32 cache noise is not material"
            ),
        },
        "seconds": elapsed,
        "backend": jax.default_backend(),
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else args.corpus.resolve() / "analysis.json"
    )
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


def _benchmark_live_indices(
    contexts: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    legal = ~contexts["invalid_actions"]
    unresolved = legal & (contexts["categorical_outcome"] == -1)
    certified_win = legal & (contexts["categorical_outcome"] == 1)
    has_unresolved = np.any(unresolved, axis=-1)
    imminent_categorical = np.any(certified_win, axis=-1) | ~has_unresolved
    live = (
        contexts["active"]
        & (contexts["n_down"] > 0)
        & ~imminent_categorical
    )
    return np.flatnonzero(live), imminent_categorical


def _stratified_context_limit(
    indices: np.ndarray,
    contexts: dict[str, np.ndarray],
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Deterministically retain checkpoint/stage coverage for smoke runs."""

    if maximum <= 0 or len(indices) <= maximum:
        return indices
    checkpoint = contexts["checkpoint_step"][indices].astype(np.int64)
    stage = contexts["stage_id"][indices].astype(np.int64)
    cell = checkpoint * 16 + stage
    cells = np.unique(cell)
    if maximum < len(cells):
        raise ValueError(
            f"--max-contexts={maximum} cannot cover {len(cells)} "
            "checkpoint/stage cells"
        )
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    base = maximum // len(cells)
    remainder = maximum % len(cells)
    for ordinal, value in enumerate(cells):
        candidates = indices[cell == value]
        count = min(len(candidates), base + (ordinal < remainder))
        chosen = rng.choice(candidates, size=count, replace=False)
        selected.append(np.sort(chosen))
    result = np.sort(np.concatenate(selected))
    if len(result) < maximum:
        remaining = np.setdiff1d(indices, result, assume_unique=True)
        extra_count = min(maximum - len(result), len(remaining))
        extra = rng.choice(remaining, size=extra_count, replace=False)
        result = np.sort(np.concatenate([result, extra]))
    return result


def _semantic_distribution(alpha: np.ndarray) -> np.ndarray:
    concentration = np.sum(alpha, axis=-1, keepdims=True)
    return np.divide(
        alpha,
        concentration,
        out=np.zeros_like(alpha, dtype=np.float64),
        where=concentration > 0,
    )


def _cache_from_policy(
    policy: np.ndarray,
    cache_alpha: np.ndarray,
    value_prior: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    descendant = np.sum(policy[..., None] * cache_alpha, axis=-2)
    return (
        (1.0 - gamma)[..., None] * value_prior
        + gamma[..., None] * descendant
    )


def _jensen_shannon_rows(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    midpoint = 0.5 * (lhs + rhs)

    def kl(distribution: np.ndarray) -> np.ndarray:
        terms = np.where(
            distribution > 0,
            distribution
            * (
                np.log(np.maximum(distribution, np.finfo(np.float64).tiny))
                - np.log(np.maximum(midpoint, np.finfo(np.float64).tiny))
            ),
            0.0,
        )
        return np.sum(terms, axis=-1)

    return 0.5 * (kl(lhs) + kl(rhs))


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    positive = weights > 0
    if not np.any(positive):
        return 0.0
    if np.any(~np.isfinite(values[positive])):
        return float("inf")
    return float(np.average(values[positive], weights=weights[positive]))


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    positive = weights > 0
    values = values[positive]
    weights = weights[positive]
    if len(values) == 0:
        return 0.0
    if np.any(~np.isfinite(values)):
        return float("inf")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = quantile * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, target, side="left")),
        len(sorted_values) - 1,
    )
    return float(sorted_values[index])


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return float("inf") if numerator > 0 else 0.0


def _paired_cluster_ratio_bootstrap(
    *,
    cluster_id: np.ndarray,
    bootstrap_stratum: np.ndarray,
    candidate_raw: np.ndarray,
    baseline_raw: np.ndarray,
    candidate_semantic: np.ndarray,
    baseline_semantic: np.ndarray,
    weights: np.ndarray,
    repetitions: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | str]:
    """Resample complete games and preserve candidate/baseline pairing."""

    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be in (0.5, 1)")
    unique_clusters = np.unique(cluster_id)
    rows: list[tuple[int, int, float, float, float, float]] = []
    for cluster in unique_clusters:
        mask = cluster_id == cluster
        values = np.unique(bootstrap_stratum[mask])
        if len(values) != 1:
            raise ValueError(
                f"game cluster {cluster} spans bootstrap strata {values}"
            )
        rows.append(
            (
                int(cluster),
                int(values[0]),
                float(np.sum(weights[mask] * candidate_raw[mask])),
                float(np.sum(weights[mask] * baseline_raw[mask])),
                float(np.sum(weights[mask] * candidate_semantic[mask])),
                float(np.sum(weights[mask] * baseline_semantic[mask])),
            )
        )
    aggregate = np.asarray(rows, dtype=np.float64)
    strata = np.unique(aggregate[:, 1].astype(np.int64))
    raw_samples = np.empty(repetitions, dtype=np.float64)
    semantic_samples = np.empty(repetitions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for repetition in range(repetitions):
        sampled: list[np.ndarray] = []
        for stratum in strata:
            part = aggregate[aggregate[:, 1] == stratum]
            sampled.append(
                part[rng.integers(0, len(part), size=len(part))]
            )
        sample = np.concatenate(sampled, axis=0)
        raw_samples[repetition] = _ratio(
            float(np.sum(sample[:, 2])),
            float(np.sum(sample[:, 3])),
        )
        semantic_samples[repetition] = _ratio(
            float(np.sum(sample[:, 4])),
            float(np.sum(sample[:, 5])),
        )
    return {
        "method": "paired_stratified_whole_game_cluster_bootstrap",
        "clusters": len(unique_clusters),
        "repetitions": repetitions,
        "confidence": confidence,
        "raw_mse_ratio_ucb": float(
            np.quantile(raw_samples, confidence)
        ),
        "semantic_mse_ratio_ucb": float(
            np.quantile(semantic_samples, confidence)
        ),
        "raw_mse_ratio_median": float(np.median(raw_samples)),
        "semantic_mse_ratio_median": float(
            np.median(semantic_samples)
        ),
    }


def _benchmark_reference(
    *,
    contexts: dict[str, np.ndarray],
    indices: np.ndarray,
    kappa: float,
    m32_samples: int,
    batch_size: int,
    half_width: int,
    step: float,
) -> dict[str, np.ndarray]:
    import functools

    import jax
    import jax.numpy as jnp

    from scacchi.dirichlet_mctx.estimator_diagnostics import (
        analytic_cache_noise,
        binary_posterior_best_policy_quadrature,
    )

    reference_kernel = jax.jit(
        functools.partial(
            binary_posterior_best_policy_quadrature,
            half_width=half_width,
            step=step,
        )
    )
    baseline_kernel = jax.jit(
        functools.partial(
            analytic_cache_noise,
            kappa=kappa,
            num_samples=m32_samples,
        )
    )
    chunks: dict[str, list[np.ndarray]] = {
        "policy": [],
        "cache": [],
        "semantic": [],
        "normalization_error": [],
        "finite": [],
        "m32_raw_mse": [],
        "m32_semantic_mse": [],
    }
    for start in range(0, len(indices), batch_size):
        index = indices[start : start + batch_size]
        alpha = jnp.asarray(
            contexts["effective_alpha"][index],
            dtype=jnp.float64,
        )
        invalid = jnp.asarray(contexts["invalid_actions"][index])
        categorical = jnp.asarray(contexts["categorical_outcome"][index])
        reference = jax.block_until_ready(
            reference_kernel(alpha, invalid, categorical)
        )
        diagnostics = jax.block_until_ready(
            baseline_kernel(
                reference.policy,
                jnp.asarray(
                    contexts["cache_alpha"][index],
                    dtype=jnp.float64,
                ),
                jnp.asarray(
                    contexts["value_prior"][index],
                    dtype=jnp.float64,
                ),
                jnp.asarray(contexts["n_down"][index]),
                previous_value_alpha=jnp.asarray(
                    contexts["previous_value_alpha"][index],
                    dtype=jnp.float64,
                ),
            )
        )
        policy = np.asarray(reference.policy, dtype=np.float64)
        cache = np.asarray(
            diagnostics.exact_cache_alpha,
            dtype=np.float64,
        )
        chunks["policy"].append(policy)
        chunks["cache"].append(cache)
        chunks["semantic"].append(_semantic_distribution(cache))
        chunks["normalization_error"].append(
            np.asarray(reference.normalization_error, dtype=np.float64)
        )
        chunks["finite"].append(np.asarray(reference.finite, dtype=bool))
        chunks["m32_raw_mse"].append(
            np.asarray(diagnostics.raw_alpha_mse, dtype=np.float64)
        )
        chunks["m32_semantic_mse"].append(
            np.asarray(
                diagnostics.semantic_mean_delta_mse,
                dtype=np.float64,
            )
        )
    return {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }


def _evaluate_estimator_candidate(
    *,
    kernel: Callable[..., Any],
    stochastic_repetitions: int,
    contexts: dict[str, np.ndarray],
    indices: np.ndarray,
    reference: dict[str, np.ndarray],
    batch_size: int,
    dtype_name: str,
    seed: int,
    simplex_tolerance: float,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    dtype = jnp.float32 if dtype_name == "float32" else jnp.float64
    count = len(indices)
    metric_names = (
        "raw_cache_squared_error",
        "semantic_squared_error",
        "policy_l1_error",
        "policy_js_nats",
        "argmax_disagreement",
    )
    metrics = {
        name: np.zeros((count,), dtype=np.float64)
        for name in metric_names
    }
    max_normalization_error = np.zeros((count,), dtype=np.float64)
    any_failure = np.zeros((count,), dtype=bool)
    nonfinite_observations = 0
    simplex_failure_observations = 0
    observation_count = count * stochastic_repetitions
    coordinate_count_min: int | None = None
    coordinate_count_max: int | None = None
    coordinate_imbalance_max: int | None = None
    prefix_grid_half_range_min: float | None = None
    prefix_grid_half_range_max: float | None = None
    prefix_tail_clipped_observations = 0
    prefix_observations = 0
    prefix_density_log_integral_errors: list[np.ndarray] = []
    prefix_fallback_intervals = 0
    prefix_fallback_rows = 0
    base_key = jax.random.PRNGKey(seed)

    for repetition in range(stochastic_repetitions):
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            index = indices[start:stop]
            key = jax.random.fold_in(
                base_key,
                repetition * count + start,
            )
            result = jax.block_until_ready(
                kernel(
                    key,
                    jnp.asarray(
                        contexts["effective_alpha"][index],
                        dtype=dtype,
                    ),
                    jnp.asarray(contexts["invalid_actions"][index]),
                    jnp.asarray(contexts["categorical_outcome"][index]),
                )
            )
            policy = np.asarray(result.policy, dtype=np.float64)
            result_finite = np.asarray(result.finite, dtype=bool)
            normalization_error = np.asarray(
                result.normalization_error,
                dtype=np.float64,
            )
            finite_policy = np.all(np.isfinite(policy), axis=-1)
            nonnegative = np.all(policy >= -simplex_tolerance, axis=-1)
            invalid = contexts["invalid_actions"][index]
            invalid_mass = np.sum(
                np.where(invalid, np.abs(policy), 0.0),
                axis=-1,
            )
            simplex_error = np.abs(np.sum(policy, axis=-1) - 1.0)
            nonfinite = (
                ~result_finite
                | ~finite_policy
                | ~np.isfinite(normalization_error)
            )
            simplex_failure = (
                ~nonnegative
                | (invalid_mass > simplex_tolerance)
                | (simplex_error > simplex_tolerance)
            )
            failure = nonfinite | simplex_failure
            nonfinite_observations += int(np.sum(nonfinite))
            simplex_failure_observations += int(np.sum(simplex_failure))
            any_failure[start:stop] |= failure
            max_normalization_error[start:stop] = np.maximum(
                max_normalization_error[start:stop],
                np.where(
                    np.isfinite(normalization_error),
                    normalization_error,
                    np.inf,
                ),
            )

            grid_half_range = getattr(result, "grid_half_range", None)
            tail_range_clipped = getattr(
                result,
                "tail_range_clipped",
                None,
            )
            if (
                grid_half_range is not None
                and tail_range_clipped is not None
            ):
                ranges = np.asarray(grid_half_range, dtype=np.float64)
                clipped = np.asarray(tail_range_clipped, dtype=bool)
                range_min = float(np.min(ranges))
                range_max = float(np.max(ranges))
                prefix_grid_half_range_min = (
                    range_min
                    if prefix_grid_half_range_min is None
                    else min(prefix_grid_half_range_min, range_min)
                )
                prefix_grid_half_range_max = (
                    range_max
                    if prefix_grid_half_range_max is None
                    else max(prefix_grid_half_range_max, range_max)
                )
                prefix_tail_clipped_observations += int(
                    np.sum(clipped)
                )
                prefix_observations += int(clipped.size)
            density_log_integral = getattr(
                result,
                "density_log_integral",
                None,
            )
            if density_log_integral is not None:
                log_integral = np.asarray(
                    density_log_integral,
                    dtype=np.float64,
                )
                unresolved = (
                    ~contexts["invalid_actions"][index]
                    & (contexts["categorical_outcome"][index] == -1)
                )
                prefix_density_log_integral_errors.append(
                    np.abs(log_integral[unresolved])
                )
            fallback_interval_count = getattr(
                result,
                "fallback_interval_count",
                None,
            )
            if fallback_interval_count is not None:
                fallback_count = np.asarray(
                    fallback_interval_count,
                    dtype=np.int64,
                )
                prefix_fallback_intervals += int(
                    np.sum(fallback_count)
                )
                prefix_fallback_rows += int(
                    np.sum(fallback_count > 0)
                )

            safe_policy = np.where(np.isfinite(policy), policy, 0.0)
            cache = _cache_from_policy(
                safe_policy,
                contexts["cache_alpha"][index].astype(np.float64),
                contexts["value_prior"][index].astype(np.float64),
                contexts["gamma"][index].astype(np.float64),
            )
            semantic = _semantic_distribution(cache)
            raw_error = np.sum(
                (cache - reference["cache"][start:stop]) ** 2,
                axis=-1,
            )
            semantic_error = np.sum(
                (semantic - reference["semantic"][start:stop]) ** 2,
                axis=-1,
            )
            policy_l1 = np.sum(
                np.abs(safe_policy - reference["policy"][start:stop]),
                axis=-1,
            )
            policy_js = _jensen_shannon_rows(
                safe_policy,
                reference["policy"][start:stop],
            )
            argmax_disagreement = (
                np.argmax(safe_policy, axis=-1)
                != np.argmax(reference["policy"][start:stop], axis=-1)
            ).astype(np.float64)
            for name, value in (
                ("raw_cache_squared_error", raw_error),
                ("semantic_squared_error", semantic_error),
                ("policy_l1_error", policy_l1),
                ("policy_js_nats", policy_js),
                ("argmax_disagreement", argmax_disagreement),
            ):
                metrics[name][start:stop] += (
                    np.where(failure, np.inf, value)
                    / stochastic_repetitions
                )

            coordinate_counts = getattr(result, "coordinate_counts", None)
            if coordinate_counts is not None:
                counts = np.asarray(coordinate_counts, dtype=np.int64)
                eligible = (
                    ~contexts["invalid_actions"][index]
                    & (
                        contexts["categorical_outcome"][index]
                        == -1
                    )
                )
                eligible_counts = counts[eligible]
                if eligible_counts.size == 0:
                    continue
                minimum = int(np.min(eligible_counts))
                maximum = int(np.max(eligible_counts))
                coordinate_count_min = (
                    minimum
                    if coordinate_count_min is None
                    else min(coordinate_count_min, minimum)
                )
                coordinate_count_max = (
                    maximum
                    if coordinate_count_max is None
                    else max(coordinate_count_max, maximum)
                )
                row_imbalances = [
                    int(np.max(row_counts[row_eligible]))
                    - int(np.min(row_counts[row_eligible]))
                    for row_counts, row_eligible in zip(
                        counts,
                        eligible,
                        strict=True,
                    )
                    if np.any(row_eligible)
                ]
                if row_imbalances:
                    maximum_imbalance = max(row_imbalances)
                    coordinate_imbalance_max = (
                        maximum_imbalance
                        if coordinate_imbalance_max is None
                        else max(
                            coordinate_imbalance_max,
                            maximum_imbalance,
                        )
                    )

    density_integral_errors = (
        np.concatenate(prefix_density_log_integral_errors)
        if prefix_density_log_integral_errors
        else np.empty((0,), dtype=np.float64)
    )
    return {
        **metrics,
        "max_normalization_error": max_normalization_error,
        "any_failure": any_failure,
        "failure_diagnostics": {
            "observations": observation_count,
            "nonfinite_observations": nonfinite_observations,
            "simplex_failure_observations": simplex_failure_observations,
            "rows_with_any_failure": int(np.sum(any_failure)),
            "coordinate_count_min": coordinate_count_min,
            "coordinate_count_max": coordinate_count_max,
            "coordinate_imbalance_max": coordinate_imbalance_max,
            "coordinate_count_spread": (
                None
                if (
                    coordinate_count_min is None
                    or coordinate_count_max is None
                )
                else coordinate_count_max - coordinate_count_min
            ),
            "prefix_grid_half_range_min": prefix_grid_half_range_min,
            "prefix_grid_half_range_max": prefix_grid_half_range_max,
            "prefix_tail_clipped_observations": (
                prefix_tail_clipped_observations
            ),
            "prefix_observations": prefix_observations,
            "prefix_tail_clipped_fraction": (
                None
                if prefix_observations == 0
                else (
                    prefix_tail_clipped_observations
                    / prefix_observations
                )
            ),
            "prefix_density_log_integral_abs_mean": (
                None
                if density_integral_errors.size == 0
                else float(np.mean(density_integral_errors))
            ),
            "prefix_density_log_integral_abs_p95": (
                None
                if density_integral_errors.size == 0
                else float(
                    np.quantile(density_integral_errors, 0.95)
                )
            ),
            "prefix_density_log_integral_abs_p99": (
                None
                if density_integral_errors.size == 0
                else float(
                    np.quantile(density_integral_errors, 0.99)
                )
            ),
            "prefix_density_log_integral_abs_max": (
                None
                if density_integral_errors.size == 0
                else float(np.max(density_integral_errors))
            ),
            "prefix_fallback_intervals": prefix_fallback_intervals,
            "prefix_fallback_rows": prefix_fallback_rows,
        },
    }


def _candidate_group_summary(
    *,
    mask: np.ndarray,
    evaluation: dict[str, Any],
    reference: dict[str, np.ndarray],
    contexts: dict[str, np.ndarray],
    indices: np.ndarray,
    repetitions: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    weights = contexts["root_weight"][indices][mask].astype(np.float64)
    raw_error = evaluation["raw_cache_squared_error"][mask]
    semantic_error = evaluation["semantic_squared_error"][mask]
    baseline_raw = reference["m32_raw_mse"][mask]
    baseline_semantic = reference["m32_semantic_mse"][mask]
    candidate_raw_mean = _weighted_average(raw_error, weights)
    candidate_semantic_mean = _weighted_average(semantic_error, weights)
    baseline_raw_mean = _weighted_average(baseline_raw, weights)
    baseline_semantic_mean = _weighted_average(
        baseline_semantic,
        weights,
    )
    bootstrap = _paired_cluster_ratio_bootstrap(
        cluster_id=contexts["game_cluster_id"][indices][mask],
        bootstrap_stratum=contexts["checkpoint_step"][indices][mask],
        candidate_raw=raw_error,
        baseline_raw=baseline_raw,
        candidate_semantic=semantic_error,
        baseline_semantic=baseline_semantic,
        weights=weights,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
    )
    return {
        "contexts": int(np.sum(mask)),
        "whole_game_clusters": int(
            len(np.unique(contexts["game_cluster_id"][indices][mask]))
        ),
        "candidate_raw_cache_mse": candidate_raw_mean,
        "analytic_m32_raw_cache_mse": baseline_raw_mean,
        "raw_cache_mse_ratio_to_m32": _ratio(
            candidate_raw_mean,
            baseline_raw_mean,
        ),
        "candidate_semantic_mse": candidate_semantic_mean,
        "analytic_m32_semantic_delta_mse": baseline_semantic_mean,
        "semantic_mse_ratio_to_m32": _ratio(
            candidate_semantic_mean,
            baseline_semantic_mean,
        ),
        "raw_cache_squared_error_p95": _weighted_quantile(
            raw_error,
            weights,
            0.95,
        ),
        "semantic_squared_error_p95": _weighted_quantile(
            semantic_error,
            weights,
            0.95,
        ),
        "policy_l1_mean": _weighted_average(
            evaluation["policy_l1_error"][mask],
            weights,
        ),
        "policy_l1_p95": _weighted_quantile(
            evaluation["policy_l1_error"][mask],
            weights,
            0.95,
        ),
        "policy_l1_p99": _weighted_quantile(
            evaluation["policy_l1_error"][mask],
            weights,
            0.99,
        ),
        "policy_l1_max": float(
            np.max(evaluation["policy_l1_error"][mask])
        ),
        "policy_js_nats_mean": _weighted_average(
            evaluation["policy_js_nats"][mask],
            weights,
        ),
        "argmax_disagreement_fraction": _weighted_average(
            evaluation["argmax_disagreement"][mask],
            weights,
        ),
        "row_failure_fraction": _weighted_average(
            evaluation["any_failure"][mask].astype(np.float64),
            weights,
        ),
        "raw_normalization_error_p95": _weighted_quantile(
            evaluation["max_normalization_error"][mask],
            weights,
            0.95,
        ),
        "raw_normalization_error_max": float(
            np.max(evaluation["max_normalization_error"][mask])
        ),
        "bootstrap": bootstrap,
    }


def _time_estimator_kernel(
    *,
    kernel: Callable[..., Any],
    alpha: Any,
    invalid_actions: Any,
    categorical_outcome: Any,
    repetitions: int,
    warmups: int,
    seed: int,
) -> dict[str, float | int]:
    import jax

    if repetitions < 1 or warmups < 0:
        raise ValueError("timing repetitions must be positive and warmups nonnegative")
    key = jax.random.PRNGKey(seed)
    started = time.perf_counter()
    jax.block_until_ready(
        kernel(key, alpha, invalid_actions, categorical_outcome)
    )
    compile_and_first = time.perf_counter() - started
    for index in range(warmups):
        warm_key = jax.random.fold_in(key, index + 1)
        jax.block_until_ready(
            kernel(
                warm_key,
                alpha,
                invalid_actions,
                categorical_outcome,
            )
        )
    samples = np.empty((repetitions,), dtype=np.float64)
    for index in range(repetitions):
        sample_key = jax.random.fold_in(key, warmups + index + 1)
        started = time.perf_counter()
        jax.block_until_ready(
            kernel(
                sample_key,
                alpha,
                invalid_actions,
                categorical_outcome,
            )
        )
        samples[index] = time.perf_counter() - started
    batch_size = int(alpha.shape[0])
    median = float(np.median(samples))
    return {
        "batch_size": batch_size,
        "compile_and_first_seconds": compile_and_first,
        "warm_repetitions": repetitions,
        "warmup_repetitions": warmups,
        "warm_seconds_mean": float(np.mean(samples)),
        "warm_seconds_median": median,
        "warm_seconds_p95": float(np.quantile(samples, 0.95)),
        "warm_rows_per_second_at_median": batch_size / median,
    }


def _subset_table(
    table: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: value[indices] for name, value in table.items()}


def _time_complete_search_estimators(
    *,
    candidate_specs: list[dict[str, Any]],
    manifest: dict[str, Any],
    roots: dict[str, np.ndarray],
    step: int,
    root_count: int,
    repetitions: int,
    warmups: int,
    seed: int,
    dtype_name: str,
) -> dict[str, Any]:
    """Time candidate estimators in the complete native search loop."""

    import functools

    from flax import nnx
    import jax
    import jax.numpy as jnp

    from scacchi.dirichlet_mctx.action_selection import posterior_best_policy
    from scacchi.dirichlet_mctx.posterior_updates import (
        update_posterior_with_estimator,
    )

    config, env, model, _ = _load_checkpoint_config_and_model(
        Path(manifest["checkpoint"]),
        step,
    )
    available = np.flatnonzero(roots["checkpoint_step"] == step)
    if len(available) == 0:
        raise ValueError(
            f"corpus has no roots at checkpoint step {step}"
        )
    # Cycle through stages so a small timing batch is not an early-only batch.
    stage_rows = [
        list(
            int(value)
            for value in available[
                roots["stage_id"][available] == stage_id
            ]
        )
        for stage_id, _ in enumerate(STAGES)
    ]
    chosen: list[int] = []
    for row_index in range(max(len(values) for values in stage_rows)):
        for values in stage_rows:
            if row_index < len(values):
                chosen.append(values[row_index])
    chosen_indices = np.resize(
        np.asarray(chosen, dtype=np.int64),
        root_count,
    )
    unique_timing_roots = len(np.unique(chosen_indices))
    state = jax.block_until_ready(
        _replay_roots(env, _subset_table(roots, chosen_indices))
    )
    search_config = config.selfplay.search.dirichlet_thompson
    posterior_samples = (
        int(search_config.policy_samples)
        if search_config.posterior_policy_samples is None
        else int(search_config.posterior_policy_samples)
    )
    chunk_size = (
        4
        if search_config.policy_sample_chunk_size is None
        else int(search_config.policy_sample_chunk_size)
    )
    kappa = float(search_config.kappa)
    dtype = jnp.float32 if dtype_name == "float32" else jnp.float64

    def baseline_estimator(estimator_key, snapshot):
        return posterior_best_policy(
            estimator_key,
            snapshot.effective_alpha,
            snapshot.invalid_actions,
            max(1, posterior_samples),
            chunk_size=max(1, chunk_size),
            categorical_outcome=snapshot.categorical_outcome,
        )

    def make_search(estimator):
        update = functools.partial(
            update_posterior_with_estimator,
            estimator=estimator,
            kappa=kappa,
        )

        @nnx.jit
        def search(model, root_state, key):
            return _run_tree(
                env,
                model,
                root_state,
                key,
                search_config,
                update,
                public_policy_samples=max(
                    1,
                    int(search_config.policy_samples),
                ),
                public_policy_sample_chunk_size=max(1, chunk_size),
            )

        return search

    searches: dict[str, Any] = {
        "m32": make_search(baseline_estimator),
    }
    for spec in candidate_specs:
        candidate_kernel = spec["kernel"]

        def estimator(estimator_key, snapshot, kernel=candidate_kernel):
            result = kernel(
                estimator_key,
                snapshot.effective_alpha.astype(dtype),
                snapshot.invalid_actions,
                snapshot.categorical_outcome,
            )
            return result.policy.astype(snapshot.effective_alpha.dtype)

        searches[spec["name"]] = make_search(estimator)

    # Compile every kernel before timing, then use an identical key sequence
    # and interleave methods.  This pairs stochastic tree depth and reduces
    # thermal/order confounding.
    base_key = jax.random.PRNGKey(seed)
    compile_seconds: dict[str, float] = {}
    for name, search in searches.items():
        started = time.perf_counter()
        jax.block_until_ready(search(model, state, base_key))
        compile_seconds[name] = time.perf_counter() - started
    for index in range(warmups):
        key = jax.random.fold_in(base_key, index + 1)
        order = list(searches)
        if index % 2:
            order.reverse()
        for name in order:
            jax.block_until_ready(searches[name](model, state, key))

    timing_samples = {
        name: np.empty((repetitions,), dtype=np.float64)
        for name in searches
    }
    method_names = list(searches)
    for repetition in range(repetitions):
        key = jax.random.fold_in(base_key, warmups + repetition + 1)
        offset = repetition % len(method_names)
        order = method_names[offset:] + method_names[:offset]
        if repetition % 2:
            order.reverse()
        for name in order:
            started = time.perf_counter()
            jax.block_until_ready(searches[name](model, state, key))
            timing_samples[name][repetition] = (
                time.perf_counter() - started
            )

    diagnostic_key = jax.random.fold_in(
        base_key,
        warmups + repetitions + 1,
    )
    work_diagnostics: dict[str, dict[str, float]] = {}
    for name, search in searches.items():
        output = jax.block_until_ready(
            search(model, state, diagnostic_key)
        )
        simulation_active = np.asarray(output[-2], dtype=np.float64)
        executed_calls = np.asarray(output[-1], dtype=np.float64)
        work_diagnostics[name] = {
            "simulation_active_count_mean": float(
                np.mean(simulation_active)
            ),
            "executed_simulation_call_count_mean": float(
                np.mean(executed_calls)
            ),
        }

    def summarize_timing(name: str) -> dict[str, Any]:
        samples = timing_samples[name]
        median = float(np.median(samples))
        return {
            "root_batch_size": root_count,
            "unique_corpus_roots": unique_timing_roots,
            "roots_tiled_to_batch": bool(
                root_count > unique_timing_roots
            ),
            "checkpoint_step": step,
            "public_policy_samples": int(search_config.policy_samples),
            "public_policy_sample_chunk_size": max(1, chunk_size),
            "compile_and_first_seconds": compile_seconds[name],
            "warm_seconds_mean": float(np.mean(samples)),
            "warm_seconds_median": median,
            "warm_seconds_p95": float(np.quantile(samples, 0.95)),
            "warm_roots_per_second_at_median": root_count / median,
            "warm_repetitions": repetitions,
            "warmup_repetitions": warmups,
            "timing_protocol": (
                "same folded key per repetition; methods interleaved with "
                "rotating/reversed order"
            ),
            **work_diagnostics[name],
        }

    baseline = summarize_timing("m32")
    candidates: dict[str, Any] = {}
    baseline_samples = timing_samples["m32"]
    for ordinal, spec in enumerate(candidate_specs):
        timing = summarize_timing(spec["name"])
        timing["relative_warm_time_to_m32"] = (
            float(timing["warm_seconds_median"])
            / float(baseline["warm_seconds_median"])
        )
        paired_ratio = timing_samples[spec["name"]] / baseline_samples
        log_ratio = np.log(paired_ratio)
        bootstrap_rng = np.random.default_rng(
            seed + 1_000_003 * (ordinal + 1)
        )
        bootstrap_indices = bootstrap_rng.integers(
            0,
            repetitions,
            size=(10_000, repetitions),
        )
        bootstrap_geomean = np.exp(
            np.mean(log_ratio[bootstrap_indices], axis=-1)
        )
        timing["paired_time_ratio_median"] = float(
            np.median(paired_ratio)
        )
        timing["paired_time_ratio_geometric_mean"] = float(
            np.exp(np.mean(log_ratio))
        )
        timing["paired_time_ratio_p95"] = float(
            np.quantile(paired_ratio, 0.95)
        )
        timing["paired_geometric_mean_ratio_ci95"] = [
            float(np.quantile(bootstrap_geomean, 0.025)),
            float(np.quantile(bootstrap_geomean, 0.975)),
        ]
        timing["paired_bootstrap_repetitions"] = 10_000
        candidates[spec["name"]] = timing
    return {
        "status": "measured",
        "hook": "update_posterior_with_estimator",
        "baseline_m32": baseline,
        "candidates": candidates,
    }


def _benchmark(args: argparse.Namespace) -> None:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    import functools

    import jax
    import jax.numpy as jnp

    from scacchi.dirichlet_mctx.action_selection import posterior_best_policy
    from scacchi.dirichlet_mctx.estimator_diagnostics import (
        binary_posterior_best_policy_prefix_quadrature,
        binary_posterior_best_policy_quadrature,
        binary_posterior_best_policy_rao_blackwell,
    )

    for name in (
        "batch_size",
        "stochastic_repetitions",
        "bootstrap_repetitions",
        "timing_batch_size",
        "timing_repetitions",
        "search_timing_repetitions",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("timing_warmups", "search_timing_warmups"):
        if int(getattr(args, name)) < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be nonnegative"
            )
    if args.complete_search_roots < 0 or args.max_contexts < 0:
        raise ValueError(
            "--complete-search-roots and --max-contexts must be nonnegative"
        )
    if not 0.5 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence must be in (0.5, 1)")
    if args.maximum_mse_ratio <= 0 or args.maximum_runtime_ratio <= 0:
        raise ValueError("acceptance ratios must be positive")
    if args.simplex_tolerance < 0:
        raise ValueError("--simplex-tolerance must be nonnegative")

    manifest, roots, contexts = _load_corpus(args.corpus.resolve())
    settings = manifest["config"]["selfplay"]["search"][
        "dirichlet_thompson"
    ]
    kappa = float(settings["kappa"])
    m32_samples = settings.get("posterior_policy_samples")
    if m32_samples is None:
        m32_samples = settings["policy_samples"]
    m32_samples = int(m32_samples)
    sample_chunk_size = settings.get("policy_sample_chunk_size")
    if sample_chunk_size is None:
        sample_chunk_size = 4
    sample_chunk_size = int(sample_chunk_size)
    grids = _parse_optional_quadrature_grids(args.quadrature_grids)
    prefix_half_widths = _parse_optional_positive_ints(
        args.prefix_half_widths
    )
    rb_samples = _parse_optional_positive_ints(
        args.rao_blackwell_samples
    )
    winner_samples = _parse_optional_positive_ints(
        args.winner_mc_samples
    )
    live_indices, imminent_categorical = _benchmark_live_indices(contexts)
    live_indices = _stratified_context_limit(
        live_indices,
        contexts,
        maximum=args.max_contexts,
        seed=args.seed,
    )
    if len(live_indices) == 0:
        raise RuntimeError("corpus has no live unresolved repair contexts")
    live_unresolved = (
        ~contexts["invalid_actions"][live_indices]
        & (contexts["categorical_outcome"][live_indices] == -1)
    )
    live_alpha_components = contexts["effective_alpha"][live_indices][
        live_unresolved
    ].reshape(-1)
    if live_alpha_components.size == 0:
        raise RuntimeError(
            "live benchmark population has no unresolved alpha components"
        )

    reference_started = time.perf_counter()
    reference = _benchmark_reference(
        contexts=contexts,
        indices=live_indices,
        kappa=kappa,
        m32_samples=m32_samples,
        batch_size=args.batch_size,
        half_width=args.reference_half_width,
        step=args.reference_step,
    )
    reference_seconds = time.perf_counter() - reference_started
    if not np.all(reference["finite"]):
        raise FloatingPointError(
            "exact reference produced non-finite rows"
        )
    reference_max_normalization_error = float(
        np.max(reference["normalization_error"])
    )
    if (
        reference_max_normalization_error
        > args.max_reference_normalization_error
    ):
        raise FloatingPointError(
            "reference normalization error exceeds gate: "
            f"{reference_max_normalization_error} > "
            f"{args.max_reference_normalization_error}"
        )

    candidate_dtype = (
        jnp.float32 if args.candidate_dtype == "float32" else jnp.float64
    )

    def make_quadrature_kernel(half_width: int, step: float):
        return jax.jit(
            functools.partial(
                lambda _key, alpha, invalid, categorical, *, h, s: (
                    binary_posterior_best_policy_quadrature(
                        alpha,
                        invalid,
                        categorical,
                        half_width=h,
                        step=s,
                    )
                ),
                h=half_width,
                s=step,
            )
        )

    def make_prefix_kernel(half_width: int):
        @jax.jit
        def kernel(_key, alpha, invalid, categorical):
            return binary_posterior_best_policy_prefix_quadrature(
                alpha,
                invalid,
                categorical,
                half_width=half_width,
                adaptive_range=True,
                tail_scale=8.0,
                min_half_range=6.0,
                max_half_range=11.0,
            )

        return kernel

    def make_rb_kernel(num_samples: int):
        return jax.jit(
            functools.partial(
                binary_posterior_best_policy_rao_blackwell,
                num_samples=num_samples,
                sample_chunk_size=min(sample_chunk_size, num_samples),
            )
        )

    def make_winner_kernel(num_samples: int):
        @jax.jit
        def kernel(key, alpha, invalid, categorical):
            policy = posterior_best_policy(
                key,
                alpha,
                invalid,
                num_samples,
                chunk_size=min(sample_chunk_size, num_samples),
                categorical_outcome=categorical,
            )
            policy_mass = jnp.sum(policy, axis=-1)
            normalization_error = jnp.abs(
                policy_mass
                - jnp.any(~invalid, axis=-1).astype(policy.dtype)
            )
            finite = (
                jnp.all(jnp.isfinite(policy), axis=-1)
                & jnp.isfinite(normalization_error)
            )
            return _PolicyEstimate(
                policy=policy,
                normalization_error=normalization_error,
                finite=finite,
            )

        return kernel

    candidate_specs: list[dict[str, Any]] = []
    for half_width in prefix_half_widths:
        candidate_specs.append(
            {
                "name": (
                    f"prefix_cdf_mass_conserving_adaptive_h{half_width}_"
                    f"q{2 * half_width + 1}"
                ),
                "kind": (
                    "mass_conserving_adaptive_prefix_cdf_quadrature"
                ),
                "parameters": {
                    "half_width": half_width,
                    "grid_points": 2 * half_width + 1,
                    "tail_scale": 8.0,
                    "min_half_range": 6.0,
                    "max_half_range": 11.0,
                    "mass_conserving": True,
                    "range_rule": (
                        "clip(asinh(8 / min_legal_unresolved_alpha), "
                        "6, 11)"
                    ),
                },
                "stochastic_repetitions": 1,
                "kernel": make_prefix_kernel(half_width),
            }
        )
    for half_width, step in grids:
        candidate_specs.append(
            {
                "name": (
                    f"quadrature_h{half_width}_"
                    f"step{step:g}_q{2 * half_width + 1}"
                ),
                "kind": "fixed_sinh_logit_quadrature",
                "parameters": {
                    "half_width": half_width,
                    "step": step,
                    "grid_points": 2 * half_width + 1,
                },
                "stochastic_repetitions": 1,
                "kernel": make_quadrature_kernel(half_width, step),
            }
        )
    for num_samples in rb_samples:
        candidate_specs.append(
            {
                "name": f"rao_blackwell_m{num_samples}",
                "kind": "stochastic_rao_blackwell",
                "parameters": {
                    "num_samples": num_samples,
                    "sample_chunk_size": min(
                        sample_chunk_size,
                        num_samples,
                    ),
                },
                "stochastic_repetitions": (
                    args.stochastic_repetitions
                ),
                "kernel": make_rb_kernel(num_samples),
            }
        )
    for num_samples in winner_samples:
        candidate_specs.append(
            {
                "name": f"winner_count_m{num_samples}",
                "kind": "ordinary_winner_count_monte_carlo",
                "parameters": {
                    "num_samples": num_samples,
                    "sample_chunk_size": min(
                        sample_chunk_size,
                        num_samples,
                    ),
                },
                "stochastic_repetitions": (
                    args.stochastic_repetitions
                ),
                "kernel": make_winner_kernel(num_samples),
            }
        )

    @jax.jit
    def m32_kernel(key, alpha, invalid, categorical):
        return posterior_best_policy(
            key,
            alpha,
            invalid,
            m32_samples,
            chunk_size=min(sample_chunk_size, m32_samples),
            categorical_outcome=categorical,
        )

    timing_count = min(args.timing_batch_size, len(live_indices))
    timing_indices = live_indices[:timing_count]
    timing_alpha = jnp.asarray(
        contexts["effective_alpha"][timing_indices],
        dtype=candidate_dtype,
    )
    timing_invalid = jnp.asarray(
        contexts["invalid_actions"][timing_indices]
    )
    timing_categorical = jnp.asarray(
        contexts["categorical_outcome"][timing_indices]
    )
    baseline_timing = _time_estimator_kernel(
        kernel=m32_kernel,
        alpha=timing_alpha,
        invalid_actions=timing_invalid,
        categorical_outcome=timing_categorical,
        repetitions=args.timing_repetitions,
        warmups=args.timing_warmups,
        seed=args.seed + 17,
    )

    candidate_outputs: dict[str, Any] = {}
    for ordinal, spec in enumerate(candidate_specs):
        evaluation = _evaluate_estimator_candidate(
            kernel=spec["kernel"],
            stochastic_repetitions=spec["stochastic_repetitions"],
            contexts=contexts,
            indices=live_indices,
            reference=reference,
            batch_size=args.batch_size,
            dtype_name=args.candidate_dtype,
            seed=args.seed + 100_003 * (ordinal + 1),
            simplex_tolerance=args.simplex_tolerance,
        )
        candidate_timing = _time_estimator_kernel(
            kernel=spec["kernel"],
            alpha=timing_alpha,
            invalid_actions=timing_invalid,
            categorical_outcome=timing_categorical,
            repetitions=args.timing_repetitions,
            warmups=args.timing_warmups,
            seed=args.seed + 200_003 * (ordinal + 1),
        )
        candidate_timing["relative_warm_time_to_m32"] = (
            float(candidate_timing["warm_seconds_median"])
            / float(baseline_timing["warm_seconds_median"])
        )
        checkpoint = contexts["checkpoint_step"][live_indices]
        stage = contexts["stage_id"][live_indices]
        group_masks: dict[str, np.ndarray] = {
            "overall": np.ones((len(live_indices),), dtype=bool)
        }
        for value in sorted(int(x) for x in np.unique(checkpoint)):
            group_masks[f"checkpoint_{value}"] = checkpoint == value
        for stage_id, (stage_name, _, _) in enumerate(STAGES):
            group_masks[f"stage_{stage_name}"] = stage == stage_id
        for value in sorted(int(x) for x in np.unique(checkpoint)):
            for stage_id, (stage_name, _, _) in enumerate(STAGES):
                group_masks[
                    f"checkpoint_{value}_stage_{stage_name}"
                ] = (checkpoint == value) & (stage == stage_id)

        groups: dict[str, Any] = {}
        for group_ordinal, (name, mask) in enumerate(group_masks.items()):
            if not np.any(mask):
                continue
            groups[name] = _candidate_group_summary(
                mask=mask,
                evaluation=evaluation,
                reference=reference,
                contexts=contexts,
                indices=live_indices,
                repetitions=args.bootstrap_repetitions,
                confidence=args.bootstrap_confidence,
                seed=(
                    args.seed
                    + 1_000_003 * (ordinal + 1)
                    + 10_007 * group_ordinal
                ),
            )

        overall = groups["overall"]
        accuracy_pass = (
            overall["bootstrap"]["raw_mse_ratio_ucb"]
            <= args.maximum_mse_ratio
            and overall["bootstrap"]["semantic_mse_ratio_ucb"]
            <= args.maximum_mse_ratio
        )
        failure_diagnostics = evaluation["failure_diagnostics"]
        stability_pass = (
            failure_diagnostics["rows_with_any_failure"] == 0
            and overall["raw_normalization_error_max"]
            <= args.max_candidate_normalization_error
        )
        kernel_runtime_pass = (
            candidate_timing["relative_warm_time_to_m32"]
            <= args.maximum_runtime_ratio
        )
        candidate_outputs[spec["name"]] = {
            "kind": spec["kind"],
            "parameters": spec["parameters"],
            "dtype": args.candidate_dtype,
            "stochastic_repetitions": spec[
                "stochastic_repetitions"
            ],
            "failure_diagnostics": failure_diagnostics,
            "groups": groups,
            "estimator_kernel_timing": candidate_timing,
            "gates": {
                "accuracy_pass": bool(accuracy_pass),
                "stability_pass": bool(stability_pass),
                "kernel_runtime_pass_provisional": bool(
                    kernel_runtime_pass
                ),
                "offline_pass_before_complete_search_timing": bool(
                    accuracy_pass
                    and stability_pass
                    and kernel_runtime_pass
                ),
            },
        }

    if args.complete_search_roots > 0:
        complete_search = _time_complete_search_estimators(
            candidate_specs=candidate_specs,
            manifest=manifest,
            roots=roots,
            step=args.complete_search_step,
            root_count=args.complete_search_roots,
            repetitions=args.search_timing_repetitions,
            warmups=args.search_timing_warmups,
            seed=args.seed + 9_000_001,
            dtype_name=args.candidate_dtype,
        )
        for name, timing in complete_search["candidates"].items():
            ratio = timing["paired_geometric_mean_ratio_ci95"][1]
            candidate_outputs[name]["complete_search_timing"] = timing
            candidate_outputs[name]["gates"][
                "complete_search_runtime_pass"
            ] = bool(ratio <= args.maximum_runtime_ratio)
            candidate_outputs[name]["gates"]["accepted"] = bool(
                candidate_outputs[name]["gates"]["accuracy_pass"]
                and candidate_outputs[name]["gates"]["stability_pass"]
                and ratio <= args.maximum_runtime_ratio
            )
    else:
        complete_search = {
            "status": "not_measured",
            "integration_point": (
                "rerun benchmark with --complete-search-roots N; "
                "the script injects each candidate through "
                "update_posterior_with_estimator without changing traversal"
            ),
        }
        for output in candidate_outputs.values():
            output["complete_search_timing"] = {
                "status": "pending",
            }
            output["gates"]["accepted"] = None

    output = {
        "corpus": str(args.corpus.resolve()),
        "population": {
            "all_active_contexts": int(len(contexts["root_id"])),
            "available_live_unresolved_contexts": int(
                len(_benchmark_live_indices(contexts)[0])
            ),
            "benchmarked_live_unresolved_contexts": int(
                len(live_indices)
            ),
            "subsampled": bool(
                args.max_contexts > 0
                and len(live_indices)
                < len(_benchmark_live_indices(contexts)[0])
            ),
            "imminent_categorical_contexts": int(
                np.sum(contexts["active"] & imminent_categorical)
            ),
            "whole_game_clusters": int(
                len(
                    np.unique(
                        contexts["game_cluster_id"][live_indices]
                    )
                )
            ),
            "legal_unresolved_alpha_component_min": float(
                np.min(live_alpha_components)
            ),
            "legal_unresolved_alpha_component_p01": float(
                np.quantile(live_alpha_components, 0.01)
            ),
            "legal_unresolved_alpha_component_p50": float(
                np.quantile(live_alpha_components, 0.50)
            ),
        },
        "reference": {
            "kind": "fixed_sinh_logit_exact_beta",
            "dtype": "float64",
            "half_width": args.reference_half_width,
            "step": args.reference_step,
            "grid_points": 2 * args.reference_half_width + 1,
            "finite_fraction": float(np.mean(reference["finite"])),
            "normalization_error_p95": float(
                np.quantile(reference["normalization_error"], 0.95)
            ),
            "normalization_error_max": (
                reference_max_normalization_error
            ),
            "seconds_including_compile": reference_seconds,
        },
        "analytic_m32_baseline": {
            "population_samples": m32_samples,
            "raw_cache_mse_weighted_mean": _weighted_average(
                reference["m32_raw_mse"],
                contexts["root_weight"][live_indices].astype(np.float64),
            ),
            "semantic_delta_mse_weighted_mean": _weighted_average(
                reference["m32_semantic_mse"],
                contexts["root_weight"][live_indices].astype(np.float64),
            ),
            "estimator_kernel_timing": baseline_timing,
        },
        "candidates": candidate_outputs,
        "complete_search_timing": complete_search,
        "acceptance_thresholds": {
            "maximum_raw_and_semantic_mse_ratio_ucb": (
                args.maximum_mse_ratio
            ),
            "maximum_warm_runtime_ratio": args.maximum_runtime_ratio,
            "complete_search_runtime_statistic": (
                "upper 95% paired-bootstrap confidence bound on the "
                "geometric-mean candidate/M32 time ratio"
            ),
            "maximum_candidate_normalization_error": (
                args.max_candidate_normalization_error
            ),
            "simplex_tolerance": args.simplex_tolerance,
            "bootstrap_confidence": args.bootstrap_confidence,
            "note": (
                "kernel timing is provisional; acceptance remains pending "
                "until complete-search timing is measured"
            ),
        },
        "backend": jax.default_backend(),
    }
    output_path = (
        args.output.resolve()
        if args.output is not None
        else args.corpus.resolve() / "benchmark.json"
    )
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect",
        help="collect actual pre-estimator posterior-repair contexts",
    )
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--steps", default="0,50,100")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--seed", type=int, default=20260724)
    collect.add_argument("--rollout-batch-size", type=int, default=256)
    collect.add_argument("--max-steps", type=int, default=36)
    collect.add_argument("--roots-per-stage", type=int, default=64)
    collect.add_argument("--trace-batch-size", type=int, default=64)
    collect.set_defaults(handler=_collect)

    verify = subparsers.add_parser(
        "verify",
        help="verify stored hashes, shapes, and native invariants",
    )
    verify.add_argument("--corpus", type=Path, required=True)
    verify.set_defaults(handler=_verify)

    analyze = subparsers.add_parser(
        "analyze",
        help="measure exact-Beta M32 cache noise and materiality",
    )
    analyze.add_argument("--corpus", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--batch-size", type=int, default=2048)
    analyze.add_argument("--quadrature-half-width", type=int, default=160)
    analyze.add_argument("--quadrature-step", type=float, default=0.1)
    analyze.add_argument("--max-normalization-error", type=float, default=2e-4)
    analyze.add_argument("--bootstrap-repetitions", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=20260725)
    analyze.add_argument("--minimum-rho", type=float, default=0.10)
    analyze.add_argument("--tail-rms-ratio", type=float, default=0.25)
    analyze.add_argument(
        "--minimum-tail-fraction",
        type=float,
        default=0.10,
    )
    analyze.set_defaults(handler=_analyze)

    benchmark = subparsers.add_parser(
        "benchmark",
        help=(
            "compare prefix-CDF, fixed quadrature, and Rao-Blackwell "
            "estimators against the exact reference and analytic M32 baseline"
        ),
    )
    benchmark.add_argument("--corpus", type=Path, required=True)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--batch-size", type=int, default=2048)
    benchmark.add_argument(
        "--max-contexts",
        type=int,
        default=0,
        help="stratified context cap for smoke runs; zero uses the corpus",
    )
    benchmark.add_argument(
        "--reference-half-width",
        type=int,
        default=160,
    )
    benchmark.add_argument("--reference-step", type=float, default=0.1)
    benchmark.add_argument(
        "--max-reference-normalization-error",
        type=float,
        default=2e-4,
    )
    benchmark.add_argument(
        "--prefix-half-widths",
        default="20",
        help=(
            "comma-separated half-widths for adaptive prefix-CDF grids; "
            "20 is Q41, and 'none' disables them"
        ),
    )
    benchmark.add_argument(
        "--quadrature-grids",
        default="20:0.3,40:0.2,60:0.15,80:0.1",
        help=(
            "comma-separated HALF_WIDTH:STEP fixed exact-CDF grids; "
            "'none' disables them"
        ),
    )
    benchmark.add_argument(
        "--rao-blackwell-samples",
        default="8,16,32,64",
        help="comma-separated stochastic sample budgets",
    )
    benchmark.add_argument(
        "--winner-mc-samples",
        default="128",
        help=(
            "comma-separated ordinary winner-count budgets used as a "
            "matched-cost control"
        ),
    )
    benchmark.add_argument(
        "--stochastic-repetitions",
        type=int,
        default=16,
    )
    benchmark.add_argument(
        "--candidate-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    benchmark.add_argument(
        "--simplex-tolerance",
        type=float,
        default=1e-5,
    )
    benchmark.add_argument(
        "--max-candidate-normalization-error",
        type=float,
        default=2e-3,
    )
    benchmark.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=2000,
    )
    benchmark.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
    )
    benchmark.add_argument("--seed", type=int, default=20260726)
    benchmark.add_argument(
        "--timing-batch-size",
        type=int,
        default=8192,
    )
    benchmark.add_argument(
        "--timing-repetitions",
        type=int,
        default=20,
    )
    benchmark.add_argument("--timing-warmups", type=int, default=2)
    benchmark.add_argument(
        "--maximum-mse-ratio",
        type=float,
        default=0.25,
    )
    benchmark.add_argument(
        "--maximum-runtime-ratio",
        type=float,
        default=1.25,
    )
    benchmark.add_argument(
        "--complete-search-roots",
        type=int,
        default=0,
        help=(
            "opt-in root batch for end-to-end native search timing; zero "
            "leaves the required search-runtime gate pending"
        ),
    )
    benchmark.add_argument(
        "--complete-search-step",
        type=int,
        default=100,
    )
    benchmark.add_argument(
        "--search-timing-repetitions",
        type=int,
        default=5,
    )
    benchmark.add_argument(
        "--search-timing-warmups",
        type=int,
        default=1,
    )
    benchmark.set_defaults(handler=_benchmark)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
