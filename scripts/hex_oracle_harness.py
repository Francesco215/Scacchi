#!/usr/bin/env python3
"""Freeze and score an exact late-game Hex corpus.

Examples
--------
Create a balanced, exact Hex6 corpus with at most six empty cells::

    uv run python scripts/hex_oracle_harness.py sample \
      --output experiments/hex6_oracle.json --count 128

Score policies and optional value-head probabilities stored in JSON::

    uv run python scripts/hex_oracle_harness.py score \
      --corpus experiments/hex6_oracle.json \
      --predictions /tmp/hex6_predictions.json \
      --output /tmp/hex6_oracle_scores.json

Score a Scacchi checkpoint directly, using the checkpoint's self-play
Dirichlet-Thompson configuration.  Exact checkpoint scoring is conservatively
limited to positions with at most 15 empty cells::

    JAX_PLATFORMS=cuda,cpu uv run python scripts/hex_oracle_harness.py checkpoint \
      --corpus experiments/hex6_oracle.json \
      --checkpoint checkpoints/hex6_info_baseline_s0 \
      --checkpoint-step 75 \
      --output /tmp/hex6_checkpoint_oracle_scores.json \
      --predictions-output /tmp/hex6_checkpoint_predictions.json

The prediction file has this shape::

    {
      "predictions": [
        {
          "position_id": "...",
          "prior_policy": [0.0, "... one value per model action ..."],
          "search_policy": [0.0, "... mean of repeated readouts ..."],
          "search_policy_readouts": [
            [0.0, "... readout 1 from the same fixed tree ..."],
            [0.0, "... readout 2 from the same fixed tree ..."]
          ],
          "prior_win_probability": 0.4,
          "search_win_probability": 0.7,
          "prior_value_concentration": 5.0,
          "search_value_concentration": 7.0
        }
      ]
    }

The win-probability fields and repeated readouts are optional.  Policies may
include PGX's final swap action; occupied cells and the swap action are masked
before scoring.  Repeated checkpoint readouts hold the searched tree fixed and
resample only its final finite Thompson population, so their pairwise
dispersion estimates readout noise rather than search displacement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import functools
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, cast

from scacchi.hex_oracle import (
    assess_policy_readout_noise,
    canonical_action_sequence,
    compare_binary_outcome_probabilities,
    compare_policies_against_oracle,
    load_frozen_hex_corpus,
    position_from_pgx_state,
    sample_late_game_hex_corpus,
    write_frozen_hex_corpus,
)


MAX_CHECKPOINT_ORACLE_EMPTY_CELLS = 15


def _sample(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    corpus = sample_late_game_hex_corpus(
        count=args.count,
        size=args.size,
        min_empty=args.min_empties,
        max_empty=args.max_empties,
        seed=args.seed,
        balanced_outcomes=not args.unbalanced,
        max_attempts=args.max_attempts,
    )
    elapsed = time.perf_counter() - started
    write_frozen_hex_corpus(args.output, corpus)
    cache_states = sum(
        1 + len(position.action_values)
        for position in corpus.positions
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "positions": len(corpus.positions),
                "attempts": corpus.attempts,
                "seconds": elapsed,
                "positions_per_second": len(corpus.positions) / elapsed,
                "root_and_action_labels": cache_states,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(cast(float, value)))
    return statistics.fmean(values)


def _score_prediction_payload(
    corpus: Any,
    prediction_payload: dict[str, Any],
) -> dict[str, Any]:
    predictions = {
        str(item["position_id"]): item
        for item in prediction_payload["predictions"]
    }
    rows: list[dict[str, Any]] = []
    for frozen in corpus.positions:
        try:
            prediction = predictions[frozen.position_id]
        except KeyError as error:
            raise ValueError(
                f"missing prediction for {frozen.position_id}"
            ) from error
        comparison = compare_policies_against_oracle(
            prior_policy=prediction["prior_policy"],
            search_policy=prediction["search_policy"],
            result=frozen.result,
        )
        row: dict[str, Any] = {
            "position_id": frozen.position_id,
            "empty_count": frozen.cells.count(0),
            "oracle_outcome": frozen.oracle_outcome,
            "policy": asdict(comparison),
        }
        has_prior_value = "prior_win_probability" in prediction
        has_search_value = "search_win_probability" in prediction
        if has_prior_value != has_search_value:
            raise ValueError(
                "prior_win_probability and search_win_probability must "
                f"appear together for {frozen.position_id}"
            )
        if has_prior_value:
            row["outcome"] = asdict(
                compare_binary_outcome_probabilities(
                    oracle_outcome=frozen.oracle_outcome,
                    prior_win_probability=float(
                        prediction["prior_win_probability"]
                    ),
                    search_win_probability=float(
                        prediction["search_win_probability"]
                    ),
                )
            )
        has_prior_concentration = "prior_value_concentration" in prediction
        has_search_concentration = "search_value_concentration" in prediction
        if has_prior_concentration != has_search_concentration:
            raise ValueError(
                "prior_value_concentration and "
                "search_value_concentration must appear together for "
                f"{frozen.position_id}"
            )
        if has_prior_concentration:
            categorical_outcome = int(
                prediction.get("search_value_categorical_outcome", -1)
            )
            row["value_diagnostics"] = {
                "prior_concentration": float(
                    prediction["prior_value_concentration"]
                ),
                "search_concentration": float(
                    prediction["search_value_concentration"]
                ),
                "search_is_categorical": categorical_outcome >= 0,
                "search_categorical_outcome": categorical_outcome,
            }
        readouts = prediction.get("search_policy_readouts")
        if readouts is not None:
            row["policy_readout_noise"] = asdict(
                assess_policy_readout_noise(
                    prior_policy=prediction["prior_policy"],
                    search_policy_readouts=readouts,
                    result=frozen.result,
                )
            )
        rows.append(row)

    summary: dict[str, Any] = {
        "positions": len(rows),
        "mean_prior_regret": _mean(rows, ("policy", "prior", "regret")),
        "mean_search_regret": _mean(rows, ("policy", "search", "regret")),
        "mean_prior_optimal_action_mass": _mean(
            rows,
            ("policy", "prior", "optimal_action_mass"),
        ),
        "mean_search_optimal_action_mass": _mean(
            rows,
            ("policy", "search", "optimal_action_mass"),
        ),
        "prior_top_action_optimal_fraction": _mean(
            rows,
            ("policy", "prior", "top_action_is_optimal"),
        ),
        "search_top_action_optimal_fraction": _mean(
            rows,
            ("policy", "search", "top_action_is_optimal"),
        ),
        "mean_regret_reduction": _mean(
            rows,
            ("policy", "regret_reduction"),
        ),
        "mean_prior_policy_log_loss": _mean(
            rows,
            ("policy", "proper_scores", "prior_log_loss"),
        ),
        "mean_search_policy_log_loss": _mean(
            rows,
            ("policy", "proper_scores", "search_log_loss"),
        ),
        "mean_policy_log_score_gain": _mean(
            rows,
            ("policy", "proper_scores", "log_score_gain"),
        ),
        "mean_prior_policy_brier_loss": _mean(
            rows,
            ("policy", "proper_scores", "prior_brier_loss"),
        ),
        "mean_search_policy_brier_loss": _mean(
            rows,
            ("policy", "proper_scores", "search_brier_loss"),
        ),
        "mean_policy_brier_score_gain": _mean(
            rows,
            ("policy", "proper_scores", "brier_score_gain"),
        ),
    }
    if rows and "outcome" in rows[0]:
        summary.update(
            {
                "mean_prior_outcome_log_loss": _mean(
                    rows,
                    ("outcome", "prior_log_loss"),
                ),
                "mean_search_outcome_log_loss": _mean(
                    rows,
                    ("outcome", "search_log_loss"),
                ),
                "mean_outcome_log_score_gain": _mean(
                    rows,
                    ("outcome", "log_score_gain"),
                ),
                "mean_prior_outcome_brier_loss": _mean(
                    rows,
                    ("outcome", "prior_brier_loss"),
                ),
                "mean_search_outcome_brier_loss": _mean(
                    rows,
                    ("outcome", "search_brier_loss"),
                ),
                "mean_outcome_brier_score_gain": _mean(
                    rows,
                    ("outcome", "brier_score_gain"),
                ),
            }
        )
    if rows and "value_diagnostics" in rows[0]:
        summary.update(
            {
                "mean_prior_value_concentration": _mean(
                    rows,
                    ("value_diagnostics", "prior_concentration"),
                ),
                "mean_search_value_concentration": _mean(
                    rows,
                    ("value_diagnostics", "search_concentration"),
                ),
                "search_value_categorical_fraction": _mean(
                    rows,
                    ("value_diagnostics", "search_is_categorical"),
                ),
            }
        )
    if rows and "policy_readout_noise" in rows[0]:
        summary.update(
            {
                "mean_prior_to_mean_search_js_nats": _mean(
                    rows,
                    (
                        "policy_readout_noise",
                        "prior_to_mean_search_js_nats",
                    ),
                ),
                "mean_pairwise_search_readout_js_nats": _mean(
                    rows,
                    (
                        "policy_readout_noise",
                        "mean_pairwise_search_readout_js_nats",
                    ),
                ),
                "mean_readout_noise_squared_l2": _mean(
                    rows,
                    (
                        "policy_readout_noise",
                        "readout_noise_squared_l2",
                    ),
                ),
                "mean_noise_corrected_displacement_squared_l2": _mean(
                    rows,
                    (
                        "policy_readout_noise",
                        "noise_corrected_displacement_squared_l2",
                    ),
                ),
            }
        )
    return {
        "prediction_metadata": prediction_payload.get("metadata"),
        "summary": summary,
        "positions": rows,
    }


def _emit_scores(
    output: dict[str, Any],
    output_path: Path | None,
) -> None:
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if output_path is None:
        print(encoded, end="")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded, encoding="utf-8")
        print(json.dumps(output["summary"], indent=2, sort_keys=True))


def _score(args: argparse.Namespace) -> None:
    corpus = load_frozen_hex_corpus(args.corpus, verify=True)
    prediction_payload = json.loads(
        Path(args.predictions).read_text(encoding="utf-8")
    )
    _emit_scores(
        _score_prediction_payload(corpus, prediction_payload),
        args.output,
    )


def _load_checkpoint_config_and_step(
    checkpoint_path: Path,
    requested_step: int | None,
):
    """Read config and compute counters from one exact Orbax step."""

    from scacchi import checkpoint as checkpoint_io

    resolved = checkpoint_path.resolve()
    checkpoint_io._suppress_orbax_logs()
    options = checkpoint_io._checkpoint_manager_options(read_only=True)
    with checkpoint_io.ocp.CheckpointManager(
        str(resolved),
        options=options,
    ) as manager:
        available_steps = tuple(int(step) for step in manager.all_steps())
        if not available_steps:
            raise FileNotFoundError(f"No checkpoint found in {resolved}")
        if requested_step is None:
            latest_step = manager.latest_step()
            if latest_step is None:
                raise FileNotFoundError(f"No checkpoint found in {resolved}")
            step = int(latest_step)
        else:
            step = int(requested_step)
            if step not in set(available_steps):
                raise FileNotFoundError(
                    f"checkpoint step {step} not found in {resolved}; "
                    f"available steps: {sorted(available_steps)}"
                )
        restored = manager.restore(
            step,
            args=checkpoint_io.ocp.args.Composite(
                meta=checkpoint_io.ocp.args.JsonRestore()
            ),
        )
    meta = restored["meta"]
    stored_step = int(meta.get("step", step))
    if stored_step != step:
        raise ValueError(
            f"checkpoint directory step {step} stores metadata step "
            f"{stored_step}"
        )
    config = checkpoint_io._load_checkpoint_config(meta["config"])
    progress = {
        "checkpoint_hours": (
            None if meta.get("hours") is None else float(meta["hours"])
        ),
        "checkpoint_frames": (
            None if meta.get("frames") is None else int(meta["frames"])
        ),
        "checkpoint_optimizer_updates": (
            None
            if meta.get("optimizer_updates") is None
            else int(meta["optimizer_updates"])
        ),
        # Checkpoints are saved after training iteration ``step`` completes.
        "checkpoint_completed_iterations": step + 1,
    }
    return config, int(step), progress


def _load_checkpoint_model_at_step(
    checkpoint_path: Path,
    *,
    step: int,
    config: Any,
    env: Any,
):
    """Build and restore weights from exactly the selected retained step."""

    from flax import nnx

    from scacchi import checkpoint as checkpoint_io
    from scacchi.network import build_model

    model = build_model(
        config,
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        rngs=nnx.Rngs(0),
    )
    resolved = checkpoint_path.resolve()
    checkpoint_io._suppress_orbax_logs()
    options = checkpoint_io._checkpoint_manager_options(read_only=True)
    with checkpoint_io.ocp.CheckpointManager(
        str(resolved),
        options=options,
    ) as manager:
        if step not in set(int(value) for value in manager.all_steps()):
            raise FileNotFoundError(
                f"checkpoint step {step} not found in {resolved}"
            )
        restored = manager.restore(
            step,
            args=checkpoint_io.ocp.args.Composite(
                model=checkpoint_io.ocp.args.StandardRestore(
                    nnx.state(model)
                )
            ),
        )
    nnx.update(model, restored["model"])
    return model


def _make_frozen_position_replayer(env: Any, max_actions: int):
    """Compile one padded replay program and reuse it for every corpus row."""

    import jax
    import jax.numpy as jnp

    @jax.jit
    def replay(key, padded_actions, action_count):
        state = env.init(key)

        def replay_one(state, indexed_action):
            index, action = indexed_action
            state = jax.lax.cond(
                index < action_count,
                lambda current: env.step(current, action),
                lambda current: current,
                state,
            )
            return state, None

        state, _ = jax.lax.scan(
            replay_one,
            state,
            (
                jnp.arange(max_actions, dtype=jnp.int32),
                padded_actions,
            ),
        )
        return state

    return replay


def _replay_frozen_position(
    env: Any,
    frozen: Any,
    key: Any,
    *,
    replayer: Any | None = None,
    max_actions: int | None = None,
):
    """Recreate a frozen fixed-colour board as a scalar PGX state."""

    import jax
    import jax.numpy as jnp

    actions = canonical_action_sequence(frozen.position)
    if max_actions is None:
        max_actions = frozen.position.size * frozen.position.size
    if len(actions) > max_actions:
        raise ValueError(
            f"position {frozen.position_id} needs {len(actions)} actions, "
            f"above replay capacity {max_actions}"
        )
    if replayer is None:
        replayer = _make_frozen_position_replayer(env, max_actions)
    padded_actions = jnp.zeros((max_actions,), dtype=jnp.int32)
    if actions:
        padded_actions = padded_actions.at[: len(actions)].set(
            jnp.asarray(actions, dtype=jnp.int32)
        )
    state = replayer(
        key,
        padded_actions,
        jnp.asarray(len(actions), dtype=jnp.int32),
    )
    if bool(jax.device_get(state.terminated)):
        raise ValueError(
            "frozen corpus position has a terminal replay prefix: "
            f"{frozen.position_id}"
        )
    reconstructed = position_from_pgx_state(state)
    if reconstructed != frozen.position:
        raise ValueError(
            "PGX replay disagrees with the frozen board for "
            f"{frozen.position_id}"
        )
    return state


def _make_fixed_tree_inference(env: Any, search_config: Any):
    """Build a jitted checkpoint readout that searches each tree exactly once."""

    from flax import nnx
    import jax
    import jax.numpy as jnp

    from scacchi import dirichlet_mctx
    from scacchi.dirichlet_mctx import action_selection
    from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
    from scacchi.dirichlet_mctx.posterior_updates import (
        DEFAULT_POLICY_SAMPLE_CHUNK_SIZE,
    )
    from scacchi.dirichlet_q_search import (
        make_dirichlet_expand_fn,
        terminal_outcome_from_reward,
    )
    from scacchi.play_search import make_evaluator

    posterior_policy_samples = (
        int(search_config.policy_samples)
        if search_config.posterior_policy_samples is None
        else int(search_config.posterior_policy_samples)
    )
    posterior_update = functools.partial(
        dirichlet_mctx.update_posterior,
        kappa=float(search_config.kappa),
        policy_samples=max(1, posterior_policy_samples),
        policy_sample_chunk_size=(
            max(1, int(search_config.policy_sample_chunk_size))
            if search_config.policy_sample_chunk_size is not None
            else DEFAULT_POLICY_SAMPLE_CHUNK_SIZE
        ),
    )
    public_policy_samples = max(1, int(search_config.policy_samples))

    @nnx.jit
    def infer(model, env_state, tree_key, readout_keys):
        evaluator = make_evaluator(model)
        prediction = evaluator(env_state.observation)
        if prediction.alpha_v is None or prediction.alpha_q is None:
            raise ValueError(
                "checkpoint must expose Dirichlet V and Q heads"
            )
        alpha_v = prediction.alpha_v
        alpha_q = prediction.alpha_q
        invalid_actions = ~env_state.legal_action_mask
        root_reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            env_state.current_player,
        ]
        terminal_outcome = jnp.where(
            env_state.terminated,
            terminal_outcome_from_reward(root_reward, alpha_v.shape[-1]),
            jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
        )
        root = dirichlet_mctx.RootFnOutput(
            prior_logits=prediction.logits,
            value=alpha_v,
            action_values=alpha_q,
            embedding=env_state,
            terminal_outcome=terminal_outcome,
            to_play=env_state.current_player,
        )
        expand_fn = make_dirichlet_expand_fn(env, evaluator)
        # The wrapper splits ``tree_key`` into an internal traversal key and an
        # unused one-sample public readout.  Returning only the tree lets all
        # reported readouts use explicit, independently folded-in keys.
        tree = dirichlet_mctx.dirichlet_thompson_policy(
            params=(),
            rng_key=tree_key,
            root=root,
            recurrent_fn=expand_fn,
            num_simulations=int(search_config.num_simulations),
            invalid_actions=invalid_actions,
            posterior_update=posterior_update,
            max_depth=search_config.max_depth,
            policy_samples=1,
            policy_sample_chunk_size=1,
        ).search_tree
        summary = tree.summary()
        root_categorical_outcome = tree.node_categorical_outcome[
            :, tree.ROOT_INDEX
        ]
        edge_categorical_outcome = tree.edge_categorical_outcome[
            :, tree.ROOT_INDEX
        ]
        edge_distance = tree.edge_payload[:, tree.ROOT_INDEX]

        def fixed_tree_readout(readout_key):
            categorical_action = action_selection.categorical_action(
                jax.random.fold_in(readout_key, 0),
                root_categorical_outcome,
                edge_categorical_outcome,
                edge_distance,
                invalid_actions,
                num_outcomes=summary.alpha.shape[-1],
            )
            categorical_policy = jax.nn.one_hot(
                categorical_action,
                summary.alpha.shape[-2],
                dtype=summary.alpha.dtype,
            )
            sampled_policy = action_selection.posterior_best_policy(
                readout_key,
                summary.alpha,
                invalid_actions,
                public_policy_samples,
                chunk_size=search_config.policy_sample_chunk_size,
                categorical_outcome=edge_categorical_outcome,
            )
            return jnp.where(
                (root_categorical_outcome != int(NO_OUTCOME))[:, None],
                categorical_policy,
                sampled_policy,
            )

        policy_readouts = jax.vmap(fixed_tree_readout)(readout_keys)
        masked_logits = jnp.where(
            invalid_actions,
            jnp.finfo(prediction.logits.dtype).min,
            prediction.logits,
        )
        prior_policy = jax.nn.softmax(masked_logits, axis=-1)
        prior_win_probability = (
            alpha_v[..., -1] / jnp.sum(alpha_v, axis=-1)
        )
        posterior_win_probability = (
            summary.value_alpha[..., -1]
            / jnp.sum(summary.value_alpha, axis=-1)
        )
        search_win_probability = jnp.where(
            summary.v_categorical_outcome != int(NO_OUTCOME),
            (
                summary.v_categorical_outcome
                == summary.value_alpha.shape[-1] - 1
            ).astype(summary.value_alpha.dtype),
            posterior_win_probability,
        )
        return (
            prior_policy,
            policy_readouts,
            prior_win_probability,
            search_win_probability,
            jnp.sum(alpha_v, axis=-1),
            jnp.sum(summary.value_alpha, axis=-1),
            summary.v_categorical_outcome,
        )

    return infer


def _predict_checkpoint(
    args: argparse.Namespace,
    corpus: Any,
) -> dict[str, Any]:
    """Generate prior/search predictions directly from a saved model."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    from scacchi.envs import make_env
    from scacchi.types import SearchKind

    if args.policy_readouts < 2:
        raise ValueError("--policy-readouts must be at least 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if corpus.max_empty > MAX_CHECKPOINT_ORACLE_EMPTY_CELLS or any(
        frozen.position.empty_count > MAX_CHECKPOINT_ORACLE_EMPTY_CELLS
        for frozen in corpus.positions
    ):
        raise ValueError(
            "checkpoint oracle evaluation is restricted to positions with "
            f"at most {MAX_CHECKPOINT_ORACLE_EMPTY_CELLS} empty cells"
        )

    (
        config,
        checkpoint_step,
        checkpoint_progress,
    ) = _load_checkpoint_config_and_step(
        args.checkpoint,
        getattr(args, "checkpoint_step", None),
    )
    if config.env.id != "hex":
        raise ValueError(
            f"checkpoint environment must be hex; got {config.env.id!r}"
        )
    if config.env.board_size != corpus.size:
        raise ValueError(
            "checkpoint/corpus board-size mismatch: "
            f"{config.env.board_size} != {corpus.size}"
        )
    if config.env.num_outcomes not in (None, 2):
        raise ValueError(
            "Hex oracle checkpoint scoring requires two outcomes; got "
            f"{config.env.num_outcomes}"
        )
    search = (
        config.selfplay.search
        if args.search_source == "selfplay"
        else config.eval.player_search
    )
    if search.kind != SearchKind.dirichlet_thompson:
        raise ValueError(
            f"{args.search_source} search must be dirichlet_thompson; "
            f"got {search.kind!r}"
        )
    search_config = search.dirichlet_thompson

    env = make_env("hex", board_size=corpus.size)
    model = _load_checkpoint_model_at_step(
        args.checkpoint,
        step=checkpoint_step,
        config=config,
        env=env,
    )
    infer = _make_fixed_tree_inference(env, search_config)
    max_replay_actions = corpus.size * corpus.size
    replay_position = _make_frozen_position_replayer(
        env,
        max_replay_actions,
    )
    base_key = jax.random.PRNGKey(args.seed)
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for start in range(0, len(corpus.positions), args.batch_size):
        frozen_batch = corpus.positions[start : start + args.batch_size]
        states = [
            _replay_frozen_position(
                env,
                frozen,
                jax.random.fold_in(base_key, start + offset),
                replayer=replay_position,
                max_actions=max_replay_actions,
            )
            for offset, frozen in enumerate(frozen_batch)
        ]
        batched_state = jax.tree.map(
            lambda *values: jnp.stack(values),
            *states,
        )
        tree_key = jax.random.fold_in(base_key, 1_000_000 + start)
        readout_base_key = jax.random.fold_in(
            base_key,
            2_000_000 + start,
        )
        readout_keys = jax.random.split(
            readout_base_key,
            args.policy_readouts,
        )
        (
            prior_policy,
            policy_readouts,
            prior_win_probability,
            search_win_probability,
            prior_value_concentration,
            search_value_concentration,
            search_categorical_outcome,
        ) = jax.device_get(
            infer(
                model,
                batched_state,
                tree_key,
                readout_keys,
            )
        )
        mean_search_policy = np.mean(policy_readouts, axis=0)
        for offset, frozen in enumerate(frozen_batch):
            predictions.append(
                {
                    "position_id": frozen.position_id,
                    "prior_policy": prior_policy[offset].tolist(),
                    "search_policy": mean_search_policy[offset].tolist(),
                    "search_policy_readouts": policy_readouts[
                        :, offset
                    ].tolist(),
                    "prior_win_probability": float(
                        prior_win_probability[offset]
                    ),
                    "search_win_probability": float(
                        search_win_probability[offset]
                    ),
                    "prior_value_concentration": float(
                        prior_value_concentration[offset]
                    ),
                    "search_value_concentration": float(
                        search_value_concentration[offset]
                    ),
                    "search_value_categorical_outcome": int(
                        search_categorical_outcome[offset]
                    ),
                }
            )
        print(
            f"checkpoint oracle: {len(predictions)}/{len(corpus.positions)}",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - started
    return {
        "format": "scacchi.hex_oracle_predictions.v1",
        "metadata": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": checkpoint_step,
            **checkpoint_progress,
            "checkpoint_alignment": (
                "checkpoint step N contains weights after training "
                "iteration N; completed_iterations=N+1"
            ),
            "corpus": str(args.corpus.resolve()),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "policy_readouts": args.policy_readouts,
            "search_source": args.search_source,
            "search_kind": str(search.kind),
            "search_parameters": asdict(search_config),
            "jax_backend": jax.default_backend(),
            "seconds": elapsed,
            "positions_per_second": len(predictions) / elapsed,
            "readout_protocol": (
                "one deterministic searched tree per batch; independent "
                "final posterior-best populations on that fixed tree"
            ),
        },
        "predictions": predictions,
    }


def _checkpoint(args: argparse.Namespace) -> None:
    corpus = load_frozen_hex_corpus(args.corpus, verify=True)
    prediction_payload = _predict_checkpoint(args, corpus)
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            json.dumps(prediction_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _emit_scores(
        _score_prediction_payload(corpus, prediction_payload),
        args.output,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser(
        "sample",
        help="sample and freeze exact late-game positions",
    )
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--count", type=int, default=128)
    sample.add_argument("--size", type=int, default=6)
    sample.add_argument("--min-empties", type=int, default=1)
    sample.add_argument("--max-empties", type=int, default=6)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument(
        "--unbalanced",
        action="store_true",
        help="do not enforce equal winning/losing root outcomes",
    )
    sample.add_argument("--max-attempts", type=int)
    sample.set_defaults(function=_sample)

    score = subparsers.add_parser(
        "score",
        help="score prior/search predictions on a frozen corpus",
    )
    score.add_argument("--corpus", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.set_defaults(function=_score)

    checkpoint = subparsers.add_parser(
        "checkpoint",
        help=(
            "score a Scacchi checkpoint directly with fixed-tree repeated "
            "Dirichlet-Thompson readouts (at most 15 empty cells)"
        ),
    )
    checkpoint.add_argument("--corpus", type=Path, required=True)
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument(
        "--checkpoint-step",
        type=int,
        help=(
            "retained Orbax step to score; defaults to the latest available "
            "step"
        ),
    )
    checkpoint.add_argument("--output", type=Path)
    checkpoint.add_argument("--predictions-output", type=Path)
    checkpoint.add_argument("--seed", type=int, default=0)
    checkpoint.add_argument("--batch-size", type=int, default=16)
    checkpoint.add_argument(
        "--policy-readouts",
        type=int,
        default=4,
        help=(
            "independent final Monte Carlo populations per fixed searched "
            "tree (minimum 2)"
        ),
    )
    checkpoint.add_argument(
        "--search-source",
        choices=("selfplay", "eval"),
        default="selfplay",
        help="use the checkpoint's self-play or evaluation player search",
    )
    checkpoint.set_defaults(function=_checkpoint)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
