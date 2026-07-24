from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from scripts import hex_balanced_eval as balanced
from scripts import hex_checkpoint_league as league
from scacchi.envs import make_env
from scacchi.play_search import PlayerOutput


def _returns() -> list[list[int]]:
    return [
        [1, 1, -1, 1],
        [-1, -1, 1, -1],
        [1, -1, 1, 1],
        [-1, 1, -1, -1],
    ]


def _minimal_pair_payload(job_spec: dict[str, object]) -> dict[str, Any]:
    strata, pairwise = league.summarize_pair_returns(
        _returns(),
        competitor_a="alpha",
        competitor_b="beta",
    )
    return {
        "schema_version": league.PAIR_SCHEMA_VERSION,
        "kind": league.PAIR_KIND,
        "job_spec": job_spec,
        "job_spec_sha256": league.canonical_sha256(job_spec),
        "strata": strata,
        "pairwise": pairwise,
    }


def _fake_prepared(
    spec: league.CompetitorSpec,
) -> league.PreparedCompetitor:
    selection = balanced.CheckpointSelection(
        directory=spec.checkpoint.resolve(),
        requested_step=spec.step,
        selected_step=spec.step,
        available_steps=(spec.step,),
    )
    config = SimpleNamespace(
        env=SimpleNamespace(id="hex", board_size=6),
        eval=SimpleNamespace(
            player_search=SimpleNamespace(),
            player_action_commitment_type="posterior_argmax",
        ),
        training=SimpleNamespace(
            losses=SimpleNamespace(q_loss_weight_mode="evidence_mass")
        ),
    )
    loaded = balanced.LoadedCheckpointMetadata(
        selection=selection,
        metadata={"step": spec.step},
        config=config,
    )
    return league.PreparedCompetitor(
        spec=spec,
        loaded=loaded,
        effective_config=config,
        overrides={
            "requested": {
                "root_action_estimator": spec.root_action_estimator,
                "kappa": spec.kappa,
            }
        },
        metadata_sha256=f"meta-{spec.competitor_id}",
        checkpoint_tree_sha256=f"tree-{spec.competitor_id}",
        effective_eval_sha256=f"eval-{spec.competitor_id}",
    )


def test_pair_summary_is_balanced_and_inverts_both_perspectives():
    strata, pairwise = league.summarize_pair_returns(
        _returns(),
        competitor_a="alpha",
        competitor_b="beta",
    )

    assert len(strata) == 4
    assert [
        (
            item["competitor_a_logical_player_id"],
            item["competitor_a_seat"],
            item["competitor_b_logical_player_id"],
            item["competitor_b_seat"],
        )
        for item in strata
    ] == [
        (0, "first", 1, "second"),
        (0, "second", 1, "first"),
        (1, "first", 0, "second"),
        (1, "second", 0, "first"),
    ]
    assert pairwise["overall"]["competitor_a"]["wins"] == 8
    assert pairwise["overall"]["competitor_b"]["wins"] == 8
    assert pairwise["by_seat"]["first"]["competitor_a"]["wins"] == 6
    assert pairwise["by_seat"]["second"]["competitor_a"]["wins"] == 2
    assert pairwise["by_seat"]["first"]["competitor_b"]["wins"] == 6
    assert pairwise["by_seat"]["second"]["competitor_b"]["wins"] == 2
    assert (
        pairwise["competitor_a"]["by_seat"]["first"]
        == pairwise["by_seat"]["first"]["competitor_a"]
    )
    assert (
        pairwise["competitor_b"]["by_seat"]["second"]
        == pairwise["by_seat"]["second"]["competitor_b"]
    )


def test_every_pair_summary_has_the_expected_wilson_interval():
    strata, pairwise = league.summarize_pair_returns(
        _returns(),
        competitor_a="alpha",
        competitor_b="beta",
    )
    summaries = [
        side
        for stratum in strata
        for side in (
            stratum["competitor_a"],
            stratum["competitor_b"],
        )
    ]
    summaries.extend(
        [
            pairwise["overall"]["competitor_a"],
            pairwise["overall"]["competitor_b"],
            *pairwise["competitor_a"]["by_seat"].values(),
            *pairwise["competitor_b"]["by_seat"].values(),
        ]
    )

    for summary in summaries:
        expected = balanced.wilson_interval(
            summary["wins"],
            summary["games"],
        )
        assert summary["wilson_95"] == pytest.approx(expected)
        assert 0.0 <= summary["wilson_95"][0]
        assert summary["wilson_95"][1] <= 1.0


def test_raw_returns_reuse_canonical_rng_and_coordinate_contract():
    pair = league.PairSpec(
        competitor_a="alpha",
        competitor_b="beta",
        games=16,
        batch_size=2,
        seed=77,
        include_game_returns=True,
    )
    chunks = [
        [[1, -1], [1, 1]],
        [[-1, -1], [1, -1]],
        [[1, 1], [-1, 1]],
        [[-1, 1], [-1, -1]],
    ]
    run_keys = jax.random.split(jax.random.PRNGKey(pair.seed), 8)

    payload = league.build_league_game_returns(
        chunks,
        run_keys=run_keys,
        pair=pair,
    )

    assert payload["kind"] == league.GAME_RETURNS_KIND
    assert payload["perspective"]["returns"] == "competitor_a"
    assert balanced.flatten_game_returns(payload) == [
        1,
        -1,
        1,
        1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
    ]
    assert len(payload["pairing_layout_sha256"]) == 64


def test_evaluation_overrides_are_independent_and_do_not_mutate_config():
    @dataclass
    class Active:
        root_action_estimator: str = "winner_mc"
        root_policy_target_estimator: str = "winner_mc"
        prefix_cdf_half_width: int = 20
        kappa: float = 3.0

    @dataclass
    class Search:
        dirichlet_thompson: Active
        kind: str = "dirichlet_thompson"

        def active(self):
            return self.dirichlet_thompson

    @dataclass
    class Eval:
        player_search: Search

    @dataclass
    class Config:
        eval: Eval

    original = Config(eval=Eval(player_search=Search(Active())))
    changed, provenance = league.override_evaluation_search(
        original,
        root_action_estimator="prefix_cdf",
        prefix_cdf_half_width=10,
        kappa=7.5,
    )

    assert (
        changed.eval.player_search.dirichlet_thompson.root_action_estimator
        == "prefix_cdf"
    )
    assert changed.eval.player_search.dirichlet_thompson.kappa == 7.5
    assert (
        changed.eval.player_search.dirichlet_thompson.prefix_cdf_half_width
        == 10
    )
    assert (
        changed.eval.player_search.dirichlet_thompson
        .root_policy_target_estimator
        == "winner_mc"
    )
    assert (
        original.eval.player_search.dirichlet_thompson.root_action_estimator
        == "winner_mc"
    )
    assert original.eval.player_search.dirichlet_thompson.kappa == 3.0
    assert (
        original.eval.player_search.dirichlet_thompson.prefix_cdf_half_width
        == 20
    )
    assert provenance["stored"]["kappa"] == 3.0
    assert provenance["effective"]["kappa"] == 7.5
    assert provenance["effective"]["prefix_cdf_grid_points"] == 21
    assert provenance["checkpoint_exact"] is False


def test_tree_hash_is_order_stable_and_content_sensitive(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "nested").mkdir(parents=True)
    (second / "nested").mkdir(parents=True)
    (first / "z").write_text("one", encoding="utf-8")
    (first / "nested" / "a").write_text("two", encoding="utf-8")
    (second / "nested" / "a").write_text("two", encoding="utf-8")
    (second / "z").write_text("one", encoding="utf-8")

    assert league.tree_sha256(first) == league.tree_sha256(second)
    (second / "z").write_text("changed", encoding="utf-8")
    assert league.tree_sha256(first) != league.tree_sha256(second)


def test_manifest_expands_all_unordered_pairs_and_relative_paths(
    tmp_path: Path,
):
    manifest_path = tmp_path / "league.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": league.MANIFEST_SCHEMA_VERSION,
                "kind": league.MANIFEST_KIND,
                "output_directory": "results",
                "games": 64,
                "batch_size": 4,
                "seed": 91,
                "include_game_returns": False,
                "competitors": [
                    {
                        "id": "a",
                        "checkpoint": "checkpoints/a",
                        "step": 25,
                    },
                    {
                        "id": "b-q21",
                        "checkpoint": "checkpoints/b",
                        "step": 50,
                        "root_action_estimator": "prefix_cdf",
                    },
                    {
                        "id": "c-kappa",
                        "checkpoint": "checkpoints/c",
                        "step": 75,
                        "kappa": 9.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = league.load_manifest(manifest_path)

    assert manifest.output_directory == (tmp_path / "results").resolve()
    assert [
        (pair.competitor_a, pair.competitor_b)
        for pair in manifest.pairs
    ] == [("a", "b-q21"), ("a", "c-kappa"), ("b-q21", "c-kappa")]
    assert manifest.competitors[0].checkpoint == (
        tmp_path / "checkpoints" / "a"
    ).resolve()
    assert manifest.competitors[1].root_action_estimator == "prefix_cdf"
    assert manifest.competitors[2].kappa == 9.0
    assert all(pair.games == 64 for pair in manifest.pairs)


def test_manifest_requires_exact_steps_and_rejects_reversed_duplicate(
    tmp_path: Path,
):
    base = {
        "schema_version": league.MANIFEST_SCHEMA_VERSION,
        "kind": league.MANIFEST_KIND,
        "output_directory": "out",
        "games": 16,
        "batch_size": 2,
        "seed": 1,
        "competitors": [
            {"id": "a", "checkpoint": "a", "step": 1},
            {"id": "b", "checkpoint": "b", "step": 2},
        ],
    }
    missing_step = dict(base)
    missing_step["competitors"] = [
        {"id": "a", "checkpoint": "a"},
        {"id": "b", "checkpoint": "b", "step": 2},
    ]
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps(missing_step), encoding="utf-8")
    with pytest.raises(ValueError, match=r"competitors\[0\]\.step"):
        league.load_manifest(missing_path)

    duplicate = dict(base)
    duplicate["pairs"] = [{"a": "a", "b": "b"}, {"a": "b", "b": "a"}]
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates unordered pair"):
        league.load_manifest(duplicate_path)


def test_pair_artifacts_are_create_once_and_job_hash_checked(tmp_path: Path):
    path = tmp_path / "pair.json"
    payload = _minimal_pair_payload({"experiment": "one"})
    digest = league._write_json_create_once(path, payload)
    original = path.read_bytes()

    assert digest == league.file_sha256(path)
    reused = league.reuse_pair_artifact(
        path,
        payload["job_spec_sha256"],
    )
    assert reused is not None
    assert reused[1] == digest
    with pytest.raises(FileExistsError):
        league._write_json_create_once(
            path,
            _minimal_pair_payload({"experiment": "two"}),
        )
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="different job spec"):
        league.reuse_pair_artifact(
            path,
            league.canonical_sha256({"experiment": "two"}),
        )


def test_resume_reuses_valid_pair_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    specs = (
        league.CompetitorSpec("alpha", tmp_path / "a", 25),
        league.CompetitorSpec("beta", tmp_path / "b", 50),
    )
    pair = league.PairSpec(
        "alpha",
        "beta",
        games=16,
        batch_size=2,
        seed=4,
        include_game_returns=False,
    )
    manifest = league.LeagueManifest(
        path=tmp_path / "manifest.json",
        file_sha256="manifest-sha",
        output_directory=tmp_path / "results",
        competitors=specs,
        pairs=(pair,),
    )
    prepared = {
        spec.competitor_id: _fake_prepared(spec)
        for spec in specs
    }
    monkeypatch.setattr(
        league,
        "prepare_competitor",
        lambda spec: prepared[spec.competitor_id],
    )
    job_spec = league.build_job_spec(
        pair,
        prepared["alpha"],
        prepared["beta"],
    )
    job_hash = league.canonical_sha256(job_spec)
    output = league._default_pair_output(
        manifest.output_directory,
        pair,
        job_hash,
    )
    payload = _minimal_pair_payload(job_spec)
    league._write_json_create_once(output, payload)

    monkeypatch.setattr(
        league,
        "load_prepared_model",
        lambda *args, **kwargs: pytest.fail("resume loaded a model"),
    )
    summary = league.run_manifest(
        manifest,
        summary_output=None,
        reproduction_argv=("hex_checkpoint_league.py", "matrix"),
    )

    assert summary["kind"] == league.RUN_KIND
    assert summary["runtime"]["created"] == 0
    assert summary["runtime"]["reused"] == 1
    assert summary["runtime"]["models_loaded"] == 0
    assert summary["pairs"][0]["status"] == "reused"
    assert summary["pair_artifacts"] == [
        {
            "path": str(output),
            "sha256": league.file_sha256(output),
            "job_spec_sha256": job_hash,
        }
    ]


def test_validate_pair_payload_rejects_non_inverse_results():
    payload = _minimal_pair_payload({"experiment": "tamper"})
    payload["strata"][0]["competitor_b"]["wins"] += 1

    with pytest.raises(ValueError, match="not inverse"):
        league.validate_pair_payload(payload)


def test_positive_float_validation_rejects_nan_and_zero():
    for value in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="finite positive"):
            league._optional_positive_float(value, "kappa")


def test_jitted_league_evaluator_keeps_identity_across_logical_ids(
    monkeypatch: pytest.MonkeyPatch,
):
    class EmptyModel(nnx.Module):
        pass

    traced_behaviors: list[tuple[str, str]] = []

    def fake_make_search_player(
        env,
        model,
        search,
        commitment,
        *,
        q_loss_weight_mode,
    ):
        del env, model, commitment
        traced_behaviors.append((search.label, q_loss_weight_mode))

        def player(state, key):
            del key
            legal = state.legal_action_mask
            if search.label == "a":
                action = jnp.argmax(legal, axis=-1)
            else:
                reverse_index = jnp.argmax(legal[..., ::-1], axis=-1)
                action = legal.shape[-1] - 1 - reverse_index
            return PlayerOutput(
                action=action.astype(jnp.int32),
                posterior=None,
            )

        return player

    monkeypatch.setattr(
        league,
        "make_search_player",
        fake_make_search_player,
    )

    def config(label: str, q_mode: str):
        return SimpleNamespace(
            eval=SimpleNamespace(
                player_search=SimpleNamespace(label=label),
                player_action_commitment_type="posterior_argmax",
            ),
            training=SimpleNamespace(
                losses=SimpleNamespace(q_loss_weight_mode=q_mode)
            ),
        )

    evaluator = league.make_league_evaluator(
        make_env("hex", 3),
        config("a", "a-mode"),
        config("b", "b-mode"),
        batch_size=2,
    )
    key = jax.random.PRNGKey(19)
    model_a = EmptyModel()
    model_b = EmptyModel()
    a0_first = evaluator(
        key,
        model_a,
        model_b,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True),
    ).competitor_a_returns
    a1_first = evaluator(
        key,
        model_a,
        model_b,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(True),
    ).competitor_a_returns
    a0_second = evaluator(
        key,
        model_a,
        model_b,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
    ).competitor_a_returns
    a1_second = evaluator(
        key,
        model_a,
        model_b,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
    ).competitor_a_returns

    assert jnp.array_equal(a0_first, a1_first)
    assert jnp.array_equal(a0_second, a1_second)
    assert ("a", "a-mode") in traced_behaviors
    assert ("b", "b-mode") in traced_behaviors
