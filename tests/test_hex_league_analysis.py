from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts import hex_league_analysis as league


def _summary(
    wins: int,
    games: int = 10,
    *,
    draws: int = 0,
) -> dict[str, float | int]:
    losses = games - wins - draws
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / games,
        "score_rate": (wins + 0.5 * draws) / games,
    }


def _reverse(summary: dict[str, float | int]) -> dict[str, float | int]:
    games = int(summary["games"])
    wins = int(summary["losses"])
    draws = int(summary["draws"])
    return _summary(wins, games, draws=draws)


def _artifact(
    competitor_a: str,
    competitor_b: str,
    *,
    a_first_wins: int,
    a_second_wins: int,
    games: int = 10,
    a_first_draws: int = 0,
    a_second_draws: int = 0,
) -> dict:
    a_first = _summary(a_first_wins, games, draws=a_first_draws)
    a_second = _summary(a_second_wins, games, draws=a_second_draws)
    return {
        "schema_version": 1,
        "kind": league.PAIR_KIND,
        "competitors": {
            "a": {"id": competitor_a},
            "b": {"id": competitor_b},
        },
        "pairwise": {
            "competitor_a": {
                "id": competitor_a,
                "by_seat": {
                    "first": a_first,
                    "second": a_second,
                },
            },
            "competitor_b": {
                "id": competitor_b,
                "by_seat": {
                    "first": _reverse(a_second),
                    "second": _reverse(a_first),
                },
            },
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _cycle_inputs(tmp_path: Path) -> league.LoadedInputs:
    # Balanced dominance is A>B, B>C, C>A, each by 0.60 to 0.40.
    paths = [
        _write(
            tmp_path / "a-b.json",
            _artifact(
                "A",
                "B",
                a_first_wins=8,
                a_second_wins=4,
            ),
        ),
        _write(
            tmp_path / "b-c.json",
            _artifact(
                "B",
                "C",
                a_first_wins=8,
                a_second_wins=4,
            ),
        ),
        _write(
            tmp_path / "a-c.json",
            _artifact(
                "A",
                "C",
                a_first_wins=7,
                a_second_wins=1,
            ),
        ),
    ]
    return league.load_inputs(paths)


def test_analysis_keeps_seats_primary_and_detects_three_cycle(
    tmp_path: Path,
):
    result = league.analyze_league(
        _cycle_inputs(tmp_path),
        bootstrap_replicates=500,
        bootstrap_seed=19,
        rating_regularization=0.1,
    )

    assert result["competitor_ids"] == ["A", "B", "C"]
    assert result["coverage"]["complete"] is True
    first = result["primary_seat_conditioned"]["first"]["win_rate_matrix"]
    second = result["primary_seat_conditioned"]["second"]["win_rate_matrix"]
    pooled = result["secondary_balanced_pooled"]["score_rate_matrix"]

    assert first[0][1] == pytest.approx(0.8)
    assert second[0][1] == pytest.approx(0.4)
    # Reverse first-seat evidence is the original second-seat loss rate.
    assert first[1][0] == pytest.approx(0.6)
    assert second[1][0] == pytest.approx(0.2)
    assert pooled[0][1] == pytest.approx(0.6)
    assert pooled[1][0] == pytest.approx(0.4)
    assert result["nontransitivity"]["three_cycle_count"] == 1
    assert result["nontransitivity"]["three_cycles"][0]["cycle"] == [
        "A",
        "B",
        "C",
        "A",
    ]
    # A scalar transitive fit collapses this symmetric rock-paper-scissors
    # league to equal ratings; the explicit cycle still preserves the signal.
    ratings = result["seat_adjusted_regularized_bradley_terry"]["ratings"]
    assert {item["elo_like"] for item in ratings} == {0.0}

    robust_a = result["robust_first_seat_conversion"]["players"]["A"]
    assert robust_a["worst_opponent_by_point_estimate"] == "C"
    assert robust_a["minimum_first_seat_win_rate"] == pytest.approx(0.7)
    assert (
        robust_a["minimum_bonferroni_simultaneous_lower_95"]
        < robust_a["minimum_first_seat_win_rate"]
    )


def test_same_balanced_scalar_does_not_hide_different_seat_profiles(
    tmp_path: Path,
):
    paths = [
        _write(
            tmp_path / "a-b.json",
            _artifact("A", "B", a_first_wins=9, a_second_wins=1),
        ),
        _write(
            tmp_path / "c-d.json",
            _artifact("C", "D", a_first_wins=5, a_second_wins=5),
        ),
    ]

    result = league.analyze_league(
        league.load_inputs(paths),
        bootstrap_replicates=200,
        bootstrap_seed=8,
    )
    ids = result["competitor_ids"]
    index = {competitor_id: offset for offset, competitor_id in enumerate(ids)}
    pooled = result["secondary_balanced_pooled"]["score_rate_matrix"]
    first = result["primary_seat_conditioned"]["first"]["win_rate_matrix"]
    second = result["primary_seat_conditioned"]["second"]["win_rate_matrix"]

    assert pooled[index["A"]][index["B"]] == pytest.approx(0.5)
    assert pooled[index["C"]][index["D"]] == pytest.approx(0.5)
    assert first[index["A"]][index["B"]] == pytest.approx(0.9)
    assert second[index["A"]][index["B"]] == pytest.approx(0.1)
    assert first[index["C"]][index["D"]] == pytest.approx(0.5)
    assert second[index["C"]][index["D"]] == pytest.approx(0.5)


def test_cycle_can_be_point_detected_without_overclaiming_confidence(
    tmp_path: Path,
):
    result = league.analyze_league(
        _cycle_inputs(tmp_path),
        bootstrap_replicates=500,
        bootstrap_seed=21,
    )

    cycle = result["nontransitivity"]["three_cycles"][0]
    assert cycle["all_edges_bootstrap_lower_above_threshold"] is False
    assert "point estimates" in result["nontransitivity"][
        "confidence_flag_note"
    ]


def test_manifest_relative_paths_and_hashes_are_verified(tmp_path: Path):
    pair_path = _write(
        tmp_path / "pairs" / "a-b.json",
        _artifact("A", "B", a_first_wins=8, a_second_wins=2),
    )
    digest = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": league.MANIFEST_KIND,
        "roster": [{"id": "A"}, {"id": "B"}],
        "pair_artifacts": [
            {
                "path": "pairs/a-b.json",
                "sha256": digest,
            }
        ],
    }
    manifest_path = _write(tmp_path / "manifest.json", manifest)

    loaded = league.load_inputs([manifest_path])

    assert loaded.roster_ids == ("A", "B")
    assert len(loaded.evidence) == 1
    assert loaded.evidence[0].file_sha256 == digest
    assert loaded.manifests[0]["path"] == str(manifest_path.resolve())

    manifest["pair_artifacts"][0]["sha256"] = "0" * 64
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="does not match manifest"):
        league.load_inputs([manifest_path])


def test_run_summary_kind_and_nested_roster_are_supported(tmp_path: Path):
    pair_path = _write(
        tmp_path / "pairs" / "a-b.json",
        _artifact("A", "B", a_first_wins=8, a_second_wins=2),
    )
    run_summary = _write(
        tmp_path / "run.json",
        {
            "schema_version": 1,
            "kind": league.RUN_KIND,
            "roster": {
                "competitors": [{"id": "A"}, {"id": "B"}],
            },
            "pair_artifacts": [
                {
                    "path": "pairs/a-b.json",
                    "sha256": hashlib.sha256(
                        pair_path.read_bytes()
                    ).hexdigest(),
                    "job_spec_sha256": "f" * 64,
                }
            ],
        },
    )

    loaded = league.load_inputs([run_summary])

    assert loaded.roster_ids == ("A", "B")
    assert len(loaded.evidence) == 1


def test_input_manifest_can_discover_completed_output_directory(
    tmp_path: Path,
):
    _write(
        tmp_path / "league-output" / "a-b.json",
        _artifact("A", "B", a_first_wins=8, a_second_wins=2),
    )
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "kind": league.MANIFEST_KIND,
            "roster": [{"id": "A"}, {"id": "B"}],
            "output_directory": "league-output",
        },
    )

    loaded = league.load_inputs([manifest])

    assert loaded.roster_ids == ("A", "B")
    assert len(loaded.evidence) == 1


def test_pair_loader_rejects_inconsistent_reverse_perspective(
    tmp_path: Path,
):
    artifact = _artifact("A", "B", a_first_wins=8, a_second_wins=2)
    artifact["pairwise"]["competitor_b"]["by_seat"]["first"] = _summary(9)
    path = _write(tmp_path / "bad.json", artifact)

    with pytest.raises(ValueError, match="exact reverse"):
        league.load_pair_artifact(path)


def test_replicate_artifacts_aggregate_counts(tmp_path: Path):
    paths = [
        _write(
            tmp_path / "seed-1.json",
            _artifact("A", "B", a_first_wins=8, a_second_wins=2),
        ),
        _write(
            tmp_path / "seed-2.json",
            _artifact("B", "A", a_first_wins=7, a_second_wins=3),
        ),
    ]
    result = league.analyze_league(
        league.load_inputs(paths),
        bootstrap_replicates=300,
        bootstrap_seed=2,
    )

    cell = result["primary_seat_conditioned"]["first"]["matrix"][0][1]
    # First artifact: A-first 8/10. Second artifact: A-first is the reverse
    # of B-second 3/10, hence 7/10. They aggregate to 15/20.
    assert cell["games"] == 20
    assert cell["wins"] == 15
    assert cell["win_rate"] == pytest.approx(0.75)
    assert result["coverage"]["replicate_artifact_counts"]["A vs B"] == 2


def test_optional_self_play_informs_seat_intercept_and_diagonal(
    tmp_path: Path,
):
    path = _write(
        tmp_path / "a-a.json",
        _artifact("A", "A", a_first_wins=8, a_second_wins=2),
    )

    result = league.analyze_league(
        league.load_inputs([path]),
        bootstrap_replicates=300,
        rating_regularization=0.01,
    )

    assert result["coverage"]["expected_unordered_pairs"] == 0
    assert result["coverage"]["observed_self_play_pairs"] == 1
    assert result["primary_seat_conditioned"]["first"][
        "win_rate_matrix"
    ] == [[pytest.approx(0.8)]]
    assert result["primary_seat_conditioned"]["second"][
        "win_rate_matrix"
    ] == [[pytest.approx(0.2)]]
    assert result["secondary_balanced_pooled"]["score_rate_matrix"] == [
        [pytest.approx(0.5)]
    ]
    assert result["seat_adjusted_regularized_bradley_terry"][
        "equal_ability_first_score_probability"
    ] == pytest.approx(0.8, abs=1e-3)


def test_seat_adjusted_rating_recovers_equal_players_and_seat_advantage():
    ids = ("A", "B")
    directed = {
        ("A", "B", "first"): league.OutcomeSummary(100, 75, 0, 25),
        ("A", "B", "second"): league.OutcomeSummary(100, 25, 0, 75),
        ("B", "A", "first"): league.OutcomeSummary(100, 75, 0, 25),
        ("B", "A", "second"): league.OutcomeSummary(100, 25, 0, 75),
    }

    rating = league._fit_regularized_bradley_terry(
        ids,
        directed,
        regularization=0.01,
    )

    ratings = {
        item["id"]: item["ability_log_odds"]
        for item in rating["ratings"]
    }
    assert ratings["A"] == pytest.approx(ratings["B"], abs=1e-10)
    assert rating["equal_ability_first_score_probability"] == pytest.approx(
        0.75,
        abs=2e-4,
    )
    assert rating["first_seat_elo_like"] > 0.0
    assert "non-transitive" in " ".join(rating["caveats"])


def test_bootstrap_is_reproducible_and_wilson_handles_boundaries(
    tmp_path: Path,
):
    loaded = _cycle_inputs(tmp_path)
    first = league.analyze_league(
        loaded,
        bootstrap_replicates=300,
        bootstrap_seed=55,
    )
    second = league.analyze_league(
        loaded,
        bootstrap_replicates=300,
        bootstrap_seed=55,
    )

    assert first["primary_seat_conditioned"] == second[
        "primary_seat_conditioned"
    ]
    low, high = league.wilson_interval(0, 20)
    assert low == 0.0
    assert 0.0 < high < 0.2
    low, high = league.wilson_interval(20, 20)
    assert 0.8 < low < 1.0
    assert high == 1.0


def test_pooled_wilson_90_reports_registered_equivalence_gate(
    tmp_path: Path,
):
    paths = [
        _write(
            tmp_path / "equivalent.json",
            _artifact(
                "A",
                "B",
                a_first_wins=256,
                a_second_wins=256,
                games=512,
            ),
        ),
        _write(
            tmp_path / "not-equivalent.json",
            _artifact(
                "A",
                "C",
                a_first_wins=307,
                a_second_wins=307,
                games=512,
            ),
        ),
    ]

    result = league.analyze_league(
        league.load_inputs(paths),
        bootstrap_replicates=300,
        bootstrap_seed=56,
    )
    ids = result["competitor_ids"]
    index = {competitor_id: offset for offset, competitor_id in enumerate(ids)}
    matrix = result["secondary_balanced_pooled"]["matrix"]
    equivalent = matrix[index["A"]][index["B"]]
    not_equivalent = matrix[index["A"]][index["C"]]

    expected = league.wilson_interval(512, 1024, z=league.Z_90)
    assert equivalent[
        "seat_balanced_score_rate_wilson_90"
    ] == pytest.approx(expected)
    assert equivalent["raw_pooled_win_rate_wilson_90"] == pytest.approx(
        expected
    )
    assert equivalent["equivalence_90"] == {
        "eligible": True,
        "equal_seat_sample_sizes": True,
        "draw_free": True,
        "acceptance_bounds": [0.45, 0.55],
        "interval_contained": True,
    }
    assert not_equivalent["equivalence_90"]["eligible"] is True
    assert not_equivalent["equivalence_90"]["interval_contained"] is False

    contract = result["secondary_balanced_pooled"][
        "equivalence_contract"
    ]
    assert contract["confidence_level"] == pytest.approx(0.90)
    assert contract["acceptance_bounds"] == pytest.approx([0.45, 0.55])


def test_pooled_score_equivalence_is_ineligible_with_draws(tmp_path: Path):
    path = _write(
        tmp_path / "draws.json",
        _artifact(
            "A",
            "B",
            a_first_wins=250,
            a_second_wins=250,
            a_first_draws=12,
            a_second_draws=12,
            games=512,
        ),
    )

    result = league.analyze_league(
        league.load_inputs([path]),
        bootstrap_replicates=200,
    )
    cell = result["secondary_balanced_pooled"]["matrix"][0][1]

    assert cell["seat_balanced_score_rate_wilson_90_valid"] is False
    assert cell["seat_balanced_score_rate_wilson_90"] is None
    assert cell["equivalence_90"]["eligible"] is False
    assert cell["equivalence_90"]["draw_free"] is False
    assert cell["equivalence_90"]["interval_contained"] is None
    assert cell["raw_pooled_win_rate_wilson_90"] is not None


def test_equivalence_margin_is_configurable_and_validated(tmp_path: Path):
    path = _write(
        tmp_path / "a-b.json",
        _artifact(
            "A",
            "B",
            a_first_wins=256,
            a_second_wins=256,
            games=512,
        ),
    )
    loaded = league.load_inputs([path])

    result = league.analyze_league(
        loaded,
        bootstrap_replicates=200,
        equivalence_margin=0.01,
    )
    cell = result["secondary_balanced_pooled"]["matrix"][0][1]
    assert cell["equivalence_90"]["acceptance_bounds"] == pytest.approx(
        [0.49, 0.51]
    )
    assert cell["equivalence_90"]["interval_contained"] is False

    for invalid in (0.0, 0.5, math.inf, math.nan):
        with pytest.raises(ValueError, match="equivalence_margin"):
            league.analyze_league(
                loaded,
                bootstrap_replicates=200,
                equivalence_margin=invalid,
            )


def test_incomplete_roster_coverage_is_explicit(tmp_path: Path):
    pair_path = _write(
        tmp_path / "a-b.json",
        _artifact("A", "B", a_first_wins=8, a_second_wins=2),
    )
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "kind": league.MANIFEST_KIND,
            "roster": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
            "pair_artifacts": [{"path": pair_path.name}],
        },
    )

    result = league.analyze_league(
        league.load_inputs([manifest]),
        bootstrap_replicates=300,
    )

    assert result["coverage"]["complete"] is False
    assert result["coverage"]["missing_unordered_pairs"] == [
        ["A", "C"],
        ["B", "C"],
    ]
    assert "incomplete" in result["interpretation_contract"]["warnings"][0]


def test_cli_writes_reproducible_local_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    loaded = _cycle_inputs(tmp_path)
    paths = [item.path for item in loaded.evidence]
    output = tmp_path / "analysis.json"
    monkeypatch.chdir(tmp_path)

    league.main(
        [
            *(str(path) for path in paths),
            "--output",
            str(output),
            "--bootstrap-replicates",
            "300",
            "--bootstrap-seed",
            "77",
            "--equivalence-margin",
            "0.04",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["kind"] == league.RESULT_KIND
    assert payload["local_only"]["network_access"] is False
    assert payload["coverage"]["complete"] is True
    assert payload["secondary_balanced_pooled"]["equivalence_contract"][
        "margin"
    ] == pytest.approx(0.04)
    assert payload["reproduction"]["working_directory"] == str(
        tmp_path.resolve()
    )
    assert math.isfinite(
        payload["seat_adjusted_regularized_bradley_terry"][
            "first_seat_log_odds"
        ]
    )
