from __future__ import annotations

import json
from pathlib import Path

import jax
import pytest

from scripts import hex_balanced_eval as balanced
from scripts import hex_paired_eval_analysis as paired


def _game_returns(
    returns: list[list[list[int]]],
    *,
    seed: int = 73,
) -> dict:
    return balanced.build_game_returns_payload(
        returns,
        run_keys=jax.random.split(jax.random.PRNGKey(seed), 4),
        seed=seed,
        games=16,
        games_per_stratum=4,
        batch_size=4,
    )


def _result(game_returns: dict, estimator: str) -> dict:
    return {
        "candidate_checkpoint_selection": {
            "directory": "/tmp/candidate",
            "requested_step": 100,
            "selected_step": 100,
            "selection_mode": "exact",
            "available_steps": [0, 100],
        },
        "baseline_checkpoint_selection": {
            "directory": "/tmp/baseline",
            "requested_step": None,
            "selected_step": 300,
            "selection_mode": "latest",
            "available_steps": [300],
        },
        "candidate_root_action_estimator": estimator,
        "games": game_returns["games"],
        "games_per_stratum": game_returns["games_per_stratum"],
        "batch_size": game_returns["batch_size"],
        "seed": game_returns["seed"],
        "environment": {"id": "hex", "board_size": 6},
        "role_and_rng_construction": {"role_balance": "four cells"},
        "monitor": {
            "candidate_search": {
                "kind": "dirichlet_thompson",
                "num_simulations": 32,
                "root_action_estimator": estimator,
            },
            "candidate_action_commitment": "posterior_sample",
            "opponent_search": {
                "kind": "dirichlet_thompson",
                "num_simulations": 300,
            },
            "opponent_action_commitment": "posterior_argmax",
        },
        "game_returns": game_returns,
    }


def _write_result(
    path: Path,
    returns: list[list[list[int]]],
    estimator: str,
    *,
    seed: int = 73,
) -> Path:
    path.write_text(
        json.dumps(_result(_game_returns(returns, seed=seed), estimator)),
        encoding="utf-8",
    )
    return path


CONTROL = [
    [[1, 1, -1, -1]],
    [[-1, -1, -1, 1]],
    [[1, -1, -1, -1]],
    [[-1, -1, 1, 1]],
]

TREATMENT = [
    [[1, 1, 1, -1]],
    [[-1, -1, -1, -1]],
    [[1, 1, -1, -1]],
    [[-1, -1, -1, 1]],
]


def test_paired_analysis_reports_seat_structure_and_optimal_error(
    tmp_path: Path,
):
    control = paired.load_result(
        _write_result(tmp_path / "m32.json", CONTROL, "winner_mc")
    )
    treatment = paired.load_result(
        _write_result(
            tmp_path / "q21.json",
            TREATMENT,
            "prefix_cdf",
        )
    )

    result = paired.analyze_pair(
        control,
        treatment,
        control_label="M32",
        treatment_label="Q21",
    )

    assert result["pairing_validation"]["valid"] is True
    assert (
        result["pairing_validation"]["per_game_coordinates_compared"]
        == 16
    )
    methods = result["methods"]
    assert methods["control"]["overall"]["win_rate"] == pytest.approx(0.375)
    assert methods["treatment"]["overall"]["win_rate"] == pytest.approx(
        0.375
    )
    assert methods["control"]["by_seat"]["first"]["win_rate"] == pytest.approx(
        0.375
    )
    assert methods["treatment"]["by_seat"]["first"]["win_rate"] == pytest.approx(
        0.625
    )
    assert methods["control"]["by_seat"]["second"]["win_rate"] == pytest.approx(
        0.375
    )
    assert methods["treatment"]["by_seat"]["second"]["win_rate"] == pytest.approx(
        0.125
    )

    overall = result["paired"]["overall"]
    assert overall["delta_win_rate"] == pytest.approx(0.0)
    assert overall["discordance_table"] == {
        "both_win": 4,
        "control_win_treatment_not_win": 2,
        "control_not_win_treatment_win": 2,
        "neither_win": 8,
        "discordant_total": 4,
    }
    assert overall["exact_mcnemar"]["p_value_two_sided"] == 1.0
    assert result["paired"]["by_seat"]["first"][
        "delta_win_rate"
    ] == pytest.approx(0.25)
    assert result["paired"]["by_seat"]["second"][
        "delta_win_rate"
    ] == pytest.approx(-0.25)

    seat_error = result["paired"]["seat_optimal_error"]
    assert seat_error["control"] == pytest.approx(0.5)
    assert seat_error["treatment"] == pytest.approx(0.25)
    assert seat_error["delta"] == pytest.approx(-0.25)
    assert seat_error["delta_ci95"]["estimate"] == pytest.approx(-0.25)
    assert seat_error["delta_ci95"]["interval"] is not None


@pytest.mark.parametrize(
    ("control_only", "treatment_only", "expected"),
    [
        (0, 0, 1.0),
        (0, 4, 0.125),
        (1, 4, 0.375),
        (2, 2, 1.0),
    ],
)
def test_exact_two_sided_mcnemar(
    control_only: int,
    treatment_only: int,
    expected: float,
):
    assert paired.exact_mcnemar_two_sided(
        control_only,
        treatment_only,
    ) == pytest.approx(expected)


def test_exact_mcnemar_keeps_a_decimal_when_float_underflows():
    result = paired._exact_mcnemar_result(0, 4096)

    assert result["p_value_two_sided"] == 0.0
    assert result["floating_point_underflow"] is True
    assert result["p_value_two_sided_decimal"].endswith("E-1233")


def test_loader_recomputes_layout_and_returns_digests(tmp_path: Path):
    result = _result(_game_returns(CONTROL), "winner_mc")
    result["game_returns"]["strata"][0]["chunks"][0][
        "rng_key_data"
    ][0] += 1
    path = tmp_path / "corrupt-layout.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="pairing_layout_sha256"):
        paired.load_result(path)

    result = _result(_game_returns(CONTROL), "winner_mc")
    result["game_returns"]["strata"][0]["chunks"][0]["returns"][0] = -1
    path = tmp_path / "corrupt-returns.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="summary|returns_sha256"):
        paired.load_result(path)


def test_pairing_rejects_different_rng_coordinates(tmp_path: Path):
    control = paired.load_result(
        _write_result(tmp_path / "m32.json", CONTROL, "winner_mc")
    )
    treatment = paired.load_result(
        _write_result(
            tmp_path / "q21.json",
            TREATMENT,
            "prefix_cdf",
            seed=74,
        )
    )

    with pytest.raises(ValueError, match="pairing_layout_sha256 mismatch"):
        paired.validate_pairing(control, treatment)


def test_pairing_rejects_checkpoint_context_mismatch(tmp_path: Path):
    control_path = _write_result(
        tmp_path / "m32.json",
        CONTROL,
        "winner_mc",
    )
    treatment_payload = _result(
        _game_returns(TREATMENT),
        "prefix_cdf",
    )
    treatment_payload["candidate_checkpoint_selection"][
        "selected_step"
    ] = 150
    treatment_path = tmp_path / "q21.json"
    treatment_path.write_text(
        json.dumps(treatment_payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="candidate_checkpoint_selection",
    ):
        paired.validate_pairing(
            paired.load_result(control_path),
            paired.load_result(treatment_path),
        )


def test_pairing_can_isolate_kappa_as_the_only_varying_factor(
    tmp_path: Path,
):
    control_payload = _result(_game_returns(CONTROL), "prefix_cdf")
    treatment_payload = _result(_game_returns(TREATMENT), "prefix_cdf")
    for payload, kappa in (
        (control_payload, 3.0),
        (treatment_payload, 0.5),
    ):
        payload["candidate_kappa"] = {
            "stored": 3.0,
            "effective": kappa,
        }
        payload["monitor"]["candidate_search"]["kappa"] = kappa
    control_path = tmp_path / "kappa3.json"
    treatment_path = tmp_path / "kappa0_5.json"
    control_path.write_text(json.dumps(control_payload), encoding="utf-8")
    treatment_path.write_text(
        json.dumps(treatment_payload),
        encoding="utf-8",
    )

    result = paired.analyze_pair(
        paired.load_result(control_path),
        paired.load_result(treatment_path),
        control_label="kappa=3",
        treatment_label="kappa=0.5",
        varying_factor="kappa",
    )

    varying = result["pairing_validation"]["deliberately_varying_factor"]
    assert varying["field"] == "candidate_kappa"
    assert varying["control"]["effective"] == 3.0
    assert varying["treatment"]["effective"] == 0.5


def test_pairing_can_isolate_checkpoint_as_the_only_varying_factor(
    tmp_path: Path,
):
    control_payload = _result(_game_returns(CONTROL), "prefix_cdf")
    treatment_payload = _result(_game_returns(TREATMENT), "prefix_cdf")
    treatment_payload["candidate_checkpoint_selection"] = {
        **treatment_payload["candidate_checkpoint_selection"],
        "directory": "/tmp/treatment",
    }
    control_path = tmp_path / "control.json"
    treatment_path = tmp_path / "treatment.json"
    control_path.write_text(json.dumps(control_payload), encoding="utf-8")
    treatment_path.write_text(
        json.dumps(treatment_payload),
        encoding="utf-8",
    )

    result = paired.analyze_pair(
        paired.load_result(control_path),
        paired.load_result(treatment_path),
        control_label="E8",
        treatment_label="E9",
        varying_factor="checkpoint",
    )

    varying = result["pairing_validation"]["deliberately_varying_factor"]
    assert varying["field"] == "candidate_checkpoint_selection"
    assert varying["control"]["directory"] == "/tmp/candidate"
    assert varying["treatment"]["directory"] == "/tmp/treatment"


def test_cli_writes_a_reproducible_local_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    control_path = _write_result(
        tmp_path / "m32.json",
        CONTROL,
        "winner_mc",
    )
    treatment_path = _write_result(
        tmp_path / "q21.json",
        TREATMENT,
        "prefix_cdf",
    )
    output = tmp_path / "comparison.json"
    monkeypatch.chdir(tmp_path)

    paired.main(
        [
            "--control",
            str(control_path),
            "--treatment",
            str(treatment_path),
            "--output",
            str(output),
        ]
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["kind"] == paired.RESULT_KIND
    assert artifact["local_only"]["external_export"] is False
    assert artifact["inputs"]["control"]["label"] == "control"
    assert artifact["inputs"]["treatment"]["label"] == "treatment"
    assert artifact["paired"]["overall"]["games"] == 16
