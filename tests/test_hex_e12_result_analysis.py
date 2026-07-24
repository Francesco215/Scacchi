from __future__ import annotations

import numpy as np
import pytest

from scripts import hex_e12_result_analysis as analysis


def test_wilson90_matches_registered_step100_boundary():
    lower, upper = analysis.wilson_interval(
        4480,
        8192,
        confidence=0.90,
    )

    assert lower == pytest.approx(0.5378144053)
    assert upper == pytest.approx(0.5559046424)
    assert upper > 0.55


def test_step100_gate_distinguishes_containment_from_seat_error():
    summary = {
        "pooled_win_rate": 4480 / 8192,
        "pooled_wilson90": list(
            analysis.wilson_interval(4480, 8192, confidence=0.90)
        ),
        "e_seat": 0.04931640625,
    }

    gates = analysis._balanced_step_gates(100, summary)

    assert gates["wilson90_contained_in_0p45_0p55"] is False
    assert gates["e_seat_at_most_0p08"] is True
    assert gates["all_passed"] is False


def test_synchronized_roster_bootstrap_is_deterministic_and_paired():
    # Four strata, four rows, two anchors.  Anchor 0 has +0.25 mean and
    # anchor 1 has -0.25 mean, so the roster point estimate is exactly zero.
    deltas = np.zeros((4, 4, 2), dtype=np.float64)
    deltas[:, 0, 0] = 1.0
    deltas[:, 1, 1] = -1.0

    first = analysis.synchronized_roster_bootstrap(
        deltas,
        seed=17,
        replicates=1_000,
        batch_size=31,
    )
    second = analysis.synchronized_roster_bootstrap(
        deltas,
        seed=17,
        replicates=1_000,
        batch_size=31,
    )

    assert first == second
    assert first["point_estimate"] == pytest.approx(0.0)
    assert first["point_by_anchor"] == pytest.approx([0.25, -0.25])


@pytest.mark.parametrize(
    "wins,games",
    [(-1, 10), (11, 10), (1, 0)],
)
def test_wilson_rejects_invalid_counts(wins: int, games: int):
    with pytest.raises(ValueError, match="Wilson counts"):
        analysis.wilson_interval(wins, games, confidence=0.90)
