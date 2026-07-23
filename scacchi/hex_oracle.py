"""Exact late-game Hex oracle and frozen-corpus utilities.

The solver deliberately does not call PGX while searching.  PGX's internal
Hex board stores connected-component labels whose signs are relative to the
side to move.  :func:`position_from_pgx_state` converts that representation
once to fixed colours:

``0``
    empty
``1``
    colour 0, which connects the top and bottom edges
``2``
    colour 1, which connects the left and right edges

Actions use PGX's flattened ``row * size + column`` convention.  The pie-rule
action at ``size * size`` is intentionally excluded: Scacchi disables it, and
late-game positions cannot legally use it anyway.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Sequence


EMPTY = 0
COLOR_0 = 1
COLOR_1 = 2
FROZEN_CORPUS_FORMAT = "scacchi.hex_oracle.v1"


@dataclass(frozen=True)
class HexOracleResult:
    """Exact result from the perspective of ``current_color``.

    ``outcome`` and every value in ``action_values`` are in ``{-1, 0, +1}``.
    Legal Hex positions never draw, so zero is only a defensive result for an
    invalid fully occupied board with no connecting path.
    """

    outcome: int
    optimal_actions: tuple[int, ...]
    action_values: tuple[tuple[int, int], ...]

    def action_value(self, action: int) -> int:
        for candidate, value in self.action_values:
            if candidate == action:
                return value
        raise KeyError(f"action {action} is not legal in this position")


@dataclass(frozen=True)
class HexPosition:
    """A fixed-colour Hex position suitable for exact solving."""

    size: int
    cells: tuple[int, ...]
    current_color: int

    def __post_init__(self) -> None:
        _validate_position(self.size, self.cells, self.current_color)

    @property
    def empty_count(self) -> int:
        return self.cells.count(EMPTY)

    @property
    def position_id(self) -> str:
        payload = (
            f"{self.size}:{self.current_color}:"
            + "".join(str(cell) for cell in self.cells)
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()[:20]


@dataclass(frozen=True)
class FrozenHexPosition:
    """A position together with its immutable exact-oracle labels."""

    position_id: str
    size: int
    cells: tuple[int, ...]
    current_color: int
    oracle_outcome: int
    optimal_actions: tuple[int, ...]
    action_values: tuple[tuple[int, int], ...]

    @classmethod
    def from_position(cls, position: HexPosition) -> FrozenHexPosition:
        result = solve_hex(position)
        return cls(
            position_id=position.position_id,
            size=position.size,
            cells=position.cells,
            current_color=position.current_color,
            oracle_outcome=result.outcome,
            optimal_actions=result.optimal_actions,
            action_values=result.action_values,
        )

    @property
    def position(self) -> HexPosition:
        return HexPosition(
            size=self.size,
            cells=self.cells,
            current_color=self.current_color,
        )

    @property
    def result(self) -> HexOracleResult:
        return HexOracleResult(
            outcome=self.oracle_outcome,
            optimal_actions=self.optimal_actions,
            action_values=self.action_values,
        )


@dataclass(frozen=True)
class FrozenHexCorpus:
    """Reproducible set of solved non-terminal late-game positions."""

    size: int
    seed: int
    min_empty: int
    max_empty: int
    balanced_outcomes: bool
    attempts: int
    positions: tuple[FrozenHexPosition, ...]
    format: str = FROZEN_CORPUS_FORMAT


@dataclass(frozen=True)
class PolicyOracleAssessment:
    """Decision quality of one policy under exact optimal continuation."""

    expected_outcome: float
    regret: float
    optimal_action_mass: float
    top_action: int
    top_action_is_optimal: bool
    induced_loss_probability: float
    induced_win_probability: float


@dataclass(frozen=True)
class ProperScoreComparison:
    """Proper-score inputs and scores for prior versus search.

    Distributions are ordered ``(loss, win)`` from the root player's
    perspective.  A positive gain means that search is better than the prior.
    """

    oracle_outcome: int
    target_distribution: tuple[float, float]
    prior_distribution: tuple[float, float]
    search_distribution: tuple[float, float]
    prior_log_loss: float
    search_log_loss: float
    log_score_gain: float
    prior_brier_loss: float
    search_brier_loss: float
    brier_score_gain: float


@dataclass(frozen=True)
class OraclePolicyComparison:
    """Exact policy regret and policy-induced proper scores."""

    prior: PolicyOracleAssessment
    search: PolicyOracleAssessment
    regret_reduction: float
    proper_scores: ProperScoreComparison


@dataclass(frozen=True)
class PolicyReadoutNoiseAssessment:
    """Separate search displacement from finite policy-readout noise.

    Every search policy must be an independent Monte Carlo readout of the
    *same* fixed search tree.  The squared-L2 decomposition is useful because,
    for unbiased independent readouts ``P_1`` and ``P_2``,

    ``E ||P_1 - prior||²
       = ||E[P_1] - prior||² + 0.5 E ||P_1 - P_2||²``.

    Thus ``readout_noise_squared_l2`` estimates finite-population noise and
    ``noise_corrected_displacement_squared_l2`` estimates the part attributable
    to the tree posterior.  Jensen--Shannon values (in nats) are also returned
    because they are bounded and easier to compare with policy KL metrics, but
    they do not have the same exact additive decomposition.
    """

    readout_count: int
    prior_to_mean_search_js_nats: float
    mean_prior_to_search_readout_js_nats: float
    mean_pairwise_search_readout_js_nats: float
    mean_prior_search_readout_squared_l2: float
    readout_noise_squared_l2: float
    noise_corrected_displacement_squared_l2: float


def _validate_position(
    size: int,
    cells: Sequence[int],
    current_color: int,
) -> None:
    if size < 1:
        raise ValueError(f"size must be positive; got {size}")
    if len(cells) != size * size:
        raise ValueError(
            f"expected {size * size} cells for size {size}; got {len(cells)}"
        )
    if current_color not in (0, 1):
        raise ValueError(
            f"current_color must be 0 or 1; got {current_color}"
        )
    invalid = sorted(set(int(cell) for cell in cells) - {EMPTY, COLOR_0, COLOR_1})
    if invalid:
        raise ValueError(f"cells contain invalid fixed-colour values: {invalid}")


def _neighbours(action: int, size: int) -> Iterable[int]:
    row, column = divmod(action, size)
    for d_row, d_column in (
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
    ):
        neighbour_row = row + d_row
        neighbour_column = column + d_column
        if (
            0 <= neighbour_row < size
            and 0 <= neighbour_column < size
        ):
            yield neighbour_row * size + neighbour_column


def hex_has_connection(
    cells: Sequence[int],
    *,
    size: int,
    color: int,
) -> bool:
    """Return whether ``color`` has connected its two target edges."""

    _validate_position(size, cells, color)
    stone = color + 1
    if color == 1:
        frontier = [
            row * size
            for row in range(size)
            if cells[row * size] == stone
        ]

        def is_target(action: int) -> bool:
            return action % size == size - 1

    else:
        frontier = [
            column
            for column in range(size)
            if cells[column] == stone
        ]

        def is_target(action: int) -> bool:
            return action // size == size - 1

    visited = set(frontier)
    while frontier:
        action = frontier.pop()
        if is_target(action):
            return True
        for neighbour in _neighbours(action, size):
            if cells[neighbour] == stone and neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return False


def _terminal_outcome(
    size: int,
    cells: tuple[int, ...],
    current_color: int,
) -> int | None:
    color_0_wins = hex_has_connection(cells, size=size, color=0)
    color_1_wins = hex_has_connection(cells, size=size, color=1)
    if color_0_wins and color_1_wins:
        raise ValueError("invalid Hex position: both colours have a connection")
    if color_0_wins:
        return 1 if current_color == 0 else -1
    if color_1_wins:
        return 1 if current_color == 1 else -1
    return None


@lru_cache(maxsize=262_144)
def _solve_cached(
    size: int,
    cells: tuple[int, ...],
    current_color: int,
) -> HexOracleResult:
    terminal_outcome = _terminal_outcome(size, cells, current_color)
    if terminal_outcome is not None:
        return HexOracleResult(
            outcome=terminal_outcome,
            optimal_actions=(),
            action_values=(),
        )

    legal_actions = tuple(
        action for action, cell in enumerate(cells) if cell == EMPTY
    )
    if not legal_actions:
        # This cannot occur for a valid Hex board, but makes the pure function
        # total on defensive/test inputs.
        return HexOracleResult(outcome=0, optimal_actions=(), action_values=())

    best_outcome = -2
    optimal_actions: list[int] = []
    action_values: list[tuple[int, int]] = []
    stone = current_color + 1
    for action in legal_actions:
        child_cells_list = list(cells)
        child_cells_list[action] = stone
        child_cells = tuple(child_cells_list)
        if hex_has_connection(
            child_cells,
            size=size,
            color=current_color,
        ):
            value = 1
        else:
            child = _solve_cached(size, child_cells, 1 - current_color)
            value = -child.outcome
        action_values.append((action, value))
        if value > best_outcome:
            best_outcome = value
            optimal_actions = [action]
        elif value == best_outcome:
            optimal_actions.append(action)

    return HexOracleResult(
        outcome=best_outcome,
        optimal_actions=tuple(optimal_actions),
        action_values=tuple(action_values),
    )


def solve_hex(position: HexPosition) -> HexOracleResult:
    """Solve a Hex position exactly by memoized minimax."""

    return _solve_cached(
        position.size,
        tuple(position.cells),
        position.current_color,
    )


def hex_oracle_cache_info():
    """Expose cache statistics for runtime/memory diagnostics."""

    return _solve_cached.cache_info()


def clear_hex_oracle_cache() -> None:
    """Clear all memoized positions, useful between large corpora."""

    _solve_cached.cache_clear()


def position_from_pgx_state(state: Any) -> HexPosition:
    """Convert a scalar ``pgx.hex.State`` to fixed-colour form.

    Batched states are intentionally rejected.  Oracle evaluation should move
    only a small frozen late-game slice from device to host.
    """

    try:
        game_state = state._x
        raw_board = game_state.board
        flat_board = tuple(int(value) for value in raw_board.tolist())
        current_color = int(game_state.color)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("expected a scalar pgx.hex.State") from error

    size = math.isqrt(len(flat_board))
    if size * size != len(flat_board):
        raise ValueError(
            f"PGX Hex board length is not square: {len(flat_board)}"
        )
    cells = tuple(
        EMPTY
        if component == 0
        else (
            current_color + 1
            if component > 0
            else (1 - current_color) + 1
        )
        for component in flat_board
    )
    return HexPosition(
        size=size,
        cells=cells,
        current_color=current_color,
    )


def solve_pgx_hex_state(state: Any) -> HexOracleResult:
    """Convert and solve a scalar PGX Hex state."""

    return solve_hex(position_from_pgx_state(state))


def canonical_action_sequence(position: HexPosition) -> tuple[int, ...]:
    """Return a deterministic legal move ordering that reaches ``position``.

    Frozen corpus entries store fixed colours rather than their original
    histories.  Hex connectivity is monotone, so a non-terminal final board
    can be reconstructed by interleaving its colour-0 and colour-1 stones in
    any deterministic order.  This helper validates the alternating stone
    counts and is used by the checkpoint harness to recreate PGX observations.
    """

    color_0_actions = tuple(
        action
        for action, cell in enumerate(position.cells)
        if cell == COLOR_0
    )
    color_1_actions = tuple(
        action
        for action, cell in enumerate(position.cells)
        if cell == COLOR_1
    )
    if len(color_0_actions) not in {
        len(color_1_actions),
        len(color_1_actions) + 1,
    }:
        raise ValueError(
            "position does not have alternating Hex stone counts: "
            f"color_0={len(color_0_actions)}, color_1={len(color_1_actions)}"
        )
    expected_color = (len(color_0_actions) + len(color_1_actions)) % 2
    if position.current_color != expected_color:
        raise ValueError(
            "current_color disagrees with the alternating stone counts: "
            f"expected {expected_color}, got {position.current_color}"
        )

    sequence: list[int] = []
    for index, action in enumerate(color_0_actions):
        sequence.append(action)
        if index < len(color_1_actions):
            sequence.append(color_1_actions[index])
    return tuple(sequence)


def _normalised_legal_policy(
    policy: Sequence[float],
    result: HexOracleResult,
) -> tuple[tuple[int, float], ...]:
    if not result.action_values:
        raise ValueError("cannot assess a policy on a terminal position")
    legal_actions = tuple(action for action, _ in result.action_values)
    minimum_length = max(legal_actions) + 1
    if len(policy) < minimum_length:
        raise ValueError(
            f"policy has {len(policy)} actions; need at least {minimum_length}"
        )
    probabilities = tuple(float(policy[action]) for action in legal_actions)
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("legal policy entries must be finite and non-negative")
    mass = sum(probabilities)
    if mass <= 0.0:
        raise ValueError("policy assigns zero mass to every legal action")
    return tuple(
        (action, probability / mass)
        for action, probability in zip(legal_actions, probabilities, strict=True)
    )


def assess_policy_against_oracle(
    policy: Sequence[float],
    result: HexOracleResult,
) -> PolicyOracleAssessment:
    """Measure exact regret of a policy, assuming optimal continuation."""

    normalised = _normalised_legal_policy(policy, result)
    values = dict(result.action_values)
    expected_outcome = sum(
        probability * values[action]
        for action, probability in normalised
    )
    optimal_actions = set(result.optimal_actions)
    optimal_action_mass = sum(
        probability
        for action, probability in normalised
        if action in optimal_actions
    )
    top_action = max(normalised, key=lambda item: item[1])[0]
    win_probability = sum(
        probability
        for action, probability in normalised
        if values[action] == 1
    )
    loss_probability = sum(
        probability
        for action, probability in normalised
        if values[action] == -1
    )
    draw_probability = max(0.0, 1.0 - win_probability - loss_probability)
    # Legal Hex has no draws.  Splitting a defensive draw result equally keeps
    # the returned pair normalized for malformed synthetic boards.
    win_probability += 0.5 * draw_probability
    loss_probability += 0.5 * draw_probability
    return PolicyOracleAssessment(
        expected_outcome=expected_outcome,
        regret=max(0.0, float(result.outcome) - expected_outcome),
        optimal_action_mass=optimal_action_mass,
        top_action=top_action,
        top_action_is_optimal=top_action in optimal_actions,
        induced_loss_probability=loss_probability,
        induced_win_probability=win_probability,
    )


def compare_binary_outcome_probabilities(
    *,
    oracle_outcome: int,
    prior_win_probability: float,
    search_win_probability: float,
    epsilon: float = 1e-12,
) -> ProperScoreComparison:
    """Compare prior/search binary predictions against the exact result."""

    if oracle_outcome not in (-1, 1):
        raise ValueError(
            f"proper scoring requires a decisive outcome; got {oracle_outcome}"
        )
    if not 0.0 < epsilon < 0.5:
        raise ValueError(f"epsilon must be in (0, 0.5); got {epsilon}")
    for name, probability in (
        ("prior_win_probability", prior_win_probability),
        ("search_win_probability", search_win_probability),
    ):
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    target = (1.0, 0.0) if oracle_outcome == -1 else (0.0, 1.0)
    prior = (1.0 - prior_win_probability, prior_win_probability)
    search = (1.0 - search_win_probability, search_win_probability)
    target_index = 0 if oracle_outcome == -1 else 1

    def log_loss(distribution: tuple[float, float]) -> float:
        return -math.log(max(epsilon, distribution[target_index]))

    def brier_loss(distribution: tuple[float, float]) -> float:
        return sum(
            (probability - truth) ** 2
            for probability, truth in zip(
                distribution,
                target,
                strict=True,
            )
        )

    prior_log_loss = log_loss(prior)
    search_log_loss = log_loss(search)
    prior_brier_loss = brier_loss(prior)
    search_brier_loss = brier_loss(search)
    return ProperScoreComparison(
        oracle_outcome=oracle_outcome,
        target_distribution=target,
        prior_distribution=prior,
        search_distribution=search,
        prior_log_loss=prior_log_loss,
        search_log_loss=search_log_loss,
        log_score_gain=prior_log_loss - search_log_loss,
        prior_brier_loss=prior_brier_loss,
        search_brier_loss=search_brier_loss,
        brier_score_gain=prior_brier_loss - search_brier_loss,
    )


def compare_policies_against_oracle(
    *,
    prior_policy: Sequence[float],
    search_policy: Sequence[float],
    result: HexOracleResult,
) -> OraclePolicyComparison:
    """Compare prior and search using exact action regret and proper scores."""

    prior = assess_policy_against_oracle(prior_policy, result)
    search = assess_policy_against_oracle(search_policy, result)
    proper_scores = compare_binary_outcome_probabilities(
        oracle_outcome=result.outcome,
        prior_win_probability=prior.induced_win_probability,
        search_win_probability=search.induced_win_probability,
    )
    return OraclePolicyComparison(
        prior=prior,
        search=search,
        regret_reduction=prior.regret - search.regret,
        proper_scores=proper_scores,
    )


def _jensen_shannon_divergence(
    lhs: Sequence[float],
    rhs: Sequence[float],
) -> float:
    midpoint = tuple(
        0.5 * (left + right)
        for left, right in zip(lhs, rhs, strict=True)
    )

    def kl(distribution: Sequence[float]) -> float:
        return sum(
            probability * math.log(probability / middle)
            for probability, middle in zip(
                distribution,
                midpoint,
                strict=True,
            )
            if probability > 0.0
        )

    return 0.5 * (kl(lhs) + kl(rhs))


def assess_policy_readout_noise(
    *,
    prior_policy: Sequence[float],
    search_policy_readouts: Sequence[Sequence[float]],
    result: HexOracleResult,
) -> PolicyReadoutNoiseAssessment:
    """Measure fixed-tree search signal and finite readout noise.

    At least two readouts are required so the pairwise term is identifiable.
    Policies are masked and renormalized over the oracle's legal actions before
    comparison, which also safely ignores PGX's disabled swap action.
    """

    if len(search_policy_readouts) < 2:
        raise ValueError("at least two search policy readouts are required")

    prior = tuple(
        probability
        for _, probability in _normalised_legal_policy(prior_policy, result)
    )
    readouts = tuple(
        tuple(
            probability
            for _, probability in _normalised_legal_policy(policy, result)
        )
        for policy in search_policy_readouts
    )
    mean_search = tuple(
        statistics.fmean(readout[action] for readout in readouts)
        for action in range(len(prior))
    )

    def squared_l2(lhs: Sequence[float], rhs: Sequence[float]) -> float:
        return sum(
            (left - right) ** 2
            for left, right in zip(lhs, rhs, strict=True)
        )

    prior_readout_js = tuple(
        _jensen_shannon_divergence(prior, readout)
        for readout in readouts
    )
    prior_readout_l2 = tuple(
        squared_l2(prior, readout)
        for readout in readouts
    )
    pairwise_js: list[float] = []
    pairwise_l2: list[float] = []
    for left_index, left in enumerate(readouts):
        for right in readouts[left_index + 1 :]:
            pairwise_js.append(_jensen_shannon_divergence(left, right))
            pairwise_l2.append(squared_l2(left, right))

    mean_prior_l2 = statistics.fmean(prior_readout_l2)
    readout_noise_l2 = 0.5 * statistics.fmean(pairwise_l2)
    return PolicyReadoutNoiseAssessment(
        readout_count=len(readouts),
        prior_to_mean_search_js_nats=_jensen_shannon_divergence(
            prior,
            mean_search,
        ),
        mean_prior_to_search_readout_js_nats=statistics.fmean(
            prior_readout_js
        ),
        mean_pairwise_search_readout_js_nats=statistics.fmean(pairwise_js),
        mean_prior_search_readout_squared_l2=mean_prior_l2,
        readout_noise_squared_l2=readout_noise_l2,
        noise_corrected_displacement_squared_l2=(
            mean_prior_l2 - readout_noise_l2
        ),
    )


def sample_late_game_hex_corpus(
    *,
    count: int,
    size: int = 6,
    min_empty: int = 1,
    max_empty: int = 6,
    seed: int = 0,
    balanced_outcomes: bool = True,
    max_attempts: int | None = None,
) -> FrozenHexCorpus:
    """Sample, solve, and freeze legal non-terminal late-game positions.

    A candidate is made by assigning the correct alternating stone counts and
    rejecting boards on which either colour already connects.  Connectivity
    is monotone, so every accepted board is reachable by some ordering of its
    stones without an earlier terminal state.
    """

    if count < 1:
        raise ValueError(f"count must be positive; got {count}")
    if not 1 <= min_empty <= max_empty <= size * size:
        raise ValueError(
            "empty range must satisfy "
            f"1 <= min_empty <= max_empty <= {size * size}"
        )
    if max_attempts is None:
        max_attempts = max(10_000, count * 10_000)
    if max_attempts < count:
        raise ValueError("max_attempts must be at least count")

    rng = random.Random(seed)
    accepted: list[FrozenHexPosition] = []
    seen: set[str] = set()
    if balanced_outcomes:
        quotas = {
            -1: count // 2,
            1: count - (count // 2),
        }
    else:
        quotas = {-1: count, 1: count}

    attempts = 0
    while len(accepted) < count and attempts < max_attempts:
        attempts += 1
        empty_count = rng.randint(min_empty, max_empty)
        occupied_count = size * size - empty_count
        color_0_count = (occupied_count + 1) // 2
        color_1_count = occupied_count // 2
        indices = list(range(size * size))
        rng.shuffle(indices)
        cells = [EMPTY] * (size * size)
        for action in indices[:color_0_count]:
            cells[action] = COLOR_0
        for action in indices[
            color_0_count : color_0_count + color_1_count
        ]:
            cells[action] = COLOR_1
        cell_tuple = tuple(cells)
        if hex_has_connection(cell_tuple, size=size, color=0):
            continue
        if hex_has_connection(cell_tuple, size=size, color=1):
            continue

        position = HexPosition(
            size=size,
            cells=cell_tuple,
            current_color=occupied_count % 2,
        )
        if position.position_id in seen:
            continue
        frozen = FrozenHexPosition.from_position(position)
        if frozen.oracle_outcome not in (-1, 1):
            continue
        if balanced_outcomes:
            outcome_count = sum(
                existing.oracle_outcome == frozen.oracle_outcome
                for existing in accepted
            )
            if outcome_count >= quotas[frozen.oracle_outcome]:
                continue
        seen.add(position.position_id)
        accepted.append(frozen)

    if len(accepted) != count:
        raise RuntimeError(
            f"accepted only {len(accepted)} of {count} positions after "
            f"{attempts} attempts; widen the empty range or max_attempts"
        )
    return FrozenHexCorpus(
        size=size,
        seed=seed,
        min_empty=min_empty,
        max_empty=max_empty,
        balanced_outcomes=balanced_outcomes,
        attempts=attempts,
        positions=tuple(accepted),
    )


def write_frozen_hex_corpus(
    path: str | Path,
    corpus: FrozenHexCorpus,
) -> None:
    """Write a corpus as deterministic, reviewable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(corpus)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_frozen_hex_corpus(
    path: str | Path,
    *,
    verify: bool = True,
) -> FrozenHexCorpus:
    """Load a corpus, optionally recomputing every oracle label."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FROZEN_CORPUS_FORMAT:
        raise ValueError(
            f"unsupported corpus format: {payload.get('format')!r}"
        )
    positions = tuple(
        FrozenHexPosition(
            position_id=str(item["position_id"]),
            size=int(item["size"]),
            cells=tuple(int(value) for value in item["cells"]),
            current_color=int(item["current_color"]),
            oracle_outcome=int(item["oracle_outcome"]),
            optimal_actions=tuple(
                int(value) for value in item["optimal_actions"]
            ),
            action_values=tuple(
                (int(action), int(value))
                for action, value in item["action_values"]
            ),
        )
        for item in payload["positions"]
    )
    corpus = FrozenHexCorpus(
        format=str(payload["format"]),
        size=int(payload["size"]),
        seed=int(payload["seed"]),
        min_empty=int(payload["min_empty"]),
        max_empty=int(payload["max_empty"]),
        balanced_outcomes=bool(payload["balanced_outcomes"]),
        attempts=int(payload["attempts"]),
        positions=positions,
    )
    if verify:
        for frozen in corpus.positions:
            if frozen.position.position_id != frozen.position_id:
                raise ValueError(
                    f"position id mismatch for {frozen.position_id}"
                )
            recomputed = solve_hex(frozen.position)
            if recomputed != frozen.result:
                raise ValueError(
                    f"oracle labels disagree for {frozen.position_id}"
                )
    return corpus
