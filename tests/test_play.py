from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import pgx
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from scacchi import dirichlet_mctx
from scacchi.dirichlet_mctx.action_selection import sample_dirichlet
from scacchi.dirichlet_mctx.native_targets import (
    TARGET_CATEGORICAL,
    TARGET_PAD,
)
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME, outcome_utility
from scacchi.dirichlet_q_search import posterior_sample_action
from scacchi.distributed import BatchParallel, assert_batch_axis_sharded
from scacchi.network import BoardlawNet
from scacchi.play import TrainingSamples, play, make_selfplay
from scacchi.play_search import (
    EvaluatorOutput,
    PlayerOutput,
    PosteriorPrediction,
    PosteriorTargets,
    TargetMetadata,
    _dirichlet_commitment_policy,
    _dirichlet_root_policy_readout,
    _normalize_policy_on_support,
    commit_action,
    make_action_committer,
    make_search,
    make_search_player,
)
from scacchi.types import (
    ActionCommitmentConfig,
    ActionCommitmentType,
    Config,
    DirichletThompsonSearchConfig,
    EnvConfig,
    GumbelSearchConfig,
    ModelConfig,
    MonteCarloPosteriorUpdateConfig,
    Network,
    NumericalPosteriorUpdateConfig,
    PolicySearchConfig,
    PosteriorUpdateConfig,
    PosteriorUpdateKind,
    QActionSet,
    QPairReduction,
    QSupervisionConfig,
    RootPolicySupport,
    SearchConfig,
    SearchKind,
    SelfplayConfig,
)


_COLLECTIVE_HLO_NAMES = (
    "all-gather",
    "all_gather",
    "all-reduce",
    "all_reduce",
    "all-to-all",
    "all_to_all",
    "collective-permute",
    "collective_permute",
    "reduce-scatter",
    "reduce_scatter",
)

_POLICY_Q_SUPERVISION = QSupervisionConfig(
    action_set=QActionSet.positive_posterior_policy_or_solved,
    reduction=QPairReduction.mean_over_selected_state_action_pairs,
)


class _ToySearchState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    rewards: jax.Array
    terminated: jax.Array


class _ToySearchEnv:
    def step(self, state: _ToySearchState, action: jax.Array) -> _ToySearchState:
        del action
        return _ToySearchState(
            observation=state.observation + 1.0,
            legal_action_mask=state.legal_action_mask,
            current_player=state.current_player,
            rewards=state.rewards,
            terminated=state.terminated,
        )


class _TerminalToySearchEnv:
    def step(self, state: _ToySearchState, action: jax.Array) -> _ToySearchState:
        del action
        actor = state.current_player
        rewards = 2.0 * jax.nn.one_hot(actor, 2, dtype=state.observation.dtype) - 1.0
        return _ToySearchState(
            observation=state.observation + 1.0,
            legal_action_mask=jnp.zeros_like(state.legal_action_mask),
            current_player=1 - actor,
            rewards=rewards,
            terminated=jnp.asarray(True),
        )


def _toy_scalar_evaluator(obs: jax.Array) -> EvaluatorOutput:
    return EvaluatorOutput(
        logits=jnp.zeros((obs.shape[0], 3), dtype=obs.dtype),
        value=jnp.zeros((obs.shape[0],), dtype=obs.dtype),
    )


def _toy_dirichlet_evaluator(obs: jax.Array) -> EvaluatorOutput:
    batch_size = obs.shape[0]
    alpha_v = jnp.broadcast_to(
        jnp.array([2.0, 3.0], dtype=obs.dtype),
        (batch_size, 2),
    )
    alpha_q = jnp.broadcast_to(
        jnp.array(
            [
                [3.0, 1.0],
                [1.0, 4.0],
                [2.0, 2.0],
            ],
            dtype=obs.dtype,
        ),
        (batch_size, 3, 2),
    )
    return EvaluatorOutput(
        logits=jnp.zeros((batch_size, 3), dtype=obs.dtype),
        alpha_v=alpha_v,
        alpha_q=alpha_q,
    )


def _toy_dirichlet_state() -> _ToySearchState:
    return _ToySearchState(
        observation=jnp.zeros((2, 1), dtype=jnp.float32),
        legal_action_mask=jnp.array(
            [
                [True, True, False],
                [False, True, True],
            ]
        ),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        rewards=jnp.zeros((2, 2), dtype=jnp.float32),
        terminated=jnp.zeros((2,), dtype=jnp.bool_),
    )


def test_scalar_gumbel_search_preserves_batch_sharding_without_collectives():
    device_count = jax.device_count()
    mesh = jax.make_mesh(
        (device_count,),
        ("batch",),
        axis_types=(AxisType.Auto,),
    )
    parallel = BatchParallel(enabled=True, mesh=mesh)
    batch_size = max(device_count * 2, 2)
    num_actions = 3
    matrix_sharding = NamedSharding(mesh, PartitionSpec("batch", None))
    vector_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    env_state = _ToySearchState(
        observation=jax.device_put(
            jnp.zeros((batch_size, 1), dtype=jnp.float32),
            matrix_sharding,
        ),
        legal_action_mask=jax.device_put(
            jnp.ones((batch_size, num_actions), dtype=jnp.bool_),
            matrix_sharding,
        ),
        current_player=jax.device_put(
            jnp.zeros((batch_size,), dtype=jnp.int32),
            vector_sharding,
        ),
        rewards=jax.device_put(
            jnp.zeros((batch_size, 2), dtype=jnp.float32),
            matrix_sharding,
        ),
        terminated=jax.device_put(
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            vector_sharding,
        ),
    )
    search = make_search(
        _ToySearchEnv(),
        _toy_scalar_evaluator,
        SearchConfig(
            kind=SearchKind.gumbel,
            gumbel=GumbelSearchConfig(num_simulations=2),
        ),
    )

    def run(env_state, rng_key):
        return search(root_state=env_state, rng_key=rng_key)

    rng_key = jax.random.PRNGKey(0)
    lowered = jax.jit(run).lower(env_state, rng_key)
    compiler_ir = lowered.compiler_ir(dialect="hlo")
    assert compiler_ir is not None
    hlo_text = compiler_ir.as_hlo_text().lower()
    for collective in _COLLECTIVE_HLO_NAMES:
        assert collective not in hlo_text

    output = jax.jit(run)(env_state, rng_key)
    output = assert_batch_axis_sharded(output, parallel, batch_axis=0, label="scalar gumbel output")
    metadata = output.metadata
    assert metadata is not None
    assert metadata.search_action is not None
    assert output.prediction.policy.shape == (batch_size, num_actions)
    assert metadata.search_action.shape == (batch_size,)


def test_scalar_gumbel_search_rejects_dirichlet_outputs():
    env_state = _toy_dirichlet_state()
    search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        SearchConfig(
            kind=SearchKind.gumbel,
            gumbel=GumbelSearchConfig(num_simulations=2),
        ),
    )

    with pytest.raises(ValueError, match="scalar policy/value models only"):
        search(env_state, jax.random.PRNGKey(5))


def test_dirichlet_thompson_prior_only_search_preserves_rng_and_targets():
    env_state = _toy_dirichlet_state()
    search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        SearchConfig(
            kind=SearchKind.dirichlet_thompson,
            dirichlet_thompson=DirichletThompsonSearchConfig(
                num_simulations=0,
                posterior_update=PosteriorUpdateConfig(
                    monte_carlo=MonteCarloPosteriorUpdateConfig(
                        policy_samples=1,
                    ),
                ),
            ),
        ),
        _POLICY_Q_SUPERVISION,
    )
    rng_key = jax.random.PRNGKey(17)

    output = search(env_state, rng_key)

    root_prediction = _toy_dirichlet_evaluator(env_state.observation)
    assert root_prediction.alpha_q is not None
    _, policy_key = jax.random.split(rng_key)
    sampled_outcome = sample_dirichlet(
        policy_key,
        root_prediction.alpha_q,
    )
    scores = jnp.where(
        env_state.legal_action_mask,
        outcome_utility(sampled_outcome),
        -jnp.inf,
    )
    expected_action = jnp.argmax(scores, axis=-1).astype(jnp.int32)
    expected_policy = jax.nn.one_hot(
        expected_action,
        env_state.legal_action_mask.shape[-1],
        dtype=root_prediction.alpha_q.dtype,
    )

    prediction = output.prediction
    metadata = output.metadata
    assert metadata is not None
    assert jnp.array_equal(prediction.policy, expected_policy)
    assert jnp.array_equal(prediction.alpha_q, root_prediction.alpha_q)
    assert jnp.array_equal(prediction.alpha_v, root_prediction.alpha_v)
    assert metadata.q_supervision is not None
    assert jnp.array_equal(
        metadata.q_supervision.selected,
        expected_policy.astype(jnp.bool_),
    )
    assert jnp.array_equal(
        metadata.q_supervision.pair_weight,
        expected_policy,
    )
    assert jnp.array_equal(metadata.search_action, expected_action)
    assert jnp.array_equal(metadata.mask, jnp.ones((2,), dtype=jnp.bool_))


def test_numerical_update_drives_q21_root_target():
    env_state = _toy_dirichlet_state()
    native_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            posterior_update=PosteriorUpdateConfig(
                monte_carlo=MonteCarloPosteriorUpdateConfig(
                    policy_samples=8,
                    policy_sample_chunk_size=2,
                ),
            ),
        ),
    )
    numerical_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            posterior_update=PosteriorUpdateConfig(
                kind=PosteriorUpdateKind.numerical,
                monte_carlo=MonteCarloPosteriorUpdateConfig(
                    policy_samples=8,
                    policy_sample_chunk_size=2,
                ),
                numerical=NumericalPosteriorUpdateConfig(
                    fallback_policy_samples=8,
                    fallback_policy_sample_chunk_size=2,
                ),
            ),
        ),
    )
    key = jax.random.PRNGKey(91)
    native = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        native_config,
        q_supervision_config=_POLICY_Q_SUPERVISION,
    )(env_state, key)
    numerical = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        numerical_config,
        q_supervision_config=_POLICY_Q_SUPERVISION,
    )(env_state, key)

    native_metadata = native.metadata
    numerical_metadata = numerical.metadata
    assert native_metadata is not None
    assert numerical_metadata is not None
    assert native_metadata.q_supervision is not None
    assert numerical_metadata.q_supervision is not None
    assert not jnp.array_equal(
        numerical.prediction.policy,
        native.prediction.policy,
    )
    assert jnp.array_equal(
        native_metadata.q_supervision.selected,
        native.prediction.policy > 0,
    )
    assert jnp.array_equal(
        numerical_metadata.q_supervision.selected,
        numerical.prediction.policy > 0,
    )
    assert jnp.array_equal(
        numerical_metadata.q_supervision.pair_weight,
        numerical_metadata.q_supervision.selected.astype(
            numerical.prediction.policy.dtype
        ),
    )


def test_search_player_composes_q21_action_with_cubic_sampling():
    env_state = _toy_dirichlet_state()
    search_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            posterior_update=PosteriorUpdateConfig(
                kind=PosteriorUpdateKind.numerical,
                monte_carlo=MonteCarloPosteriorUpdateConfig(
                    policy_samples=8,
                    policy_sample_chunk_size=2,
                ),
                numerical=NumericalPosteriorUpdateConfig(
                    fallback_policy_samples=8,
                    fallback_policy_sample_chunk_size=2,
                ),
            ),
        ),
    )
    player_key = jax.random.PRNGKey(911)
    search_key, action_key = jax.random.split(player_key)
    search_output = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        search_config,
    )(env_state, search_key)
    policy_key, sample_key = jax.random.split(action_key)
    commitment_policy = _dirichlet_commitment_policy(
        search_output,
        env_state.legal_action_mask,
        policy_key,
        search_config.dirichlet_thompson,
        PosteriorUpdateKind.numerical,
    )
    expected = posterior_sample_action(
        sample_key,
        commitment_policy,
        env_state.legal_action_mask,
        temperature=1.0 / 3.0,
    )

    output = make_search_player(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        search_config,
        ActionCommitmentConfig(
            kind=ActionCommitmentType.posterior_sample,
            posterior_update=PosteriorUpdateKind.numerical,
            posterior_sample_temperature=1.0 / 3.0,
        ),
    )(env_state, player_key)

    assert jnp.array_equal(output.action, expected)


def test_action_commitment_can_select_a_different_posterior_update():
    env_state = _toy_dirichlet_state()
    search_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            posterior_update=PosteriorUpdateConfig(
                kind=PosteriorUpdateKind.numerical,
                monte_carlo=MonteCarloPosteriorUpdateConfig(
                    policy_samples=8,
                    policy_sample_chunk_size=2,
                ),
                numerical=NumericalPosteriorUpdateConfig(
                    fallback_policy_samples=8,
                    fallback_policy_sample_chunk_size=2,
                ),
            ),
        ),
    )
    commitment = ActionCommitmentConfig(
        kind=ActionCommitmentType.posterior_sample,
        posterior_update=PosteriorUpdateKind.monte_carlo,
    )
    player_key = jax.random.PRNGKey(913)
    search_key, action_key = jax.random.split(player_key)
    search_output = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        search_config,
    )(env_state, search_key)
    policy_key, sample_key = jax.random.split(action_key)
    expected_policy = _dirichlet_commitment_policy(
        search_output,
        env_state.legal_action_mask,
        policy_key,
        search_config.dirichlet_thompson,
        PosteriorUpdateKind.monte_carlo,
    )
    expected_action = posterior_sample_action(
        sample_key,
        expected_policy,
        env_state.legal_action_mask,
    )

    output = make_search_player(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        search_config,
        commitment,
    )(env_state, player_key)

    assert jnp.array_equal(output.action, expected_action)


def test_q21_solved_target_is_uniform_but_commitment_stays_native():
    class Summary(NamedTuple):
        alpha: jax.Array
        q_categorical_outcome: jax.Array
        q_categorical_distance: jax.Array
        v_categorical_outcome: jax.Array

    native_policy = jnp.asarray([[0.0, 1.0, 0.0]], dtype=jnp.float32)
    readout = _dirichlet_root_policy_readout(
        native_policy,
        summary=Summary(
            alpha=jnp.ones((1, 3, 2), dtype=jnp.float32),
            q_categorical_outcome=jnp.asarray([[1, 1, 0]], dtype=jnp.int8),
            q_categorical_distance=jnp.asarray([[2, 2, 5]], dtype=jnp.int32),
            v_categorical_outcome=jnp.asarray([1], dtype=jnp.int8),
        ),
        legal_action_mask=jnp.asarray([[True, True, True]]),
        search_cfg=DirichletThompsonSearchConfig(
            posterior_update=PosteriorUpdateConfig(
                kind=PosteriorUpdateKind.numerical,
            ),
        ),
    )

    assert jnp.array_equal(
        readout,
        jnp.asarray([[0.5, 0.5, 0.0]], dtype=jnp.float32),
    )


def test_replay_policy_is_masked_to_search_evidence_and_sharpened():
    class Summary(NamedTuple):
        alpha: jax.Array
        q_categorical_outcome: jax.Array
        q_categorical_distance: jax.Array
        v_categorical_outcome: jax.Array
        visit_counts: jax.Array

    native_policy = jnp.asarray([[0.2, 0.3, 0.5]], dtype=jnp.float32)
    readout = _dirichlet_root_policy_readout(
        native_policy,
        summary=Summary(
            alpha=jnp.ones((1, 3, 2), dtype=jnp.float32),
            q_categorical_outcome=jnp.full(
                (1, 3),
                int(NO_OUTCOME),
                dtype=jnp.int8,
            ),
            q_categorical_distance=jnp.zeros((1, 3), dtype=jnp.int32),
            v_categorical_outcome=jnp.asarray(
                [int(NO_OUTCOME)],
                dtype=jnp.int8,
            ),
            visit_counts=jnp.asarray([[1.0, 1.0, 0.0]]),
        ),
        legal_action_mask=jnp.ones((1, 3), dtype=jnp.bool_),
        search_cfg=DirichletThompsonSearchConfig(
            root_policy_support=RootPolicySupport.search_evidence,
            policy_target_temperature=0.5,
            posterior_update=PosteriorUpdateConfig(
                kind=PosteriorUpdateKind.numerical,
            ),
        ),
    )
    estimate = (
        dirichlet_mctx.binary_posterior_best_policy_prefix_quadrature(
            jnp.ones((1, 3, 2), dtype=jnp.float32),
            jnp.asarray([[False, False, True]]),
            jnp.full((1, 3), int(NO_OUTCOME), dtype=jnp.int8),
        ).policy
    )
    expected = _normalize_policy_on_support(
        estimate,
        jnp.asarray([[True, True, False]]),
        temperature=0.5,
    )

    assert jnp.allclose(readout, expected)
    assert readout[0, 2] == 0.0


def test_numerical_commitment_stays_on_search_support():
    batch_size = 64
    no_outcome = int(NO_OUTCOME)
    support = jnp.broadcast_to(
        jnp.asarray([True, False, True]),
        (batch_size, 3),
    )
    alpha_q = jnp.broadcast_to(
        jnp.asarray(
            [[2.0, 5.0], [1.0, 20.0], [4.0, 2.0]],
            dtype=jnp.float32,
        ),
        (batch_size, 3, 2),
    )
    categorical_outcome = jnp.full(
        (batch_size, 3),
        no_outcome,
        dtype=jnp.int8,
    )
    posterior = PosteriorTargets(
        prediction=PosteriorPrediction(
            policy=jnp.full(
                (batch_size, 3),
                1.0 / 3.0,
                dtype=jnp.float32,
            ),
            alpha_q=alpha_q,
        ),
        metadata=TargetMetadata(
            q_positive_evidence_action=support,
            search_action=jnp.zeros((batch_size,), dtype=jnp.int32),
            q_target_outcome=categorical_outcome,
            v_target_outcome=jnp.full(
                (batch_size,),
                no_outcome,
                dtype=jnp.int8,
            ),
        ),
    )
    search_config = DirichletThompsonSearchConfig(
        root_policy_support=RootPolicySupport.search_evidence,
        posterior_update=PosteriorUpdateConfig(
            kind=PosteriorUpdateKind.numerical,
            numerical=NumericalPosteriorUpdateConfig(
                fallback_policy_samples=8,
                fallback_policy_sample_chunk_size=2,
            ),
        ),
    )
    legal = jnp.ones((batch_size, 3), dtype=jnp.bool_)
    commitment_policy = _dirichlet_commitment_policy(
        posterior,
        legal,
        jax.random.PRNGKey(947),
        search_config,
        PosteriorUpdateKind.numerical,
    )
    action = make_action_committer(
        ActionCommitmentConfig(
            kind=ActionCommitmentType.posterior_sample,
            posterior_update=PosteriorUpdateKind.numerical,
        ),
        search_config,
    )(
        posterior,
        legal,
        jax.random.PRNGKey(948),
    )

    assert jnp.all(commitment_policy[~support] == 0.0)
    assert jnp.allclose(jnp.sum(commitment_policy, axis=-1), 1.0)
    selected_is_supported = jnp.take_along_axis(
        support,
        action[:, None],
        axis=-1,
    )
    assert bool(jnp.all(selected_is_supported))


def test_search_player_preserves_solved_native_action_under_temperature():
    env_state = _toy_dirichlet_state()
    search_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=4,
            posterior_update=PosteriorUpdateConfig(
                kind=PosteriorUpdateKind.numerical,
                numerical=NumericalPosteriorUpdateConfig(
                    fallback_policy_samples=4,
                ),
            ),
        ),
    )
    player_key = jax.random.PRNGKey(912)
    search_key, _ = jax.random.split(player_key)
    search_output = make_search(
        _TerminalToySearchEnv(),
        _toy_dirichlet_evaluator,
        search_config,
    )(env_state, search_key)
    metadata = search_output.metadata
    assert metadata is not None
    assert metadata.search_action is not None

    output = make_search_player(
        _TerminalToySearchEnv(),
        _toy_dirichlet_evaluator,
        search_config,
        ActionCommitmentConfig(
            kind=ActionCommitmentType.posterior_sample,
            posterior_update=PosteriorUpdateKind.numerical,
            posterior_sample_temperature=8.0,
        ),
    )(env_state, player_key)

    assert jnp.array_equal(output.action, metadata.search_action)


def test_unsafe_q21_action_update_uses_its_monte_carlo_fallback():
    env_state = _toy_dirichlet_state()

    def extreme_evaluator(obs: jax.Array) -> EvaluatorOutput:
        batch_size = obs.shape[0]
        return EvaluatorOutput(
            logits=jnp.zeros((batch_size, 3), dtype=obs.dtype),
            alpha_v=jnp.ones((batch_size, 2), dtype=obs.dtype),
            alpha_q=jnp.broadcast_to(
                jnp.asarray(
                    [
                        [1e-8, 1.0],
                        [1.0, 1e-8],
                        [1.0, 1.0],
                    ],
                    dtype=obs.dtype,
                ),
                (batch_size, 3, 2),
            ),
        )

    search_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            posterior_update=PosteriorUpdateConfig(
                kind=PosteriorUpdateKind.numerical,
                monte_carlo=MonteCarloPosteriorUpdateConfig(
                    policy_samples=8,
                    policy_sample_chunk_size=2,
                ),
                numerical=NumericalPosteriorUpdateConfig(
                    fallback_policy_samples=8,
                    fallback_policy_sample_chunk_size=2,
                ),
            ),
        ),
    )
    output = make_search(
        _ToySearchEnv(),
        extreme_evaluator,
        search_config,
    )(env_state, jax.random.PRNGKey(92))
    metadata = output.metadata
    assert metadata is not None
    assert metadata.search_action is not None
    key = jax.random.PRNGKey(123)
    numerical = _dirichlet_commitment_policy(
        output,
        env_state.legal_action_mask,
        key,
        search_config.dirichlet_thompson,
        PosteriorUpdateKind.numerical,
    )
    native = _dirichlet_commitment_policy(
        output,
        env_state.legal_action_mask,
        key,
        search_config.dirichlet_thompson,
        PosteriorUpdateKind.monte_carlo,
    )
    assert jnp.array_equal(numerical, native)


def test_one_unsafe_q21_root_lane_does_not_change_safe_lane():
    class Summary(NamedTuple):
        alpha: jax.Array
        q_categorical_outcome: jax.Array
        q_categorical_distance: jax.Array
        v_categorical_outcome: jax.Array

    native_policy = jnp.asarray(
        [[0.25, 0.50, 0.25], [0.50, 0.25, 0.25]],
        dtype=jnp.float32,
    )
    no_outcome = int(NO_OUTCOME)
    alpha = jnp.asarray(
        [
            [[2.0, 3.0], [4.0, 1.0], [1.0, 2.0]],
            [[1e-5, 1.0], [2.0, 1.0], [1.0, 3.0]],
        ],
        dtype=jnp.float32,
    )
    categorical_outcome = jnp.full(
        (2, 3),
        no_outcome,
        dtype=jnp.int8,
    )
    readout = _dirichlet_root_policy_readout(
        native_policy,
        summary=Summary(
            alpha=alpha,
            q_categorical_outcome=categorical_outcome,
            q_categorical_distance=jnp.zeros(
                (2, 3),
                dtype=jnp.int32,
            ),
            v_categorical_outcome=jnp.full(
                (2,),
                no_outcome,
                dtype=jnp.int8,
            ),
        ),
        legal_action_mask=jnp.ones((2, 3), dtype=jnp.bool_),
        search_cfg=DirichletThompsonSearchConfig(
            posterior_update=PosteriorUpdateConfig(
                kind=PosteriorUpdateKind.numerical,
            ),
        ),
    )
    estimate = (
        dirichlet_mctx.binary_posterior_best_policy_prefix_quadrature(
            alpha,
            jnp.zeros((2, 3), dtype=jnp.bool_),
            categorical_outcome,
        )
    )

    assert jnp.array_equal(
        estimate.tail_range_clipped,
        jnp.asarray([False, True]),
    )
    assert jnp.allclose(readout[0], estimate.policy[0], atol=1e-6)
    assert jnp.array_equal(readout[1], native_policy[1])


@pytest.mark.parametrize(
    "posterior_update_kind",
    [
        PosteriorUpdateKind.monte_carlo,
        PosteriorUpdateKind.numerical,
    ],
)
def test_dirichlet_thompson_tree_search_builds_legal_targets(
    posterior_update_kind: PosteriorUpdateKind,
):
    env_state = _toy_dirichlet_state()
    search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        SearchConfig(
            kind=SearchKind.dirichlet_thompson,
            dirichlet_thompson=DirichletThompsonSearchConfig(
                num_simulations=4,
                max_depth=2,
                posterior_update=PosteriorUpdateConfig(
                    kind=posterior_update_kind,
                    monte_carlo=MonteCarloPosteriorUpdateConfig(
                        policy_samples=1,
                        policy_sample_chunk_size=2,
                    ),
                    numerical=NumericalPosteriorUpdateConfig(
                        fallback_policy_samples=1,
                        fallback_policy_sample_chunk_size=2,
                    ),
                ),
            ),
        ),
    )

    output = search(env_state, jax.random.PRNGKey(23))

    prediction = output.prediction
    metadata = output.metadata
    assert metadata is not None
    assert prediction.alpha_v is not None
    assert prediction.alpha_q is not None
    assert metadata.q_supervision is not None
    assert metadata.search_action is not None
    assert metadata.mask is not None
    assert prediction.policy.shape == env_state.legal_action_mask.shape
    assert prediction.alpha_v.shape == (2, 2)
    assert prediction.alpha_q.shape == (2, 3, 2)
    assert jnp.allclose(prediction.policy.sum(axis=-1), 1.0)
    assert jnp.all(prediction.policy[~env_state.legal_action_mask] == 0.0)
    assert metadata.q_positive_evidence_action is not None
    assert metadata.q_target_kind is not None
    expected_selected = metadata.q_positive_evidence_action | (
        metadata.q_target_kind == int(TARGET_CATEGORICAL)
    )
    assert jnp.array_equal(
        metadata.q_supervision.selected,
        expected_selected,
    )
    assert jnp.array_equal(
        metadata.q_supervision.pair_weight,
        expected_selected.astype(prediction.policy.dtype),
    )
    assert bool(metadata.mask.all())
    selected_is_legal = jnp.take_along_axis(
        env_state.legal_action_mask,
        metadata.search_action[:, None],
        axis=-1,
    )
    assert bool(selected_is_legal.all())


def test_dirichlet_thompson_exports_categorical_root_and_q_targets():
    env_state = _toy_dirichlet_state()
    search = make_search(
        _TerminalToySearchEnv(),
        _toy_dirichlet_evaluator,
        SearchConfig(
            kind=SearchKind.dirichlet_thompson,
            dirichlet_thompson=DirichletThompsonSearchConfig(
                num_simulations=4,
            ),
        ),
    )

    output = search(env_state, jax.random.PRNGKey(29))
    prediction = output.prediction
    metadata = output.metadata
    assert metadata is not None
    assert metadata.search_action is not None
    assert metadata.q_target_kind is not None
    assert metadata.q_target_outcome is not None
    assert metadata.q_target_distance is not None
    assert metadata.v_target_kind is not None
    assert metadata.v_target_outcome is not None
    assert metadata.v_target_distance is not None

    batch = jnp.arange(env_state.current_player.shape[0])
    action = metadata.search_action
    assert jnp.array_equal(
        prediction.policy,
        jax.nn.one_hot(action, prediction.policy.shape[-1]),
    )
    assert jnp.all(
        metadata.q_target_kind[batch, action] == int(TARGET_CATEGORICAL)
    )
    assert jnp.all(metadata.q_target_outcome[batch, action] == 1)
    assert jnp.all(metadata.q_target_distance[batch, action] == 1)
    assert jnp.all(metadata.v_target_kind == int(TARGET_CATEGORICAL))
    assert jnp.all(metadata.v_target_outcome == 1)
    assert jnp.all(metadata.v_target_distance == 1)
    assert jnp.all(
        metadata.q_target_kind[~env_state.legal_action_mask] == int(TARGET_PAD)
    )


def test_policy_search_uses_masked_logits_without_tree_search():
    num_actions = 3
    env_state = _ToySearchState(
        observation=jnp.zeros((2, 1), dtype=jnp.float32),
        legal_action_mask=jnp.array(
            [
                [True, True, False],
                [False, True, True],
            ]
        ),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        rewards=jnp.zeros((2, 2), dtype=jnp.float32),
        terminated=jnp.zeros((2,), dtype=jnp.bool_),
    )

    def logits_only_model(obs: jax.Array) -> jax.Array:
        del obs
        return jnp.array(
            [
                [0.0, 2.0, 100.0],
                [10.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )

    player = make_search_player(
        _ToySearchEnv(),
        logits_only_model,
        SearchConfig(
            kind=SearchKind.policy,
            policy=PolicySearchConfig(temperature=1.0),
        ),
        ActionCommitmentConfig(
            kind=ActionCommitmentType.posterior_argmax,
        ),
    )

    output = player(env_state, jax.random.PRNGKey(0))

    assert output.posterior is not None
    policy = output.posterior.prediction.policy
    assert policy.shape == (2, num_actions)
    assert jnp.array_equal(policy > 0.0, env_state.legal_action_mask)
    assert jnp.allclose(policy.sum(axis=-1), 1.0)
    assert jnp.array_equal(output.action, jnp.array([1, 2], dtype=jnp.int32))


def test_policy_search_action_samples_batched_policy_rows():
    batch_size = 64
    num_actions = 3
    env_state = _ToySearchState(
        observation=jnp.zeros((batch_size, 1), dtype=jnp.float32),
        legal_action_mask=jnp.ones((batch_size, num_actions), dtype=jnp.bool_),
        current_player=jnp.zeros((batch_size,), dtype=jnp.int32),
        rewards=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        terminated=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )

    def logits_only_model(obs: jax.Array) -> jax.Array:
        return jnp.zeros((obs.shape[0], num_actions), dtype=jnp.float32)

    key = jax.random.PRNGKey(0)
    player = make_search_player(
        _ToySearchEnv(),
        logits_only_model,
        SearchConfig(
            kind=SearchKind.policy,
            policy=PolicySearchConfig(temperature=1.0),
        ),
        ActionCommitmentConfig(
            kind=ActionCommitmentType.search_action,
        ),
    )

    output = player(env_state, key)
    assert output.posterior is not None
    search_key, _ = jax.random.split(key)
    expected = posterior_sample_action(
        search_key,
        output.posterior.prediction.policy,
        env_state.legal_action_mask,
    )

    assert jnp.array_equal(output.action, expected)
    assert jnp.any(output.action != 0)


def test_commit_action_samples_posterior_target():
    key = jax.random.PRNGKey(0)
    action_weights = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    legal_action_mask = jnp.ones_like(action_weights, dtype=jnp.bool_)
    search_action = jnp.array([0, 0], dtype=jnp.int32)

    played_action = commit_action(
        "posterior_sample",
        key,
        action_weights,
        legal_action_mask,
        search_action,
    )

    expected = posterior_sample_action(key, action_weights, legal_action_mask)
    assert jnp.array_equal(played_action, expected)
    assert not jnp.array_equal(played_action, search_action)


def test_search_player_threads_posterior_sample_temperature():
    batch_size = 128
    num_actions = 3
    env_state = _ToySearchState(
        observation=jnp.zeros((batch_size, 1), dtype=jnp.float32),
        legal_action_mask=jnp.ones(
            (batch_size, num_actions),
            dtype=jnp.bool_,
        ),
        current_player=jnp.zeros((batch_size,), dtype=jnp.int32),
        rewards=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        terminated=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )

    def logits_only_model(obs: jax.Array) -> jax.Array:
        return jnp.broadcast_to(
            jnp.asarray([0.0, 0.2, -0.1], dtype=obs.dtype),
            (obs.shape[0], num_actions),
        )

    search_config = SearchConfig(
        kind=SearchKind.policy,
        policy=PolicySearchConfig(temperature=1.0),
    )
    key = jax.random.PRNGKey(20)
    player = make_search_player(
        _ToySearchEnv(),
        logits_only_model,
        search_config,
        ActionCommitmentConfig(
            kind=ActionCommitmentType.posterior_sample,
            posterior_sample_temperature=0.25,
        ),
    )

    output = player(env_state, key)

    assert output.posterior is not None
    _, action_key = jax.random.split(key)
    expected = posterior_sample_action(
        action_key,
        output.posterior.prediction.policy,
        env_state.legal_action_mask,
        temperature=0.25,
    )
    assert jnp.array_equal(output.action, expected)


def test_commit_action_uses_configured_posterior_sample_temperature():
    key = jax.random.PRNGKey(21)
    policy = jnp.broadcast_to(
        jnp.asarray([0.2, 0.8], dtype=jnp.float32),
        (128, 2),
    )
    legal = jnp.ones_like(policy, dtype=jnp.bool_)

    played_action = commit_action(
        "posterior_sample",
        key,
        policy,
        legal,
        posterior_sample_temperature=0.5,
    )

    expected = posterior_sample_action(
        key,
        policy,
        legal,
        temperature=0.5,
    )
    assert jnp.array_equal(played_action, expected)


@pytest.mark.parametrize(
    "temperature",
    [0.0, -1.0, float("nan"), float("inf"), -float("inf")],
)
def test_action_committer_rejects_invalid_posterior_sample_temperature(
    temperature: float,
):
    with pytest.raises(
        ValueError,
        match="action_commitment.posterior_sample_temperature",
    ):
        make_action_committer(
            ActionCommitmentConfig(
                kind=ActionCommitmentType.posterior_sample,
                posterior_sample_temperature=temperature,
            ),
        )


def test_commit_action_can_use_search_action():
    action_weights = jnp.array([[0.0, 1.0, 0.0]])
    search_action = jnp.array([2], dtype=jnp.int32)

    played_action = commit_action(
        "search_action",
        jax.random.PRNGKey(0),
        action_weights,
        jnp.ones_like(action_weights, dtype=jnp.bool_),
        search_action,
    )

    assert jnp.array_equal(played_action, search_action)


def test_commit_action_rejects_removed_posterior_best_alias():
    with pytest.raises(ValueError, match="action_commitment_type"):
        commit_action(
            "posterior_best",
            jax.random.PRNGKey(0),
            jnp.array([[1.0]]),
            jnp.array([[True]]),
            jnp.array([0], dtype=jnp.int32),
        )


def test_play_dispatches_training_mode():
    env = pgx.make("tic_tac_toe")

    def player(env_state, key: jax.Array) -> PlayerOutput:
        del key
        policy = env_state.legal_action_mask.astype(jnp.float32)
        policy = policy / jnp.sum(policy, axis=-1, keepdims=True)
        return PlayerOutput(
            action=jnp.argmax(env_state.legal_action_mask, axis=-1).astype(jnp.int32),
            posterior=PosteriorTargets(
                prediction=PosteriorPrediction(policy=policy),
                metadata=TargetMetadata(
                    mask=jnp.any(env_state.legal_action_mask, axis=-1),
                ),
            ),
        )

    training = play(
        env,
        player,
        player,
        jax.random.PRNGKey(0),
        mode="training",
        batch_size=2,
        max_num_steps=1,
    )

    assert isinstance(training, TrainingSamples)
    assert training.obs.shape[:2] == (2, 1)
    assert training.posterior.prediction.policy.shape == (2, 1, env.num_actions)


def test_selfplay_clears_auto_reset_transition_marker_before_player_search():
    env = pgx.make("tic_tac_toe")

    def player(env_state, key: jax.Array) -> PlayerOutput:
        del key
        policy = env_state.legal_action_mask.astype(jnp.float32)
        policy = policy / jnp.sum(policy, axis=-1, keepdims=True)
        return PlayerOutput(
            action=jnp.argmax(env_state.legal_action_mask, axis=-1).astype(jnp.int32),
            posterior=PosteriorTargets(
                prediction=PosteriorPrediction(
                    policy=policy,
                    # Surface exactly what a search root receives.  The
                    # transition ending game one is recorded separately in
                    # TrainingSamples.terminated; game two's fresh root must
                    # not still look terminal to its player.
                    value=env_state.terminated.astype(jnp.float32),
                ),
                metadata=TargetMetadata(
                    mask=jnp.any(env_state.legal_action_mask, axis=-1),
                ),
            ),
        )

    training = play(
        env,
        player,
        player,
        jax.random.PRNGKey(7),
        mode="training",
        batch_size=1,
        max_num_steps=8,
    )

    # First-legal Tic-Tac-Toe finishes within the rollout, proving an
    # auto-reset occurred.  Nonetheless every player/search invocation sees a
    # clean, playable root, including the first move of the next game.
    assert bool(training.terminated.any())
    assert training.posterior.prediction.value is not None
    assert not bool(training.posterior.prediction.value.any())


def test_make_selfplay_delegates_to_play_training_smoke():
    env = pgx.make("tic_tac_toe")
    search = SearchConfig(
        kind=SearchKind.gumbel,
        gumbel=GumbelSearchConfig(num_simulations=1),
    )
    config = Config(
        env=EnvConfig(id="tic_tac_toe", num_outcomes=3),
        model=ModelConfig(
            network=Network.boardlaw,
            num_channels=8,
            num_layers=1,
        ),
        selfplay=SelfplayConfig(
            batch_size=2,
            max_num_steps=1,
            search=search,
            action_commitment=ActionCommitmentConfig(
                kind=ActionCommitmentType.posterior_argmax,
            ),
        ),
    )
    model = BoardlawNet(
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        width=8,
        depth=1,
        rngs=nnx.Rngs(0),
    )

    data = make_selfplay(env, config)(model, jax.random.PRNGKey(1))

    assert data.obs.shape[:2] == (2, 1)
    assert data.played_action.shape == (2, 1)
    assert data.legal_action_mask.shape == (2, 1, env.num_actions)
    assert data.posterior.prediction.policy.shape == (2, 1, env.num_actions)


def test_batch_parallel_selfplay_lowers_without_search_collectives():
    device_count = jax.device_count()
    mesh = jax.make_mesh(
        (device_count,),
        ("batch",),
        axis_types=(AxisType.Auto,),
    )
    parallel = BatchParallel(enabled=True, mesh=mesh)
    env = pgx.make("tic_tac_toe")
    batch_size = max(device_count * 2, 2)
    config = Config(
        env=EnvConfig(id="tic_tac_toe", num_outcomes=3),
        model=ModelConfig(
            network=Network.boardlaw,
            num_channels=8,
            num_layers=1,
        ),
        selfplay=SelfplayConfig(
            batch_size=batch_size,
            max_num_steps=1,
            search=SearchConfig(
                kind=SearchKind.gumbel,
                gumbel=GumbelSearchConfig(num_simulations=1),
            ),
            action_commitment=ActionCommitmentConfig(
                kind=ActionCommitmentType.posterior_argmax,
            ),
        ),
    )
    model = BoardlawNet(
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        width=8,
        depth=1,
        rngs=nnx.Rngs(0),
    )
    selfplay = make_selfplay(env, config, parallel=parallel)

    with jax.set_mesh(mesh):
        compiled = selfplay.lower(model, jax.random.PRNGKey(0)).compile()
        hlo_text = compiled.as_text().lower()
        for collective in _COLLECTIVE_HLO_NAMES:
            assert collective not in hlo_text

        data = selfplay(model, jax.random.PRNGKey(1))
        data = assert_batch_axis_sharded(
            data,
            parallel,
            batch_axis=0,
            label="batch-parallel selfplay",
        )

    assert data.obs.shape[:2] == (batch_size, 1)
    assert data.posterior.prediction.policy.shape == (batch_size, 1, env.num_actions)
