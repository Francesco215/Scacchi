from typing import NamedTuple, cast

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
from scacchi.dirichlet_mctx.outcomes import outcome_utility
from scacchi.dirichlet_q_search import (
    PosteriorPluralityResult,
    posterior_best_action,
    posterior_plurality_action,
    posterior_plurality_result,
    posterior_sample_action,
)
from scacchi.distributed import BatchParallel, assert_batch_axis_sharded
from scacchi.network import BoardlawNet
from scacchi.play import TrainingSamples, play, make_selfplay
import scacchi.play_search as play_search_module
from scacchi.play_search import (
    EvaluatorOutput,
    PlayerOutput,
    PosteriorPrediction,
    PosteriorTargets,
    TargetMetadata,
    commit_action,
    make_action_committer,
    make_search,
    make_search_player,
)
from scacchi.search_diagnostics import SearchDiagnostics
from scacchi.types import (
    ActionCommitmentType,
    Config,
    DirichletThompsonSearchConfig,
    EnvConfig,
    GumbelSearchConfig,
    ModelConfig,
    Network,
    PolicySearchConfig,
    PosteriorPolicyEstimator,
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


def _as_pgx_state(state: _ToySearchState) -> pgx.State:
    """Adapt the deliberately minimal test double to the production boundary."""

    return cast(pgx.State, state)


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
    metadata = output.posterior.metadata
    assert metadata is not None
    assert metadata.search_action is not None
    assert output.posterior.prediction.policy.shape == (batch_size, num_actions)
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
                policy_samples=0,
            ),
        ),
    )
    rng_key = jax.random.PRNGKey(17)

    output = search(env_state, rng_key)
    assert output.commitment_policy is None

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

    prediction = output.posterior.prediction
    metadata = output.posterior.metadata
    assert metadata is not None
    assert jnp.array_equal(prediction.policy, expected_policy)
    assert jnp.array_equal(prediction.alpha_q, root_prediction.alpha_q)
    assert jnp.array_equal(prediction.alpha_v, root_prediction.alpha_v)
    assert jnp.array_equal(metadata.q_weight, expected_policy)
    assert jnp.array_equal(metadata.search_action, expected_action)
    assert jnp.array_equal(metadata.mask, jnp.ones((2,), dtype=jnp.bool_))
    diagnostics = output.posterior.diagnostics
    assert diagnostics is not None
    assert jnp.allclose(diagnostics.search_policy_kl_sum, jnp.log(2.0))
    assert jnp.array_equal(
        diagnostics.search_policy_kl_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.allclose(diagnostics.search_v_semantic_kl_sum, 0.0)
    assert jnp.allclose(diagnostics.search_v_dirichlet_kl_sum, 0.0)
    assert jnp.allclose(diagnostics.search_q_semantic_kl_sum, 0.0)
    assert jnp.allclose(diagnostics.search_q_dirichlet_kl_sum, 0.0)
    assert jnp.array_equal(
        diagnostics.search_q_dirichlet_kl_count,
        jnp.full((2,), 2.0),
    )
    assert jnp.array_equal(
        diagnostics.search_legal_action_count,
        jnp.full((2,), 2.0),
    )
    assert jnp.array_equal(
        diagnostics.search_root_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.all(diagnostics.search_expanded_node_count == 0)
    assert jnp.all(diagnostics.search_simulation_active_count == 0)
    assert jnp.all(
        diagnostics.search_executed_simulation_row_count == 0
    )
    assert jnp.all(diagnostics.search_requested_simulation_count == 0)
    assert jnp.all(diagnostics.search_max_depth_sum == 0)


@pytest.mark.parametrize(
    "commitment_type",
    [
        ActionCommitmentType.posterior_argmax,
        ActionCommitmentType.posterior_plurality,
        ActionCommitmentType.posterior_plurality_uniform_ties,
        ActionCommitmentType.posterior_sample,
        ActionCommitmentType.search_action,
    ],
)
def test_prefix_root_target_preserves_native_m32_commitment_and_q_weight(
    commitment_type: ActionCommitmentType,
):
    env_state = _toy_dirichlet_state()
    native_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            policy_samples=8,
            policy_sample_chunk_size=2,
        ),
    )
    target_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            policy_samples=8,
            policy_sample_chunk_size=2,
            root_policy_target_estimator=PosteriorPolicyEstimator.prefix_cdf,
            prefix_cdf_half_width=10,
        ),
    )
    key = jax.random.PRNGKey(91)
    native_search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        native_config,
        q_loss_weight_mode="policy",
    )
    target_search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        target_config,
        q_loss_weight_mode="policy",
    )

    native = native_search(_as_pgx_state(env_state), key)
    target = target_search(_as_pgx_state(env_state), key)
    native_metadata = native.posterior.metadata
    target_metadata = target.posterior.metadata
    assert native_metadata is not None
    assert target_metadata is not None
    assert target.commitment_policy is not None
    target_alpha_v = cast(jax.Array, target.posterior.prediction.alpha_v)
    native_alpha_v = cast(jax.Array, native.posterior.prediction.alpha_v)
    target_alpha_q = cast(jax.Array, target.posterior.prediction.alpha_q)
    native_alpha_q = cast(jax.Array, native.posterior.prediction.alpha_q)
    target_search_action = cast(jax.Array, target_metadata.search_action)
    native_search_action = cast(jax.Array, native_metadata.search_action)
    target_q_weight = cast(jax.Array, target_metadata.q_weight)
    native_q_weight = cast(jax.Array, native_metadata.q_weight)
    assert jnp.array_equal(
        target.commitment_policy,
        native.posterior.prediction.policy,
    )
    assert not jnp.array_equal(
        target.posterior.prediction.policy,
        native.posterior.prediction.policy,
    )
    assert jnp.array_equal(
        target_alpha_v,
        native_alpha_v,
    )
    assert jnp.array_equal(
        target_alpha_q,
        native_alpha_q,
    )
    assert jnp.array_equal(
        target_search_action,
        native_search_action,
    )
    assert jnp.array_equal(target_q_weight, native_q_weight)
    assert jnp.array_equal(target_q_weight, target.commitment_policy)

    native_player = make_search_player(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        native_config,
        commitment_type,
        q_loss_weight_mode="policy",
    )
    target_player = make_search_player(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        target_config,
        commitment_type,
        q_loss_weight_mode="policy",
    )
    assert jnp.array_equal(
        target_player(env_state, key).action,
        native_player(env_state, key).action,
    )

    diagnostics = target.posterior.diagnostics
    assert diagnostics is not None
    assert jnp.array_equal(
        diagnostics.search_root_policy_target_prefix_eligible_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        diagnostics.search_root_policy_target_prefix_accepted_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.all(
        diagnostics.search_root_policy_target_native_l1_sum > 0
    )


@pytest.mark.parametrize(
    "commitment_type",
    [
        ActionCommitmentType.posterior_argmax,
        ActionCommitmentType.posterior_plurality,
        ActionCommitmentType.posterior_plurality_uniform_ties,
        ActionCommitmentType.posterior_sample,
        ActionCommitmentType.search_action,
    ],
)
def test_prefix_root_action_estimator_changes_only_unresolved_commitment(
    commitment_type: ActionCommitmentType,
):
    env_state = _toy_dirichlet_state()
    native_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            policy_samples=8,
            policy_sample_chunk_size=2,
        ),
    )
    action_config = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=0,
            policy_samples=8,
            policy_sample_chunk_size=2,
            root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
            prefix_cdf_half_width=10,
        ),
    )
    player_key = jax.random.PRNGKey(93)
    search_key, action_key = jax.random.split(player_key)
    native_search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        native_config,
        q_loss_weight_mode="policy",
    )
    action_search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        action_config,
        q_loss_weight_mode="policy",
    )

    native = native_search(_as_pgx_state(env_state), search_key)
    action_readout = action_search(_as_pgx_state(env_state), search_key)
    native_metadata = native.posterior.metadata
    action_metadata = action_readout.posterior.metadata
    assert native_metadata is not None
    assert action_metadata is not None
    assert action_readout.commitment_policy is not None
    action_alpha_v = cast(
        jax.Array,
        action_readout.posterior.prediction.alpha_v,
    )
    native_alpha_v = cast(jax.Array, native.posterior.prediction.alpha_v)
    action_alpha_q = cast(
        jax.Array,
        action_readout.posterior.prediction.alpha_q,
    )
    native_alpha_q = cast(jax.Array, native.posterior.prediction.alpha_q)
    action_search_action = cast(jax.Array, action_metadata.search_action)
    native_search_action = cast(jax.Array, native_metadata.search_action)
    action_q_weight = cast(jax.Array, action_metadata.q_weight)
    native_q_weight = cast(jax.Array, native_metadata.q_weight)
    assert jnp.array_equal(
        action_readout.posterior.prediction.policy,
        native.posterior.prediction.policy,
    )
    assert not jnp.array_equal(
        action_readout.commitment_policy,
        native.posterior.prediction.policy,
    )
    assert jnp.array_equal(
        action_alpha_v,
        native_alpha_v,
    )
    assert jnp.array_equal(
        action_alpha_q,
        native_alpha_q,
    )
    assert jnp.array_equal(
        action_search_action,
        native_search_action,
    )
    assert jnp.array_equal(action_q_weight, native_q_weight)

    if commitment_type == ActionCommitmentType.posterior_argmax:
        expected_action = posterior_best_action(
            action_readout.commitment_policy,
            env_state.legal_action_mask,
        )
    elif commitment_type in {
        ActionCommitmentType.posterior_plurality,
        ActionCommitmentType.posterior_plurality_uniform_ties,
    }:
        expected_action = posterior_plurality_action(
            action_key,
            action_readout.commitment_policy,
            env_state.legal_action_mask,
            num_samples=action_config.posterior_plurality_samples,
            tie_break=(
                "uniform"
                if commitment_type
                == ActionCommitmentType.posterior_plurality_uniform_ties
                else "lowest"
            ),
        )
    elif commitment_type == ActionCommitmentType.posterior_sample:
        expected_action = posterior_sample_action(
            action_key,
            action_readout.commitment_policy,
            env_state.legal_action_mask,
        )
    else:
        expected_action = action_search_action
    player = make_search_player(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        action_config,
        commitment_type,
        q_loss_weight_mode="policy",
    )
    assert jnp.array_equal(
        player(env_state, player_key).action,
        expected_action,
    )

    diagnostics = action_readout.posterior.diagnostics
    assert diagnostics is not None
    assert jnp.array_equal(
        diagnostics.search_root_policy_target_enabled_count,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        diagnostics.search_root_action_prefix_accepted_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.all(diagnostics.search_root_action_native_l1_sum > 0)


def test_prefix_target_and_action_share_one_root_quadrature(monkeypatch):
    calls = 0
    estimator = (
        play_search_module.binary_posterior_best_policy_prefix_quadrature
    )

    def counted_estimator(*args, **kwargs):
        nonlocal calls
        calls += 1
        return estimator(*args, **kwargs)

    monkeypatch.setattr(
        play_search_module,
        "binary_posterior_best_policy_prefix_quadrature",
        counted_estimator,
    )
    env_state = _toy_dirichlet_state()
    search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        SearchConfig(
            kind=SearchKind.dirichlet_thompson,
            dirichlet_thompson=DirichletThompsonSearchConfig(
                num_simulations=0,
                policy_samples=8,
                policy_sample_chunk_size=2,
                root_policy_target_estimator=(
                    PosteriorPolicyEstimator.prefix_cdf
                ),
                root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
                prefix_cdf_half_width=10,
            ),
        ),
    )

    output = search(_as_pgx_state(env_state), jax.random.PRNGKey(94))

    assert calls == 1
    assert output.commitment_policy is not None
    assert jnp.allclose(
        output.posterior.prediction.policy,
        output.commitment_policy,
    )


@pytest.mark.parametrize(
    "commitment_type",
    [
        "posterior_plurality",
        "posterior_plurality_uniform_ties",
        "posterior_sample",
    ],
)
def test_prefix_root_target_falls_back_per_root_outside_tail_envelope(
    commitment_type: str,
):
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

    search = make_search(
        _ToySearchEnv(),
        extreme_evaluator,
        SearchConfig(
            kind=SearchKind.dirichlet_thompson,
            dirichlet_thompson=DirichletThompsonSearchConfig(
                num_simulations=0,
                policy_samples=8,
                policy_sample_chunk_size=2,
                root_policy_target_estimator=(
                    PosteriorPolicyEstimator.prefix_cdf
                ),
                root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
                prefix_cdf_half_width=10,
            ),
        ),
    )

    output = search(_as_pgx_state(env_state), jax.random.PRNGKey(92))
    assert output.commitment_policy is not None
    assert output.commitment_resampling_bypass is not None
    assert jnp.array_equal(
        output.posterior.prediction.policy,
        output.commitment_policy,
    )
    diagnostics = output.posterior.diagnostics
    assert diagnostics is not None
    assert jnp.array_equal(
        diagnostics.search_root_policy_target_prefix_fallback_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        diagnostics.search_root_policy_target_prefix_tail_clipped_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        diagnostics.search_root_action_prefix_fallback_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        diagnostics.search_root_action_prefix_tail_clipped_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        output.commitment_resampling_bypass,
        jnp.ones((2,), dtype=jnp.bool_),
    )

    committed = make_action_committer(
        commitment_type,
        posterior_plurality_samples=32,
        posterior_sample_temperature=8.0,
    )(
        output.posterior,
        env_state.legal_action_mask,
        jax.random.PRNGKey(123),
        output.commitment_policy,
        output.commitment_resampling_bypass,
    )
    assert jnp.array_equal(
        committed,
        posterior_best_action(
            output.commitment_policy,
            env_state.legal_action_mask,
        ),
    )


@pytest.mark.parametrize(
    "posterior_estimator",
    [
        PosteriorPolicyEstimator.winner_mc,
        PosteriorPolicyEstimator.prefix_cdf,
    ],
)
def test_dirichlet_thompson_tree_search_builds_legal_targets(
    posterior_estimator: PosteriorPolicyEstimator,
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
                policy_samples=8,
                posterior_policy_samples=1,
                policy_sample_chunk_size=2,
                posterior_policy_estimator=posterior_estimator,
            ),
        ),
    )

    output = search(env_state, jax.random.PRNGKey(23))

    prediction = output.posterior.prediction
    metadata = output.posterior.metadata
    assert metadata is not None
    assert prediction.alpha_v is not None
    assert prediction.alpha_q is not None
    assert metadata.q_weight is not None
    assert metadata.search_action is not None
    assert metadata.mask is not None
    assert prediction.policy.shape == env_state.legal_action_mask.shape
    assert prediction.alpha_v.shape == (2, 2)
    assert prediction.alpha_q.shape == (2, 3, 2)
    assert jnp.allclose(prediction.policy.sum(axis=-1), 1.0)
    assert jnp.all(prediction.policy[~env_state.legal_action_mask] == 0.0)
    assert jnp.array_equal(metadata.q_weight, prediction.policy)
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
                policy_samples=4,
            ),
        ),
    )

    output = search(env_state, jax.random.PRNGKey(29))
    prediction = output.posterior.prediction
    metadata = output.posterior.metadata
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


def test_prefix_root_target_uses_categorical_population_but_native_commitment():
    env_state = _toy_dirichlet_state()
    search = make_search(
        _TerminalToySearchEnv(),
        _toy_dirichlet_evaluator,
        SearchConfig(
            kind=SearchKind.dirichlet_thompson,
            dirichlet_thompson=DirichletThompsonSearchConfig(
                num_simulations=4,
                policy_samples=4,
                root_policy_target_estimator=(
                    PosteriorPolicyEstimator.prefix_cdf
                ),
                root_action_estimator=PosteriorPolicyEstimator.prefix_cdf,
                prefix_cdf_half_width=10,
            ),
        ),
    )

    output = search(_as_pgx_state(env_state), jax.random.PRNGKey(29))
    metadata = output.posterior.metadata
    assert metadata is not None
    assert metadata.search_action is not None
    assert output.commitment_policy is not None
    expected = jax.nn.one_hot(
        metadata.search_action,
        output.posterior.prediction.policy.shape[-1],
    )
    assert jnp.array_equal(output.commitment_policy, expected)
    assert output.commitment_resampling_bypass is not None
    assert jnp.array_equal(
        output.commitment_resampling_bypass,
        jnp.ones((2,), dtype=jnp.bool_),
    )
    # This toy tree exposes one certified shortest win per solved root, so its
    # exact categorical population is the same one-hot target.
    assert jnp.array_equal(output.posterior.prediction.policy, expected)
    for commitment_type in (
        "posterior_plurality",
        "posterior_plurality_uniform_ties",
        "posterior_sample",
    ):
        committed = make_action_committer(
            commitment_type,
            posterior_plurality_samples=32,
            posterior_sample_temperature=8.0,
        )(
            output.posterior,
            env_state.legal_action_mask,
            jax.random.PRNGKey(123),
            output.commitment_policy,
            output.commitment_resampling_bypass,
        )
        assert jnp.array_equal(committed, metadata.search_action)
    diagnostics = output.posterior.diagnostics
    assert diagnostics is not None
    assert jnp.array_equal(
        diagnostics.search_root_policy_target_categorical_population_count,
        jnp.ones((2,), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        diagnostics.search_root_policy_target_prefix_eligible_count,
        jnp.zeros((2,), dtype=jnp.float32),
    )
    assert jnp.array_equal(
        diagnostics.search_root_action_prefix_eligible_count,
        jnp.zeros((2,), dtype=jnp.float32),
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
        ActionCommitmentType.posterior_argmax,
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
        ActionCommitmentType.search_action,
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


def test_search_player_threads_posterior_plurality_sample_budget():
    batch_size = 64
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
        posterior_plurality_samples=7,
        policy=PolicySearchConfig(temperature=1.0),
    )
    key = jax.random.PRNGKey(19)
    player = make_search_player(
        _ToySearchEnv(),
        logits_only_model,
        search_config,
        ActionCommitmentType.posterior_plurality,
    )

    output = player(env_state, key)

    assert output.posterior is not None
    _, action_key = jax.random.split(key)
    expected = posterior_plurality_action(
        action_key,
        output.posterior.prediction.policy,
        env_state.legal_action_mask,
        num_samples=7,
    )
    assert jnp.array_equal(output.action, expected)


def test_search_player_threads_posterior_sample_temperature():
    batch_size = 256
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
        posterior_sample_temperature=0.25,
        policy=PolicySearchConfig(temperature=1.0),
    )
    key = jax.random.PRNGKey(20)
    player = make_search_player(
        _ToySearchEnv(),
        logits_only_model,
        search_config,
        ActionCommitmentType.posterior_sample,
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
        match="posterior_sample_temperature must be finite and > 0",
    ):
        make_action_committer(
            "posterior_sample",
            posterior_sample_temperature=temperature,
        )


def test_posterior_plurality_uses_lowest_index_for_count_ties():
    key = jax.random.PRNGKey(4)
    policy = jnp.asarray(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ],
        dtype=jnp.float32,
    )
    legal = jnp.ones_like(policy, dtype=jnp.bool_)
    logits = jnp.where(
        policy > 0,
        jnp.log(jnp.where(policy > 0, policy, 1.0)),
        -jnp.inf,
    )
    votes = jax.random.categorical(key, logits, shape=(2, 2))
    counts = jnp.sum(jax.nn.one_hot(votes, 3, dtype=jnp.int32), axis=0)
    assert jnp.array_equal(
        counts,
        jnp.asarray([[1, 1, 0], [0, 1, 1]], dtype=jnp.int32),
    )

    action = posterior_plurality_action(
        key,
        policy,
        legal,
        num_samples=2,
    )

    assert jnp.array_equal(action, jnp.asarray([0, 1], dtype=jnp.int32))


def test_uniform_tie_plurality_reuses_exact_votes_and_changes_only_ties():
    batch_size = 8_192
    policy = jnp.full((batch_size, 2), 0.5, dtype=jnp.float32)
    legal = jnp.ones_like(policy, dtype=jnp.bool_)
    key = jax.random.PRNGKey(401)

    lowest = posterior_plurality_result(
        key,
        policy,
        legal,
        num_samples=2,
        tie_break="lowest",
    )
    uniform = posterior_plurality_result(
        key,
        policy,
        legal,
        num_samples=2,
        tie_break="uniform",
    )

    assert jnp.array_equal(lowest.vote_counts, uniform.vote_counts)
    assert jnp.array_equal(
        lowest.lowest_index_action,
        uniform.lowest_index_action,
    )
    assert jnp.array_equal(
        lowest.uniform_tie_action,
        uniform.uniform_tie_action,
    )
    tied = lowest.max_count_tie_multiplicity > 1
    assert bool(jnp.any(tied))
    assert bool(jnp.any(~tied))
    assert jnp.array_equal(lowest.action, lowest.lowest_index_action)
    assert jnp.array_equal(uniform.action, uniform.uniform_tie_action)
    assert jnp.array_equal(lowest.action[~tied], uniform.action[~tied])
    selected_count = jnp.take_along_axis(
        uniform.vote_counts,
        uniform.action[:, None],
        axis=-1,
    )[:, 0]
    assert jnp.array_equal(
        selected_count,
        jnp.max(uniform.vote_counts, axis=-1),
    )


def test_uniform_tie_plurality_removes_two_vote_lowest_index_bias():
    batch_size = 131_072
    policy = jnp.full((batch_size, 2), 0.5, dtype=jnp.float32)
    legal = jnp.ones_like(policy, dtype=jnp.bool_)
    key = jax.random.PRNGKey(402)

    lowest = posterior_plurality_action(
        key,
        policy,
        legal,
        num_samples=2,
    )
    uniform = posterior_plurality_action(
        key,
        policy,
        legal,
        num_samples=2,
        tie_break="uniform",
    )

    # Lowest-index ties give P(A=0)=3/4. Uniform ties restore symmetry.
    assert jnp.mean(lowest == 0) == pytest.approx(0.75, abs=0.005)
    assert jnp.mean(uniform == 0) == pytest.approx(0.5, abs=0.005)


def test_posterior_plurality_rejects_unknown_tie_break():
    with pytest.raises(ValueError, match="tie_break"):
        posterior_plurality_action(
            jax.random.PRNGKey(403),
            jnp.asarray([[0.5, 0.5]], dtype=jnp.float32),
            jnp.asarray([[True, True]]),
            tie_break="first",  # ty: ignore[invalid-argument-type]
        )


def test_posterior_plurality_is_reproducible_and_key_sensitive():
    batch_size = 512
    policy = jnp.full((batch_size, 4), 0.25, dtype=jnp.float32)
    legal = jnp.ones_like(policy, dtype=jnp.bool_)
    first_key = jax.random.PRNGKey(7)
    second_key = jax.random.PRNGKey(8)

    first = posterior_plurality_action(
        first_key,
        policy,
        legal,
        num_samples=1,
    )
    repeated = posterior_plurality_action(
        first_key,
        policy,
        legal,
        num_samples=1,
    )
    second = posterior_plurality_action(
        second_key,
        policy,
        legal,
        num_samples=1,
    )

    assert jnp.array_equal(first, repeated)
    assert jnp.array_equal(
        first,
        posterior_sample_action(first_key, policy, legal),
    )
    assert jnp.any(first != second)


def test_posterior_plurality_matches_binomial_majority_probability():
    batch_size = 65_536
    policy = jnp.broadcast_to(
        jnp.asarray([0.7, 0.3], dtype=jnp.float32),
        (batch_size, 2),
    )
    legal = jnp.ones_like(policy, dtype=jnp.bool_)

    action = posterior_plurality_action(
        jax.random.PRNGKey(11),
        policy,
        legal,
        num_samples=3,
    )

    # P(Binomial(3, 0.7) >= 2) = 0.784.
    observed = jnp.mean(action == 0)
    assert observed == pytest.approx(0.784, abs=0.01)


def test_posterior_plurality_preserves_native_solved_one_hot_action():
    policy = jnp.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    legal = jnp.ones_like(policy, dtype=jnp.bool_)

    for seed in (0, 1, 2):
        action = posterior_plurality_action(
            jax.random.PRNGKey(seed),
            policy,
            legal,
            num_samples=32,
        )
        assert jnp.array_equal(
            action,
            jnp.asarray([1, 2], dtype=jnp.int32),
        )


def test_commit_action_uses_configured_posterior_plurality_budget():
    key = jax.random.PRNGKey(13)
    policy = jnp.asarray([[0.6, 0.4]], dtype=jnp.float32)
    legal = jnp.ones_like(policy, dtype=jnp.bool_)

    played_action = commit_action(
        "posterior_plurality",
        key,
        policy,
        legal,
        posterior_plurality_samples=7,
    )

    expected = posterior_plurality_action(
        key,
        policy,
        legal,
        num_samples=7,
    )
    assert jnp.array_equal(played_action, expected)


def test_commit_action_supports_uniform_tie_plurality():
    key = jax.random.PRNGKey(404)
    policy = jnp.broadcast_to(
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        (512, 2),
    )
    legal = jnp.ones_like(policy, dtype=jnp.bool_)

    played_action = commit_action(
        "posterior_plurality_uniform_ties",
        key,
        policy,
        legal,
        posterior_plurality_samples=2,
    )
    expected = posterior_plurality_action(
        key,
        policy,
        legal,
        num_samples=2,
        tie_break="uniform",
    )

    assert jnp.array_equal(played_action, expected)


def test_plurality_commitment_diagnostics_exclude_bypassed_roots():
    shape = (4,)
    fields = {
        field: jnp.zeros(shape, dtype=jnp.float32)
        for field in SearchDiagnostics._fields
    }
    diagnostics = SearchDiagnostics(**fields)
    output = play_search_module.SearchOutput(
        posterior=PosteriorTargets(
            prediction=PosteriorPrediction(
                policy=jnp.full((4, 2), 0.5, dtype=jnp.float32)
            ),
            diagnostics=diagnostics,
        ),
        commitment_resampling_bypass=jnp.asarray(
            [False, False, True, False]
        ),
    )
    result = PosteriorPluralityResult(
        action=jnp.asarray([1, 1, 1, 0], dtype=jnp.int32),
        lowest_index_action=jnp.asarray([0, 1, 0, 0], dtype=jnp.int32),
        uniform_tie_action=jnp.asarray([1, 1, 1, 0], dtype=jnp.int32),
        vote_counts=jnp.asarray(
            [[1, 1], [0, 2], [1, 1], [2, 0]],
            dtype=jnp.int32,
        ),
        max_count_tie_multiplicity=jnp.asarray(
            [2, 1, 2, 1],
            dtype=jnp.int32,
        ),
        resampling_eligible=jnp.asarray([True, True, True, False]),
    )

    measured = (
        play_search_module._with_root_plurality_commitment_diagnostics(
            output,
            result,
        )
    )
    measured_diagnostics = measured.posterior.diagnostics
    assert measured_diagnostics is not None
    assert jnp.array_equal(
        measured_diagnostics.search_root_plurality_commitment_count,
        jnp.asarray([1.0, 1.0, 0.0, 0.0]),
    )
    assert jnp.array_equal(
        measured_diagnostics.search_root_plurality_max_count_tie_count,
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    assert jnp.array_equal(
        measured_diagnostics.search_root_plurality_tie_multiplicity_sum,
        jnp.asarray([2.0, 0.0, 0.0, 0.0]),
    )
    assert jnp.array_equal(
        (
            measured_diagnostics
            .search_root_plurality_lowest_uniform_disagreement_count
        ),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    assert jnp.allclose(
        measured_diagnostics.search_root_plurality_expected_disagreement_sum,
        jnp.asarray([0.5, 0.0, 0.0, 0.0]),
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


def test_root_policy_top2_margin_uses_ephemeral_policy_and_excludes_solved():
    shape = (3,)
    fields = {
        field: jnp.zeros(shape, dtype=jnp.float32)
        for field in SearchDiagnostics._fields
    }
    fields.update(
        search_root_count=jnp.ones(shape, dtype=jnp.float32),
        search_solved_root_count=jnp.array([0.0, 0.0, 1.0]),
    )
    diagnostics = SearchDiagnostics(**fields)
    posterior = PosteriorTargets(
        prediction=PosteriorPrediction(
            # Deliberately different from the ephemeral policy below.
            policy=jnp.array(
                [
                    [0.9, 0.1, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.9, 0.1, 0.0],
                ]
            )
        ),
        diagnostics=diagnostics,
    )
    output = play_search_module.SearchOutput(
        posterior=posterior,
        commitment_policy=jnp.array(
            [
                [0.5, 0.5, 0.0],
                [0.7, 0.2, 0.1],
                [0.6, 0.4, 0.0],
            ]
        ),
    )
    legal = jnp.array(
        [
            [True, True, False],
            [True, True, True],
            [True, True, False],
        ]
    )

    measured = play_search_module._with_root_policy_top2_margin_diagnostics(
        output,
        legal,
        action_commitment_type="posterior_plurality",
        margin_reference_scale=0.1,
    )
    assert jnp.array_equal(
        measured.posterior.prediction.policy,
        output.posterior.prediction.policy,
    )
    assert jnp.array_equal(
        measured.commitment_policy,
        output.commitment_policy,
    )
    measured_diagnostics = measured.posterior.diagnostics
    assert measured_diagnostics is not None
    assert jnp.allclose(
        measured_diagnostics.search_root_policy_top2_margin_sum,
        jnp.array([0.0, 0.5, 0.0]),
    )
    assert jnp.array_equal(
        measured_diagnostics.search_root_policy_top2_margin_count,
        jnp.array([1.0, 1.0, 0.0]),
    )
    assert jnp.array_equal(
        measured_diagnostics.search_root_policy_top2_margin_tie_count,
        jnp.array([1.0, 0.0, 0.0]),
    )
    assert jnp.array_equal(
        (
            measured_diagnostics
            .search_root_policy_top2_margin_below_reference_count
        ),
        jnp.array([1.0, 0.0, 0.0]),
    )
    assert jnp.allclose(
        (
            measured_diagnostics
            .search_root_policy_top2_margin_reference_scale_sum
        ),
        jnp.array([0.1, 0.1, 0.0]),
    )

    search_action = (
        play_search_module._with_root_policy_top2_margin_diagnostics(
            output,
            legal,
            action_commitment_type="search_action",
            margin_reference_scale=0.1,
        )
    )
    search_action_diagnostics = search_action.posterior.diagnostics
    assert search_action_diagnostics is not None
    assert jnp.all(
        search_action_diagnostics.search_root_policy_top2_margin_count == 0
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
            action_commitment_type=ActionCommitmentType.posterior_argmax,
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
            action_commitment_type=ActionCommitmentType.posterior_argmax,
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
