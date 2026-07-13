from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import pgx
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from scacchi.dirichlet_q_search import outcome_utility, posterior_sample_action
from scacchi.distributed import BatchParallel, assert_batch_axis_sharded
from scacchi.network import BoardlawNet
from scacchi.play import TrainingSamples, play, make_selfplay
from scacchi.play_search import (
    EvaluatorOutput,
    PlayerOutput,
    PosteriorPrediction,
    PosteriorTargets,
    TargetMetadata,
    commit_action,
    make_search,
    make_search_player,
)
from scacchi.types import (
    ActionCommitmentType,
    Config,
    DirichletThompsonSearchConfig,
    EnvConfig,
    GumbelSearchConfig,
    ModelConfig,
    Network,
    PolicySearchConfig,
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
    metadata = output.posterior.metadata
    assert metadata is not None
    assert metadata.search_action is not None
    assert output.posterior.prediction.policy.shape == (batch_size, num_actions)
    assert metadata.search_action.shape == (batch_size,)


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

    root_prediction = _toy_dirichlet_evaluator(env_state.observation)
    assert root_prediction.alpha_q is not None
    search_key, _, _ = jax.random.split(rng_key, 3)
    sampled_outcome = jax.random.dirichlet(search_key, root_prediction.alpha_q)
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


def test_dirichlet_thompson_tree_search_builds_legal_targets():
    env_state = _toy_dirichlet_state()
    search = make_search(
        _ToySearchEnv(),
        _toy_dirichlet_evaluator,
        SearchConfig(
            kind=SearchKind.dirichlet_thompson,
            dirichlet_thompson=DirichletThompsonSearchConfig(
                num_simulations=2,
                max_depth=2,
                num_blocks=2,
                policy_samples=8,
                policy_sample_chunk_size=2,
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
    expected = posterior_sample_action(
        key,
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
