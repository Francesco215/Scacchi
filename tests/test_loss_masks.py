import jax
import jax.numpy as jnp
import optax
import pytest
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from scacchi.distributed import BatchParallel, assert_batch_axis_sharded
from scacchi.dirichlet_tree.types import TreeTrainingData
from scacchi.dirichlet_tree.native import TARGET_CATEGORICAL, dirichlet_nll_at_categorical
from scacchi.dirichlet_tree.native import native_fields_from_beta
from scacchi.loss import (
    DIRICHLET_KL_LOSS_CUTOFF,
    Sample,
    _compute_dirichlet_losses,
    _compute_losses,
    _masked_mean,
    _dirichlet_kl,
    make_compute_input_for_lossfn,
)
from scacchi.pipeline import _fixed_replay_window, make_minibatches
from scacchi.play import SelfplayOutput
from scacchi.types import (
    Config,
    GumbelSearchConfig,
    ModelConfig,
    SearchConfig,
    SearchConstantsConfig,
    SelfplayConfig,
    TrainingConfig,
    TrainingLossConfig,
)


def _batch_mesh():
    return jax.make_mesh(
        (jax.device_count(),),
        ("batch",),
        axis_types=(AxisType.Auto,),
    )


def _loss_config(
    *,
    max_num_steps: int = 1,
    policy_loss_weight: float = 1.0,
    value_dir_kl_weight: float = 0.0,
    q_dir_kl_weight: float = 0.0,
    value_outcome_weight: float = 0.0,
    q_outcome_weight: float = 0.0,
    categorical_epsilon: float = 1e-4,
    terminal_edge_targets: bool = False,
    terminal_parent_targets: bool = False,
) -> Config:
    return Config(
        model=ModelConfig(network="boardlaw_dirichlet"),
        selfplay=SelfplayConfig(max_num_steps=max_num_steps),
        search=SearchConfig(
            gumbel=GumbelSearchConfig(
                constants=SearchConstantsConfig(
                    categorical_epsilon=categorical_epsilon,
                ),
            ),
        ),
        training=TrainingConfig(
            losses=TrainingLossConfig(
                policy_weight=policy_loss_weight,
                value_dir_kl_weight=value_dir_kl_weight,
                q_dir_kl_weight=q_dir_kl_weight,
                value_outcome_weight=value_outcome_weight,
                q_outcome_weight=q_outcome_weight,
                terminal_edge_targets=terminal_edge_targets,
                terminal_parent_targets=terminal_parent_targets,
            ),
        ),
    )


def _sample_posterior_fields(num_rows: int, num_actions: int = 2, num_outcomes: int = 2):
    return {
        "beta_Q_target": jnp.ones((num_rows, num_actions, num_outcomes)),
        "beta_V_target": jnp.ones((num_rows, num_outcomes)),
        "q_loss_weight": jnp.zeros((num_rows, num_actions)),
    }


def test_compute_loss_input_preserves_root_legal_action_mask():
    data = SelfplayOutput(
        obs=jnp.zeros((2, 3, 1)),
        reward=jnp.zeros((2, 3)),
        terminated=jnp.array(
            [
                [False, True, False],
                [False, False, False],
            ]
        ),
        action_weights=jnp.zeros((2, 3, 4)),
        played_action=jnp.array(
            [
                [0, 1, 3],
                [2, 0, 1],
            ]
        ),
        legal_action_mask=jnp.array(
            [
                [
                    [True, True, False, False],
                    [False, True, True, False],
                    [True, False, False, True],
                ],
                [
                    [True, False, True, False],
                    [True, True, False, False],
                    [False, True, False, True],
                ],
            ]
        ),
        beta_Q_target=jnp.ones((2, 3, 4, 2)),
        beta_V_target=jnp.ones((2, 3, 2)),
        q_loss_weight=jnp.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0],
                ],
                [
                    [0.0, 0.0, 2.0, 0.0],
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0, 0.0],
                ],
            ]
        ),
        discount=-jnp.ones((2, 3)),
    )
    config = _loss_config(max_num_steps=3)

    sample = make_compute_input_for_lossfn(config)(data)

    assert jnp.array_equal(sample.policy_mask, data.legal_action_mask)
    assert jnp.array_equal(sample.played_action, data.played_action)
    assert jnp.array_equal(sample.beta_Q_target, data.beta_Q_target)
    assert jnp.array_equal(sample.beta_V_target, data.beta_V_target)
    assert jnp.array_equal(sample.q_loss_weight, data.q_loss_weight)
    assert jnp.array_equal(
        sample.value_mask,
        jnp.array(
            [
                [True, True, False],
                [False, False, False],
            ]
        ),
    )


def test_compute_loss_input_preserves_sample_batch_sharding():
    device_count = jax.device_count()
    mesh = _batch_mesh()
    parallel = BatchParallel(enabled=True, mesh=mesh)
    batch_size = max(device_count * 2, 2)
    data = SelfplayOutput(
        obs=jnp.zeros((batch_size, 2, 1), dtype=jnp.float32),
        reward=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        terminated=jnp.zeros((batch_size, 2), dtype=jnp.bool_),
        action_weights=jnp.ones((batch_size, 2, 3), dtype=jnp.float32) / 3.0,
        played_action=jnp.zeros((batch_size, 2), dtype=jnp.int32),
        legal_action_mask=jnp.ones((batch_size, 2, 3), dtype=jnp.bool_),
        beta_Q_target=jnp.ones((batch_size, 2, 3, 2), dtype=jnp.float32),
        beta_V_target=jnp.ones((batch_size, 2, 2), dtype=jnp.float32),
        q_loss_weight=jnp.zeros((batch_size, 2, 3), dtype=jnp.float32),
        discount=-jnp.ones((batch_size, 2), dtype=jnp.float32),
    )
    data = jax.tree_util.tree_map(
        lambda leaf: (
            jax.device_put(leaf, parallel.sharding_for(leaf.ndim, batch_axis=0))
            if isinstance(leaf, jax.Array)
            else leaf
        ),
        data,
    )
    config = _loss_config(max_num_steps=2)

    @jax.jit
    def compute(data):
        return make_compute_input_for_lossfn(config)(data)

    with parallel.mesh_context():
        lowered = jax.jit(compute).lower(data)
        hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text().lower()
        for collective in (
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
        ):
            assert collective not in hlo_text

        sample = compute(data)
    sample = assert_batch_axis_sharded(sample, parallel, batch_axis=0, label="computed sample")
    assert sample.q_target_weight.shape == (batch_size, 2, 3)


def test_compute_loss_input_appends_tree_rows_with_separate_loss_masks():
    tree_data = TreeTrainingData(
        obs=jnp.array([[[10.0], [20.0]]]),
        action_weights=jnp.array([[[1.0, 0.0], [0.0, 0.0]]]),
        played_action=jnp.array([[0, 0]]),
        legal_action_mask=jnp.array([[[True, False], [False, False]]]),
        beta_Q_target=jnp.ones((1, 2, 2, 2)),
        beta_V_target=jnp.ones((1, 2, 2)),
        q_loss_weight=jnp.array([[[1.0, 0.0], [0.0, 0.0]]]),
        value_tgt=jnp.array([[0.5, 1.0]]),
        policy_loss_mask=jnp.array([[True, False]]),
        value_loss_mask=jnp.array([[True, True]]),
        search_loss_mask=jnp.array([[True, False]]),
        outcome_mask=jnp.array([[False, True]]),
    )
    data = SelfplayOutput(
        obs=jnp.array([[[1.0]]]),
        reward=jnp.array([[1.0]]),
        terminated=jnp.array([[True]]),
        action_weights=jnp.array([[[0.0, 1.0]]]),
        played_action=jnp.array([[1]]),
        legal_action_mask=jnp.array([[[True, True]]]),
        beta_Q_target=jnp.ones((1, 1, 2, 2)),
        beta_V_target=jnp.ones((1, 1, 2)),
        q_loss_weight=jnp.zeros((1, 1, 2)),
        discount=jnp.zeros((1, 1)),
        tree_data=tree_data,
    )
    config = _loss_config(max_num_steps=1)

    sample = make_compute_input_for_lossfn(config)(data)

    assert sample.obs.shape == (1, 3, 1)
    assert jnp.array_equal(sample.policy_loss_mask, jnp.array([[True, True, False]]))
    assert jnp.array_equal(sample.value_loss_mask, jnp.array([[True, True, True]]))
    assert jnp.array_equal(sample.outcome_mask, jnp.array([[True, False, True]]))


def test_compute_loss_input_trains_root_search_targets_before_terminal_result():
    data = SelfplayOutput(
        obs=jnp.zeros((3, 2, 1)),
        reward=jnp.zeros((3, 2)),
        terminated=jnp.zeros((3, 2), dtype=jnp.bool_),
        action_weights=jnp.full((3, 2, 4), 0.25),
        played_action=jnp.zeros((3, 2), dtype=jnp.int32),
        legal_action_mask=jnp.ones((3, 2, 4), dtype=jnp.bool_),
        beta_Q_target=jnp.ones((3, 2, 4, 3)),
        beta_V_target=jnp.ones((3, 2, 3)),
        q_loss_weight=jnp.ones((3, 2, 4)) / 4.0,
        discount=-jnp.ones((3, 2)),
    )
    config = _loss_config(max_num_steps=2)

    sample = make_compute_input_for_lossfn(config)(data)

    assert jnp.array_equal(sample.policy_loss_mask, jnp.ones((3, 2), dtype=jnp.bool_))
    assert jnp.array_equal(sample.value_loss_mask, jnp.ones((3, 2), dtype=jnp.bool_))
    assert jnp.array_equal(sample.outcome_mask, jnp.zeros((3, 2), dtype=jnp.bool_))


def test_compute_loss_input_can_mark_played_terminal_edge_categorical():
    data = SelfplayOutput(
        obs=jnp.zeros((1, 2, 1)),
        reward=jnp.array([[0.0, 1.0]]),
        terminated=jnp.array([[False, True]]),
        action_weights=jnp.ones((1, 2, 3)) / 3.0,
        played_action=jnp.array([[0, 2]]),
        legal_action_mask=jnp.ones((1, 2, 3), dtype=jnp.bool_),
        beta_Q_target=jnp.ones((1, 2, 3, 3)),
        beta_V_target=jnp.ones((1, 2, 3)),
        q_loss_weight=jnp.zeros((1, 2, 3)),
        discount=jnp.array([[-1.0, 0.0]]),
    )
    config = _loss_config(max_num_steps=2, terminal_edge_targets=True)

    sample = make_compute_input_for_lossfn(config)(data)

    assert sample.q_target_kind[0, 1, 2] == int(TARGET_CATEGORICAL)
    assert sample.q_target_outcome[0, 1, 2] == 2
    assert sample.q_target_distance[0, 1, 2] == 1
    assert sample.q_loss_weight[0, 1, 2] == 1.0
    assert not bool(jnp.any(sample.q_target_kind[0, 0] == int(TARGET_CATEGORICAL)))
    assert not bool(jnp.any(sample.q_target_kind[0, 1, :2] == int(TARGET_CATEGORICAL)))


def test_compute_loss_input_can_mark_terminal_winning_parent_categorical():
    data = SelfplayOutput(
        obs=jnp.zeros((1, 2, 1)),
        reward=jnp.array([[0.0, 1.0]]),
        terminated=jnp.array([[False, True]]),
        action_weights=jnp.ones((1, 2, 3)) / 3.0,
        played_action=jnp.array([[0, 2]]),
        legal_action_mask=jnp.ones((1, 2, 3), dtype=jnp.bool_),
        beta_Q_target=jnp.ones((1, 2, 3, 3)),
        beta_V_target=jnp.ones((1, 2, 3)),
        q_loss_weight=jnp.zeros((1, 2, 3)),
        discount=jnp.array([[-1.0, 0.0]]),
    )
    config = _loss_config(max_num_steps=2, terminal_parent_targets=True)

    sample = make_compute_input_for_lossfn(config)(data)

    assert jnp.array_equal(sample.policy_tgt[0, 1], jnp.array([0.0, 0.0, 1.0]))
    assert bool(sample.policy_loss_mask[0, 1])
    assert bool(sample.value_loss_mask[0, 1])
    assert sample.v_target_kind[0, 1] == int(TARGET_CATEGORICAL)
    assert sample.v_target_outcome[0, 1] == 2
    assert sample.v_target_distance[0, 1] == 1
    assert not bool(jnp.any(sample.q_target_kind == int(TARGET_CATEGORICAL)))


def test_masked_mean_surfaces_active_nonfinite_terms():
    assert not bool(jnp.isfinite(_masked_mean(jnp.array([jnp.nan]), jnp.array([True]))))
    assert jnp.allclose(
        _masked_mean(jnp.array([jnp.nan, 2.0]), jnp.array([False, True])),
        2.0,
    )


def _minibatch_sample(num_steps: int, batch_size: int) -> Sample:
    obs = jnp.arange(num_steps * batch_size, dtype=jnp.float32).reshape(
        batch_size,
        num_steps,
        1,
    )
    return Sample(
        obs=obs,
        policy_tgt=jnp.ones((batch_size, num_steps, 2)) / 2,
        value_tgt=jnp.zeros((batch_size, num_steps)),
        played_action=jnp.zeros((batch_size, num_steps), dtype=jnp.int32),
        policy_mask=jnp.ones((batch_size, num_steps, 2), dtype=jnp.bool_),
        value_mask=jnp.ones((batch_size, num_steps), dtype=jnp.bool_),
        beta_Q_target=jnp.ones((batch_size, num_steps, 2, 2)),
        beta_V_target=jnp.ones((batch_size, num_steps, 2)),
        q_loss_weight=jnp.zeros((batch_size, num_steps, 2)),
    )


def test_make_minibatches_shuffles_rows_without_dropping_masked_samples():
    sample_fields = _minibatch_sample(1, 8)._asdict()
    sample_fields.update(
        policy_loss_mask=jnp.array(
            [[False], [True], [False], [False], [False], [False], [True], [False]]
        ),
        value_loss_mask=jnp.zeros((8, 1), dtype=jnp.bool_),
    )
    sample = Sample(**sample_fields)

    minibatches = make_minibatches(sample, jax.random.PRNGKey(0), 4)

    assert minibatches.obs.shape == (2, 4, 1)
    assert jnp.array_equal(
        jnp.sort(minibatches.obs[..., 0].reshape(-1)),
        jnp.arange(8, dtype=jnp.float32),
    )
    assert jnp.sum(minibatches.policy_loss_mask) == 2


def test_make_minibatches_respects_max_updates_per_iter():
    sample = _minibatch_sample(1, 6)

    minibatches = make_minibatches(
        sample,
        jax.random.PRNGKey(0),
        2,
        max_updates_per_iter=2,
    )

    assert minibatches.obs.shape == (2, 2, 1)
    assert minibatches.beta_Q_target.shape == (2, 2, 2, 2)
    assert jnp.unique(minibatches.obs[..., 0].reshape(-1)).shape[0] == 4


def test_batch_parallel_minibatches_are_sharded_and_communication_free():
    device_count = jax.device_count()
    mesh = _batch_mesh()
    parallel = BatchParallel(enabled=True, mesh=mesh)
    local_batch_size = 4
    num_steps = 3
    training_batch_size = device_count * 4
    sample = _minibatch_sample(num_steps, device_count * local_batch_size)
    sample = jax.tree_util.tree_map(
        lambda leaf: jax.device_put(
            leaf,
            parallel.sharding_for(leaf.ndim, batch_axis=0),
        ),
        sample,
    )

    def build_minibatches(sample, rng_key):
        return make_minibatches(sample, rng_key, training_batch_size, parallel=parallel)

    with parallel.mesh_context():
        lowered = jax.jit(build_minibatches).lower(sample, jax.random.PRNGKey(0))
        hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text().lower()
        for collective in (
            "all-gather",
            "all_gather",
            "all-reduce",
            "all_reduce",
            "all-to-all",
            "all_to_all",
            "collective-permute",
            "collective_permute",
            "collective-broadcast",
            "collective_broadcast",
            "reduce-scatter",
            "reduce_scatter",
        ):
            assert collective not in hlo_text

        minibatches = jax.jit(build_minibatches)(sample, jax.random.PRNGKey(0))

    assert minibatches.obs.shape == (3, training_batch_size, 1)
    assert isinstance(minibatches.obs.sharding, NamedSharding)
    if device_count > 1:
        assert parallel.sharding_for(
            minibatches.obs.ndim,
            batch_axis=1,
        ).is_equivalent_to(minibatches.obs.sharding, minibatches.obs.ndim)
        assert parallel.sharding_for(
            minibatches.beta_Q_target.ndim,
            batch_axis=1,
        ).is_equivalent_to(
            minibatches.beta_Q_target.sharding,
            minibatches.beta_Q_target.ndim,
        )

    obs = minibatches.obs[..., 0].reshape(3, device_count, -1)
    batch_indices = obs.astype(jnp.int32) // num_steps
    owner_devices = batch_indices // local_batch_size
    expected_devices = jnp.arange(device_count, dtype=jnp.int32)[None, :, None]
    assert bool(jnp.all(owner_devices == expected_devices))


def test_assert_batch_axis_sharded_rejects_replicated_input_on_multi_device():
    device_count = jax.device_count()
    if device_count < 2:
        pytest.skip("replicated and batch-sharded layouts are equivalent on one device")

    mesh = _batch_mesh()
    parallel = BatchParallel(enabled=True, mesh=mesh)

    @jax.jit
    def check(value):
        return assert_batch_axis_sharded(value, parallel, batch_axis=0, label="test_value")

    value = jax.device_put(
        jnp.arange(device_count * 2),
        NamedSharding(mesh, PartitionSpec()),
    )
    with pytest.raises(Exception, match="test_value"):
        with parallel.mesh_context():
            check(value).block_until_ready()


def test_assert_batch_axis_sharded_accepts_batch_sharded_input():
    device_count = jax.device_count()
    mesh = _batch_mesh()
    parallel = BatchParallel(enabled=True, mesh=mesh)
    sharding = parallel.sharding_for(ndim=1)
    value = jax.device_put(jnp.arange(max(device_count * 2, 1)), sharding)

    @jax.jit
    def check(value):
        return assert_batch_axis_sharded(value, parallel, batch_axis=0, label="test_value")

    with parallel.mesh_context():
        checked = check(value)
    assert checked.shape == value.shape


def test_native_defaults_preserve_beta_batch_sharding():
    device_count = jax.device_count()
    mesh = _batch_mesh()
    parallel = BatchParallel(enabled=True, mesh=mesh)
    batch_size = max(device_count * 2, 2)
    beta_q = jax.device_put(
        jnp.ones((batch_size, 3, 2, 2), dtype=jnp.float32),
        parallel.sharding_for(ndim=4, batch_axis=0),
    )
    beta_v = jax.device_put(
        jnp.ones((batch_size, 3, 2), dtype=jnp.float32),
        parallel.sharding_for(ndim=3, batch_axis=0),
    )

    @jax.jit
    def build_defaults(beta_q, beta_v):
        return native_fields_from_beta(beta_q, beta_v)

    with parallel.mesh_context():
        lowered = jax.jit(build_defaults).lower(beta_q, beta_v)
        hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text().lower()
        for collective in (
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
        ):
            assert collective not in hlo_text

        defaults = build_defaults(beta_q, beta_v)
    defaults = assert_batch_axis_sharded(defaults, parallel, batch_axis=0, label="native defaults")
    assert defaults["q_target_weight"].shape == (batch_size, 3, 2)
    assert defaults["v_target_weight"].shape == (batch_size, 3)


def test_fixed_replay_window_pads_early_batches_to_stable_shape():
    first = object()
    second = object()
    third = object()
    fourth = object()
    fifth = object()

    assert _fixed_replay_window([first], 4) == [first, first, first, first]
    assert _fixed_replay_window([first, second], 4) == [first, first, first, second]
    assert _fixed_replay_window([first, second, third, fourth, fifth], 4) == [
        second,
        third,
        fourth,
        fifth,
    ]


def test_policy_loss_ignores_illegal_logits():
    data = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[1.0, 0.0, 0.0]]),
        value_tgt=jnp.array([0.0]),
        played_action=jnp.array([0]),
        policy_mask=jnp.array([[True, False, True]]),
        value_mask=jnp.array([True]),
        **_sample_posterior_fields(1, num_actions=3),
    )
    value = jnp.array([0.0])

    high_illegal_loss, _ = _compute_losses(
        jnp.array([[0.0, 1000.0, 0.0]]),
        value,
        data,
    )
    low_illegal_loss, _ = _compute_losses(
        jnp.array([[0.0, -1000.0, 0.0]]),
        value,
        data,
    )

    assert jnp.allclose(high_illegal_loss, low_illegal_loss)
    assert jnp.allclose(high_illegal_loss, jnp.log(2.0))


def test_value_mask_excludes_policy_and_value_losses_from_average():
    data = Sample(
        obs=jnp.zeros((2, 1)),
        policy_tgt=jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        value_tgt=jnp.array([0.0, 0.0]),
        played_action=jnp.array([0, 1]),
        policy_mask=jnp.ones((2, 2), dtype=jnp.bool_),
        value_mask=jnp.array([True, False]),
        **_sample_posterior_fields(2),
    )
    logits = jnp.array(
        [
            [0.0, 0.0],
            [-1000.0, 1000.0],
        ]
    )
    value = jnp.array([0.0, 1000.0])

    policy_loss, value_loss = _compute_losses(logits, value, data)

    expected_policy_loss = optax.softmax_cross_entropy(
        logits[:1],
        data.policy_tgt[:1],
    )[0]
    assert jnp.allclose(policy_loss, expected_policy_loss)
    assert jnp.allclose(value_loss, 0.0)


def test_dirichlet_kl_is_zero_for_identical_parameters_and_positive_otherwise():
    beta = jnp.array([[2.0, 3.0]])

    same = _dirichlet_kl(beta, beta)
    different = _dirichlet_kl(beta, jnp.array([[3.0, 2.0]]))

    assert jnp.allclose(same, 0.0, atol=1e-6)
    assert different[0] > 0.0


def test_dirichlet_kl_losses_use_value_policy_and_q_evidence_masks():
    data = Sample(
        obs=jnp.zeros((2, 1)),
        policy_tgt=jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        value_tgt=jnp.array([1.0, 1.0]),
        played_action=jnp.array([0, 0]),
        policy_mask=jnp.array(
            [
                [True, False, True],
                [True, True, True],
            ]
        ),
        value_mask=jnp.array([True, False]),
        beta_Q_target=jnp.array(
            [
                [[1.0, 1.0], [1000.0, 1.0], [1.0, 1.0]],
                [[1000.0, 1.0], [1000.0, 1.0], [1000.0, 1.0]],
            ]
        ),
        beta_V_target=jnp.array([[1.0, 2.0], [1000.0, 1.0]]),
        q_loss_weight=jnp.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),
        search_loss_mask=jnp.array([True, False]),
    )
    logits = jnp.zeros((2, 3))
    alpha_v = jnp.array([[1.0, 2.0], [1.0, 1000.0]])
    alpha_q = jnp.array(
        [
            [[1.0, 1.0], [1.0, 1000.0], [2.0, 1.0]],
            [[1.0, 1000.0], [1.0, 1000.0], [1.0, 1000.0]],
        ]
    )
    config = _loss_config(
        policy_loss_weight=0.0,
        value_dir_kl_weight=1.0,
        q_dir_kl_weight=1.0,
    )

    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    expected_q = _dirichlet_kl(data.beta_Q_target[0, 2], alpha_q[0, 2])
    assert jnp.allclose(metrics.value_dir_kl_loss, 0.0, atol=1e-6)
    assert jnp.allclose(metrics.q_dir_kl_loss, expected_q, atol=1e-6)
    assert jnp.allclose(metrics.q_loss_weight_mean, 1.0)


def test_dirichlet_kl_losses_ignore_huge_and_nonfinite_terms():
    data = Sample(
        obs=jnp.zeros((3, 1)),
        policy_tgt=jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        value_tgt=jnp.array([1.0, 1.0, 1.0]),
        played_action=jnp.array([0, 0, 0]),
        policy_mask=jnp.ones((3, 3), dtype=jnp.bool_),
        value_mask=jnp.ones((3,), dtype=jnp.bool_),
        beta_Q_target=jnp.array(
            [
                [[2.0, 3.0], [1e6, 1.0], [jnp.nan, 1.0]],
                [[1e6, 1.0], [1e6, 1.0], [1e6, 1.0]],
                [[jnp.nan, 1.0], [jnp.nan, 1.0], [jnp.nan, 1.0]],
            ]
        ),
        beta_V_target=jnp.array(
            [
                [2.0, 3.0],
                [1e6, 1.0],
                [jnp.nan, 1.0],
            ]
        ),
        q_loss_weight=jnp.ones((3, 3)),
    )
    logits = jnp.zeros((3, 3))
    alpha_v = jnp.array([[2.0, 3.0], [1.0, 1e6], [1.0, 1.0]])
    alpha_q = jnp.array(
        [
            [[2.0, 3.0], [1.0, 1e6], [1.0, 1.0]],
            [[1.0, 1e6], [1.0, 1e6], [1.0, 1e6]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ]
    )
    config = _loss_config(
        policy_loss_weight=0.0,
        value_dir_kl_weight=1.0,
        q_dir_kl_weight=1.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
    )

    raw_value_kl = _dirichlet_kl(data.beta_V_target, alpha_v)
    raw_q_kl = _dirichlet_kl(data.beta_Q_target, alpha_q)
    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    assert raw_value_kl[1] > DIRICHLET_KL_LOSS_CUTOFF
    assert not bool(jnp.isfinite(raw_value_kl[2]))
    assert raw_q_kl[0, 1] > DIRICHLET_KL_LOSS_CUTOFF
    assert not bool(jnp.isfinite(raw_q_kl[0, 2]))
    assert jnp.allclose(metrics.value_dir_kl_loss, 0.0, atol=1e-6)
    assert jnp.allclose(metrics.q_dir_kl_loss, 0.0, atol=1e-6)


def test_policy_kl_hat_is_nll_minus_sampled_target_entropy():
    data = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[0.25, 0.75]]),
        value_tgt=jnp.array([1.0]),
        played_action=jnp.array([1]),
        policy_mask=jnp.array([[True, True]]),
        value_mask=jnp.array([True]),
        beta_Q_target=jnp.ones((1, 2, 2)),
        beta_V_target=jnp.ones((1, 2)),
        q_loss_weight=jnp.zeros((1, 2)),
    )
    logits = jnp.array([[0.0, 0.0]])
    alpha_v = jnp.ones((1, 2))
    alpha_q = jnp.ones((1, 2, 2))
    config = _loss_config(
        policy_loss_weight=1.0,
        value_dir_kl_weight=0.0,
        q_dir_kl_weight=0.0,
    )

    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    expected_entropy = -jnp.sum(data.policy_tgt[0] * jnp.log(data.policy_tgt[0]))
    assert jnp.allclose(metrics.policy_target_entropy, expected_entropy)
    assert jnp.allclose(
        metrics.policy_kl_hat,
        metrics.policy_nll_loss - metrics.policy_target_entropy,
    )


def test_native_categorical_targets_use_dirichlet_density_nll():
    data = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[1.0, 0.0]]),
        value_tgt=jnp.array([1.0]),
        played_action=jnp.array([0]),
        policy_mask=jnp.array([[True, True]]),
        value_mask=jnp.array([True]),
        beta_Q_target=jnp.ones((1, 2, 3)),
        beta_V_target=jnp.ones((1, 3)),
        q_loss_weight=jnp.array([[1.0, 0.0]]),
        q_target_kind=jnp.array([[int(TARGET_CATEGORICAL), 0]], dtype=jnp.int8),
        q_target_weight=jnp.ones((1, 2), dtype=jnp.float32),
        q_target_outcome=jnp.array([[2, -1]], dtype=jnp.int8),
        q_target_distance=jnp.array([[1, -1]], dtype=jnp.int32),
        v_target_kind=jnp.array([int(TARGET_CATEGORICAL)], dtype=jnp.int8),
        v_target_weight=jnp.ones((1,), dtype=jnp.float32),
        v_target_outcome=jnp.array([2], dtype=jnp.int8),
        v_target_distance=jnp.array([0], dtype=jnp.int32),
    )
    logits = jnp.zeros((1, 2))
    alpha_v = jnp.array([[1.5, 2.0, 4.0]])
    alpha_q = jnp.array([[[1.0, 1.5, 3.0], [3.0, 1.0, 1.0]]])
    config = _loss_config(
        policy_loss_weight=0.0,
        value_dir_kl_weight=1.0,
        q_dir_kl_weight=1.0,
        categorical_epsilon=1e-4,
    )

    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    expected_v = dirichlet_nll_at_categorical(alpha_v[0], jnp.asarray(2), 1e-4)
    expected_q = dirichlet_nll_at_categorical(alpha_q[0, 0], jnp.asarray(2), 1e-4)
    assert jnp.allclose(metrics.value_dir_kl_loss, expected_v)
    assert jnp.allclose(metrics.q_dir_kl_loss, expected_q)


def test_debug_outcome_losses_use_dirichlet_mean_nll_not_density():
    data = Sample(
        obs=jnp.zeros((1, 1)),
        policy_tgt=jnp.array([[1.0, 0.0]]),
        value_tgt=jnp.array([1.0]),
        played_action=jnp.array([0]),
        policy_mask=jnp.array([[True, True]]),
        value_mask=jnp.array([True]),
        outcome_mask=jnp.array([True]),
        beta_Q_target=jnp.ones((1, 2, 3)),
        beta_V_target=jnp.ones((1, 3)),
        q_loss_weight=jnp.array([[1.0, 0.0]]),
    )
    logits = jnp.zeros((1, 2))
    alpha_v = jnp.array([[0.2, 0.2, 0.6]])
    alpha_q = jnp.array([[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]])
    concentrated_alpha_v = alpha_v * 10.0
    concentrated_alpha_q = alpha_q * 10.0
    config = _loss_config(
        policy_loss_weight=0.0,
        value_dir_kl_weight=0.0,
        q_dir_kl_weight=0.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
        categorical_epsilon=1e-4,
    )

    _, metrics = _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)
    _, concentrated_metrics = _compute_dirichlet_losses(
        logits,
        concentrated_alpha_v,
        concentrated_alpha_q,
        data,
        config,
    )

    assert jnp.allclose(metrics.value_outcome_loss, -jnp.log(jnp.asarray(0.6)))
    assert jnp.allclose(metrics.q_outcome_loss, -jnp.log(jnp.asarray(0.7)))
    assert jnp.allclose(
        concentrated_metrics.value_outcome_loss,
        metrics.value_outcome_loss,
    )
    assert jnp.allclose(
        concentrated_metrics.q_outcome_loss,
        metrics.q_outcome_loss,
    )
