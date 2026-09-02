import math

from flax import nnx
import jax
import jax.numpy as jnp

from scacchi.network import (
    AZDirichletNet,
    AZNet,
    BoardlawDirichletNet,
    BoardlawNet,
    build_model,
    dirichlet_from_logits,
    policy_value_from_output,
)
from scacchi.dirichlet_mctx.outcomes import outcome_mean, outcome_utility
from scacchi.types import (
    Config,
    EnvConfig,
    ModelConfig,
    Network,
    RezeroKernelInit,
    SearchConfig,
    SearchKind,
)


def test_dirichlet_from_logits_uses_direct_log_concentration():
    mean_logits = jnp.array([[0.0, 0.0]])
    log_concentration = jnp.log(jnp.array([3.0]))

    alpha = dirichlet_from_logits(mean_logits, log_concentration)

    assert jnp.allclose(alpha.sum(axis=-1), jnp.array([3.0]))
    assert jnp.allclose(outcome_mean(alpha), jnp.array([[0.5, 0.5]]))


def test_direct_log_concentration_has_unsquashed_radial_gradient():
    mean_logits = jnp.array([[0.0, 0.0]])
    log_concentration = jnp.log(jnp.array([2.0]))

    def concentration(candidate):
        return jnp.sum(
            dirichlet_from_logits(mean_logits, candidate)
        )

    total = concentration(log_concentration)
    gradient = jax.grad(concentration)(log_concentration)

    assert jnp.allclose(total, 2.0)
    assert jnp.allclose(gradient, 2.0)


def test_categorical_reference_does_not_clip_direct_head():
    mean_logits = jnp.zeros((1, 3))
    alpha = dirichlet_from_logits(
        mean_logits,
        jnp.log(jnp.array([32.0])),
        concentration_clip=5.0,
    )

    assert jnp.allclose(alpha.sum(axis=-1), jnp.array([32.0]))
    assert jnp.allclose(outcome_mean(alpha), jnp.full((1, 3), 1 / 3))


def test_legacy_parameterization_preserves_bounded_checkpoint_semantics():
    mean_logits = jnp.zeros((1, 3))

    def concentration(logit):
        return jnp.sum(
            dirichlet_from_logits(
                mean_logits,
                logit,
                parameterization="legacy",
                concentration_floor=3.0,
                concentration_clip=8.0,
            )
        )

    low = jnp.array([-8.0])
    high = jnp.array([8.0])

    assert 3.0 < concentration(low) < 8.0
    assert 3.0 < concentration(high) < 8.0
    assert jax.grad(concentration)(low)[0] > 0.0
    assert jax.grad(concentration)(high)[0] > 0.0


def test_boardlaw_dirichlet_net_shapes_and_positive_alphas():
    model = BoardlawDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=2,
        width=16,
        depth=2,
        rngs=nnx.Rngs(0),
    )
    obs = jnp.zeros((4, 3, 3, 4))

    logits, alpha_v, alpha_q = model(obs, train=False)

    assert logits.shape == (4, 10)
    assert alpha_v.shape == (4, 2)
    assert alpha_q.shape == (4, 10, 2)
    assert jnp.all(alpha_v > 0)
    assert jnp.all(alpha_q > 0)
    assert jnp.allclose(outcome_mean(alpha_v).sum(axis=-1), 1.0)
    assert jnp.allclose(outcome_mean(alpha_q).sum(axis=-1), 1.0)


def test_boardlaw_dirichlet_heads_initialize_near_uniform_dumb_prior():
    model = BoardlawDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=3,
        width=16,
        depth=2,
        rngs=nnx.Rngs(0),
    )
    obs = jnp.ones((4, 3, 3, 4))

    logits, alpha_v, alpha_q = model(obs, train=False)

    assert model.dirichlet_head_parameterization == "log_concentration"
    assert jnp.allclose(logits, jnp.zeros_like(logits))
    assert jnp.allclose(jax.nn.softmax(logits, axis=-1), jnp.full_like(logits, 0.1))
    expected_alpha = (3.0 + 0.1) / 3.0
    assert jnp.allclose(alpha_v, jnp.full_like(alpha_v, expected_alpha))
    assert jnp.allclose(alpha_q, jnp.full_like(alpha_q, expected_alpha))


def test_boardlaw_dirichlet_heads_accept_trainable_initial_concentration():
    model = BoardlawDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=2,
        width=16,
        depth=2,
        dirichlet_concentration_clip=8.0,
        dirichlet_initial_concentration=32.0,
        rngs=nnx.Rngs(0),
    )
    obs = jnp.ones((4, 3, 3, 4))

    _, alpha_v, alpha_q = model(obs, train=False)

    assert jnp.allclose(jnp.sum(alpha_v, axis=-1), 32.0)
    assert jnp.allclose(jnp.sum(alpha_q, axis=-1), 32.0)
    assert jnp.allclose(outcome_mean(alpha_v), 0.5)
    assert jnp.allclose(outcome_mean(alpha_q), 0.5)


def test_direct_concentration_keeps_recovery_gradient_below_dumb_prior():
    mean_logits = jnp.zeros((1, 2))
    low_concentration = 0.5
    low_logit = jnp.asarray([math.log(low_concentration)])

    def concentration(logit):
        return jnp.sum(
            dirichlet_from_logits(mean_logits, logit)
        )

    assert jnp.allclose(concentration(low_logit), low_concentration)
    assert jnp.allclose(
        jax.grad(concentration)(low_logit)[0],
        low_concentration,
    )


def test_boardlaw_legacy_head_accepts_configurable_concentration_floor():
    model = BoardlawDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=2,
        width=16,
        depth=2,
        dirichlet_concentration_clip=100.0,
        dirichlet_concentration_floor=32.0,
        dirichlet_head_parameterization="legacy",
        rngs=nnx.Rngs(0),
    )
    obs = jnp.ones((4, 3, 3, 4))

    _, alpha_v, alpha_q = model(obs, train=False)

    assert model.dirichlet_concentration_floor == 32.0
    assert jnp.allclose(jnp.sum(alpha_v, axis=-1), 32.1)
    assert jnp.allclose(jnp.sum(alpha_q, axis=-1), 32.1)
    assert jnp.allclose(outcome_mean(alpha_v), 0.5)
    assert jnp.allclose(outcome_mean(alpha_q), 0.5)


def test_az_dirichlet_net_shapes_and_unit_alphas():
    model = AZDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=3,
        num_channels=8,
        num_blocks=1,
        rngs=nnx.Rngs(0),
    )
    obs = jnp.ones((2, 3, 3, 4))

    logits, alpha_v, alpha_q = model(obs, train=False)

    assert logits.shape == (2, 10)
    assert alpha_v.shape == (2, 3)
    assert alpha_q.shape == (2, 10, 3)
    assert jnp.allclose(logits, jnp.zeros_like(logits))
    assert jnp.allclose(alpha_v, jnp.ones_like(alpha_v))
    assert jnp.allclose(alpha_q, jnp.ones_like(alpha_q))


def test_az_dirichlet_heads_honor_direct_initial_concentration():
    model = AZDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=2,
        num_channels=8,
        num_blocks=1,
        dirichlet_initial_concentration=3.1,
        rngs=nnx.Rngs(0),
    )
    obs = jnp.ones((2, 3, 3, 4))

    _, alpha_v, alpha_q = model(obs, train=False)

    assert model.dirichlet_head_parameterization == "log_concentration"
    assert model.dirichlet_initial_concentration == 3.1
    assert jnp.allclose(jnp.sum(alpha_v, axis=-1), 3.1)
    assert jnp.allclose(jnp.sum(alpha_q, axis=-1), 3.1)
    assert jnp.allclose(outcome_mean(alpha_v), 0.5)
    assert jnp.allclose(outcome_mean(alpha_q), 0.5)


def test_az_dirichlet_q_heads_use_az_style_flattened_board_hidden_layer():
    model = AZDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=3,
        num_channels=8,
        num_blocks=1,
        rngs=nnx.Rngs(0),
    )

    assert model.q_dir_linear.kernel[...].shape == (3 * 3 * 3, 8)
    assert model.q_dir_out.kernel[...].shape == (8, 10 * 3)
    assert model.q_conc_linear.kernel[...].shape == (3 * 3, 8)
    assert model.q_conc_out.kernel[...].shape == (8, 10)


def test_build_model_supports_az_dirichlet_for_go_wdl3():
    config = Config(
        env=EnvConfig(id="go", board_size=8, num_outcomes=None),
        model=ModelConfig(
            network=Network.aznet_dirichlet,
            num_channels=8,
            num_layers=1,
        ),
        search=SearchConfig(kind=SearchKind.dirichlet_thompson),
    )

    model = build_model(
        config,
        num_actions=65,
        observation_shape=(8, 8, 17),
        rngs=nnx.Rngs(0),
    )

    assert isinstance(model, AZDirichletNet)
    assert model.num_outcomes == 3


def test_build_model_passes_az_direct_log_concentration_configuration():
    config = Config(
        env=EnvConfig(
            id="go_9x9_white_wins_draw",
            board_size=9,
            num_outcomes=2,
        ),
        model=ModelConfig(
            network=Network.aznet_dirichlet,
            num_channels=8,
            num_layers=1,
            dirichlet_initial_concentration=3.1,
        ),
        search=SearchConfig(kind=SearchKind.dirichlet_thompson),
    )
    config.training.regularization.dirichlet_concentration_clip = 16.0

    model = build_model(
        config,
        num_actions=82,
        observation_shape=(9, 9, 17),
        rngs=nnx.Rngs(0),
    )

    assert isinstance(model, AZDirichletNet)
    assert model.dirichlet_head_parameterization == "log_concentration"
    assert model.dirichlet_concentration_clip is None
    assert model.dirichlet_initial_concentration == 3.1


def test_dirichlet_thompson_null_hex_outcomes_builds_legacy_two_outcome_head():
    config = Config(
        env=EnvConfig(id="hex", num_outcomes=None),
        model=ModelConfig(
            network=Network.boardlaw_dirichlet,
            num_channels=16,
            num_layers=2,
        ),
        search=SearchConfig(kind=SearchKind.dirichlet_thompson),
    )

    model = build_model(
        config,
        num_actions=10,
        observation_shape=(3, 3, 4),
        rngs=nnx.Rngs(0),
    )

    assert isinstance(model, BoardlawDirichletNet)
    assert model.num_outcomes == 2


def test_legacy_random_head_initialization_uses_active_direct_transform():
    config = Config(
        env=EnvConfig(id="hex", num_outcomes=2),
        model=ModelConfig(
            network=Network.boardlaw_dirichlet,
            num_channels=16,
            num_layers=2,
            legacy_dirichlet_head_init=True,
            rezero_kernel_init=RezeroKernelInit.orthogonal,
        ),
        search=SearchConfig(kind=SearchKind.dirichlet_thompson),
    )
    model = build_model(
        config,
        num_actions=10,
        observation_shape=(3, 3, 4),
        rngs=nnx.Rngs(0),
    )

    _, alpha_v, alpha_q = model(jnp.zeros((1, 3, 3, 4)), train=False)

    expected_total = jnp.asarray(1.0)
    assert jnp.allclose(jnp.sum(alpha_v, axis=-1), expected_total)
    assert jnp.allclose(jnp.sum(alpha_q, axis=-1), expected_total)


def test_scalar_policy_heads_initialize_to_uniform_logits():
    obs = jnp.ones((2, 3, 3, 4))
    az_model = AZNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_channels=8,
        num_blocks=1,
        rngs=nnx.Rngs(0),
    )
    boardlaw_model = BoardlawNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        width=16,
        depth=1,
        rngs=nnx.Rngs(1),
    )

    az_logits, _ = az_model(obs, train=False)
    boardlaw_logits, _ = boardlaw_model(obs, train=False)

    assert jnp.allclose(az_logits, jnp.zeros_like(az_logits))
    assert jnp.allclose(boardlaw_logits, jnp.zeros_like(boardlaw_logits))


def test_policy_value_adapter_supports_scalar_and_dirichlet_boardlaw():
    obs = jnp.zeros((2, 3, 3, 4))
    scalar_model = BoardlawNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        width=16,
        depth=1,
        rngs=nnx.Rngs(1),
    )
    dirichlet_model = BoardlawDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=2,
        width=16,
        depth=1,
        rngs=nnx.Rngs(2),
    )

    scalar_logits, scalar_value = policy_value_from_output(scalar_model(obs, train=False))
    dirichlet_output = dirichlet_model(obs, train=False)
    dirichlet_logits, dirichlet_value = policy_value_from_output(dirichlet_output)
    _, alpha_v, _ = dirichlet_output

    assert scalar_logits.shape == (2, 10)
    assert scalar_value.shape == (2,)
    assert dirichlet_logits.shape == (2, 10)
    assert dirichlet_value.shape == (2,)
    assert jnp.allclose(dirichlet_value, outcome_utility(outcome_mean(alpha_v)))
