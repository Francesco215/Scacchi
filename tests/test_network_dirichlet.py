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
    outcome_mean,
    outcome_utility,
    policy_value_from_output,
)
from scacchi.train import Config


def test_dirichlet_from_logits_uses_squared_softplus_concentration():
    mean_logits = jnp.array([[0.0, 0.0]])
    concentration_logit = jnp.array([0.0])

    alpha = dirichlet_from_logits(mean_logits, concentration_logit)

    assert jnp.allclose(alpha.sum(axis=-1), jax.nn.softplus(concentration_logit) ** 2)
    assert jnp.allclose(outcome_mean(alpha), jnp.array([[0.5, 0.5]]))


def test_dirichlet_from_logits_clips_total_concentration():
    mean_logits = jnp.array([[0.0, 0.0, 0.0]])
    concentration_logit = jnp.array([100.0])

    alpha = dirichlet_from_logits(mean_logits, concentration_logit, concentration_clip=5.0)

    assert jnp.allclose(alpha.sum(axis=-1), jnp.array([5.0]))
    assert jnp.allclose(outcome_mean(alpha), jnp.full((1, 3), 1 / 3))


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


def test_boardlaw_dirichlet_heads_initialize_to_uniform_policy_and_unit_alphas():
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

    assert jnp.allclose(logits, jnp.zeros_like(logits))
    assert jnp.allclose(jax.nn.softmax(logits, axis=-1), jnp.full_like(logits, 0.1))
    assert jnp.allclose(alpha_v, jnp.ones_like(alpha_v))
    assert jnp.allclose(alpha_q, jnp.ones_like(alpha_q))


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


def test_build_model_supports_az_dirichlet_for_go_wdl3():
    config = Config(
        env_id="go",
        board_size=8,
        network="aznet_dirichlet",
        search_policy="dirichlet_thompson",
        num_outcomes=None,
        num_channels=8,
        num_layers=1,
    )

    model = build_model(
        config,
        num_actions=65,
        observation_shape=(8, 8, 17),
        rngs=nnx.Rngs(0),
    )

    assert isinstance(model, AZDirichletNet)
    assert model.num_outcomes == 3


def test_dirichlet_thompson_null_hex_outcomes_builds_legacy_two_outcome_head():
    config = Config(
        env_id="hex",
        network="boardlaw_dirichlet",
        search_policy="dirichlet_thompson",
        num_outcomes=None,
        num_channels=16,
        num_layers=2,
    )

    model = build_model(
        config,
        num_actions=10,
        observation_shape=(3, 3, 4),
        rngs=nnx.Rngs(0),
    )

    assert isinstance(model, BoardlawDirichletNet)
    assert model.num_outcomes == 2


def test_legacy_dirichlet_head_init_matches_runstate_initial_concentration():
    config = Config(
        env_id="hex",
        network="boardlaw_dirichlet",
        search_policy="dirichlet_thompson",
        num_outcomes=2,
        num_channels=16,
        num_layers=2,
        legacy_dirichlet_head_init=True,
        rezero_kernel_init="orthogonal",
    )
    model = build_model(
        config,
        num_actions=10,
        observation_shape=(3, 3, 4),
        rngs=nnx.Rngs(0),
    )

    _, alpha_v, alpha_q = model(jnp.zeros((1, 3, 3, 4)), train=False)

    expected_total = jax.nn.softplus(jnp.array(0.0)) ** 2
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
