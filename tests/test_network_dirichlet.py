from flax import nnx
import jax
import jax.numpy as jnp

from scacchi.network import (
    BoardlawDirichletNet,
    BoardlawNet,
    dirichlet_from_logits,
    outcome_mean,
    outcome_utility,
    policy_value_from_output,
)


def test_dirichlet_from_logits_uses_softplus_concentration():
    mean_logits = jnp.array([[0.0, 0.0]])
    concentration_logit = jnp.array([0.0])

    alpha = dirichlet_from_logits(mean_logits, concentration_logit)

    assert jnp.allclose(alpha.sum(axis=-1), jax.nn.softplus(concentration_logit))
    assert jnp.allclose(outcome_mean(alpha), jnp.array([[0.5, 0.5]]))


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
