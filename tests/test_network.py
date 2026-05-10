import jax
import jax.numpy as jnp

from scacchi.network import _dirichlet_from_logits


def test_dirichlet_from_logits_uses_exp_concentration_without_additive_base():
    mean_logits = jnp.array([[0.0, 1.0, -1.0]], dtype=jnp.float32)
    concentration_logit = jnp.array([jnp.log(2.5)], dtype=jnp.float32)

    alpha = _dirichlet_from_logits(mean_logits, concentration_logit)

    expected = 2.5 * jax.nn.softmax(mean_logits, axis=-1)
    assert jnp.allclose(alpha, expected)
    assert jnp.allclose(alpha.sum(axis=-1), jnp.array([2.5], dtype=jnp.float32))
    assert not jnp.allclose(alpha, 1.0 + jax.nn.softplus(concentration_logit)[..., None] * jax.nn.softmax(mean_logits, axis=-1))
