import jax.numpy as jnp
import numpy as np

from scripts.fig_8 import _coerce_dqaz_wdl3_output


def test_coerce_dqaz_wdl3_output_inserts_draw_channel_for_hex_lw_heads():
    logits = jnp.zeros((2, 3), dtype=jnp.float32)
    alpha_v = jnp.asarray([[2.0, 5.0], [3.0, 7.0]], dtype=jnp.float32)
    alpha_q = jnp.asarray(
        [
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            [[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]],
        ],
        dtype=jnp.float32,
    )

    out_logits, out_v, out_q = _coerce_dqaz_wdl3_output(
        (logits, alpha_v, alpha_q),
        draw_alpha=1e-4,
        min_alpha=0.0,
    )

    assert out_logits is logits
    assert out_v.shape == (2, 3)
    assert out_q.shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(out_v[:, 0]), np.asarray(alpha_v[:, 0]))
    np.testing.assert_allclose(np.asarray(out_v[:, 1]), 1e-4)
    np.testing.assert_allclose(np.asarray(out_v[:, 2]), np.asarray(alpha_v[:, 1]))
    np.testing.assert_allclose(np.asarray(out_q[..., 0]), np.asarray(alpha_q[..., 0]))
    np.testing.assert_allclose(np.asarray(out_q[..., 1]), 1e-4)
    np.testing.assert_allclose(np.asarray(out_q[..., 2]), np.asarray(alpha_q[..., 1]))


def test_coerce_dqaz_wdl3_output_floors_tiny_alpha_values():
    logits = jnp.zeros((1, 2), dtype=jnp.float32)
    alpha_v = jnp.asarray([[1e-8, 0.2]], dtype=jnp.float32)
    alpha_q = jnp.asarray([[[1e-8, 0.3], [0.4, 1e-8]]], dtype=jnp.float32)

    _, out_v, out_q = _coerce_dqaz_wdl3_output(
        (logits, alpha_v, alpha_q),
        draw_alpha=1e-8,
        min_alpha=0.05,
    )

    assert float(jnp.min(out_v)) >= 0.05
    assert float(jnp.min(out_q)) >= 0.05
