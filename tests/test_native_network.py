"""The INT8 search network (network_kernel=int8, ``native_binned/conv_int8.cuh``).

Its sums are exact integers, so a position's result must not depend on the
batch it is in, on its place in the batch, or on how the kernel cuts the batch
into units; and it must stay close to the bf16 network it approximates.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.envs import make_env
from scacchi.network import AZBinnedNet
from scacchi.types import BinConfig, BinnedThompsonSearchConfig


def _kernel_available():
    if jax.default_backend() != "gpu":
        return False
    try:
        from scacchi.native_binned._library import find_nvcc

        find_nvcc()
    except RuntimeError:
        return False
    return True


needs_kernel = pytest.mark.skipif(not _kernel_available(), reason="needs a CUDA GPU and nvcc")


def _model(env, seed):
    model = AZBinnedNet(env.num_actions, env.observation_shape, num_channels=128, num_blocks=6,
                        dtype=jnp.bfloat16, bins=BinConfig(count=9), rngs=nnx.Rngs(seed))
    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    nnx.update(model, jax.tree.unflatten(treedef, [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]))
    return model


def _observations(env, batch, key):
    """Positions after a random number of random moves (finished games stay as they are)."""
    init_key, key = jax.random.split(key)
    state = jax.vmap(env.init)(jax.random.split(init_key, batch))
    plies = env.num_actions - 1
    stop = jax.random.randint(jax.random.fold_in(key, 1), (batch,), 0, plies)

    def body(state, item):
        step, step_key = item
        logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
        action = jax.random.categorical(step_key, logits).astype(jnp.int32)
        stepped = jax.vmap(env.step)(state, action)
        keep = state.terminated | (step >= stop)
        return jax.tree.map(lambda old, new: jnp.where(keep.reshape(keep.shape + (1,) * (new.ndim - 1)), old, new), state, stepped), None

    state, _ = jax.lax.scan(body, state, (jnp.arange(plies), jax.random.split(key, plies)))
    return state.observation


def _evaluate(model, kernel):
    from scacchi.native_binned.network import make_native_evaluator

    @nnx.jit
    def run(model, obs):
        out = make_native_evaluator(model, kernel)(obs)
        return out.logits, out.alpha_v, out.alpha_q

    return lambda obs: [np.asarray(x) for x in run(model, obs)]


@needs_kernel
@pytest.mark.parametrize("size,batch", [(3, 61), (5, 203), (7, 517), (11, 131)])
def test_int8_network_does_not_depend_on_the_batch(size, batch):
    env = make_env("hex", size)
    model = _model(env, seed=size)
    obs = _observations(env, batch, jax.random.PRNGKey(size))
    run = _evaluate(model, "int8")
    whole = run(obs)
    assert all(np.isfinite(x).all() for x in whole)
    order = np.asarray(jax.random.permutation(jax.random.PRNGKey(1), batch))
    shuffled = run(obs[order])
    head = run(obs[: batch // 3])  # other units, other CTAs
    single = run(obs[batch - 1:])
    for a, b, c, d in zip(whole, shuffled, head, single):
        np.testing.assert_array_equal(b.view(np.uint32), a[order].view(np.uint32))
        np.testing.assert_array_equal(c.view(np.uint32), a[: batch // 3].view(np.uint32))
        np.testing.assert_array_equal(d.view(np.uint32), a[batch - 1:].view(np.uint32))


@needs_kernel
def test_int8_network_is_close_to_the_bf16_network():
    """With BatchNorm statistics that fit the positions, as training leaves them.

    The 8-bit grid of a convolution input is laid over the range its BatchNorm
    statistics predict, so the statistics are first brought to these positions
    (60 passes in training mode, momentum 0.9), as they are after training.
    """
    env = make_env("hex", 7)
    model = _model(env, seed=2)
    obs = _observations(env, 256, jax.random.PRNGKey(3))

    @nnx.jit
    def calibrate(model, obs):
        model(obs, train=True)

    for _ in range(60):
        calibrate(model, obs)
    exact, approx = _evaluate(model, "cuda")(obs), _evaluate(model, "int8")(obs)
    assert float(np.mean(np.abs(exact[0] - approx[0]))) < 0.05 * float(np.std(exact[0]))
    assert float(np.max(np.abs(exact[1] - approx[1]))) < 0.05
    assert float(np.mean(np.abs(exact[2] - approx[2]))) < 0.01


def test_network_kernel_values():
    assert BinnedThompsonSearchConfig(network_kernel="int8").network_kernel == "int8"
    with pytest.raises(ValueError, match="network_kernel"):
        BinnedThompsonSearchConfig(network_kernel="int4")
