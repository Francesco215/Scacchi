from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from scacchi.train import _build_optimizer, _muon_dimension_numbers, _psgd_kron
from scacchi.types import Config, OptimizerType


class _NestedBlock(nnx.Module):
    def __init__(self, width: int, *, rngs: nnx.Rngs):
        self.alpha = nnx.Param(jnp.zeros(()))
        self.linear = nnx.Linear(width, width, rngs=rngs)


class _NestedModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.blocks = nnx.List(
            [_NestedBlock(3, rngs=rngs), _NestedBlock(3, rngs=rngs)]
        )
        self.head = nnx.Linear(3, 2, rngs=rngs)


def test_muon_dimension_numbers_include_dense_and_convolution_kernels():
    params = {
        "dense": jnp.zeros((8, 4)),
        "conv": jnp.zeros((3, 3, 4, 16)),
        "bias": jnp.zeros((16,)),
    }

    dimensions = _muon_dimension_numbers(params)

    assert dimensions["dense"] == optax.contrib.MuonDimensionNumbers((0,), 1)
    assert dimensions["conv"] == optax.contrib.MuonDimensionNumbers((0, 1, 2), 3)
    assert dimensions["bias"] is None


def test_build_optimizer_selects_muon():
    model = nnx.Conv(3, 4, kernel_size=(3, 3), rngs=nnx.Rngs(0))
    config = Config()
    config.training.optimizer = OptimizerType.muon

    with patch(
        "scacchi.train.optax.contrib.muon",
        wraps=optax.contrib.muon,
    ) as muon:
        optimizer = _build_optimizer(model, config)

    muon.assert_called_once_with(
        config.training.learning_rate,
        muon_weight_dimension_numbers=_muon_dimension_numbers,
    )
    grads = jax.tree.map(
        jnp.ones_like,
        nnx.state(model, nnx.Param),
    )
    optimizer.update(model, grads)

    assert jax.tree.all(
        jax.tree.map(
            lambda param: jnp.all(jnp.isfinite(param)),
            nnx.state(model, nnx.Param),
        )
    )


def test_build_optimizer_selects_psgd():
    model = _NestedModel(rngs=nnx.Rngs(0))
    config = Config()
    config.training.optimizer = OptimizerType.psgd

    with patch("scacchi.train._psgd_kron", wraps=_psgd_kron) as psgd_kron:
        optimizer = _build_optimizer(model, config)

    psgd_kron.assert_called_once_with(config.training.learning_rate)
    initial_state_structure = jax.tree.structure(nnx.as_pure(optimizer.opt_state))

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_step(state, step):
        scanned_model, scanned_optimizer = state
        grads = jax.tree.map(
            jnp.ones_like,
            nnx.state(scanned_model, nnx.Param),
        )
        scanned_optimizer.update(scanned_model, grads)
        return (scanned_model, scanned_optimizer), step

    scan_step((model, optimizer), jnp.arange(2))

    assert jax.tree.structure(nnx.as_pure(optimizer.opt_state)) == (
        initial_state_structure
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda param: jnp.all(jnp.isfinite(param)),
            nnx.state(model, nnx.Param),
        )
    )
