import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax
import pytest

from scacchi.network import AZDirichletNet, BoardlawDirichletNet
from scacchi.train import _build_optimizer, _muon_weight_dimension_numbers
from scacchi.types import Config, TrainingConfig


def _az_model() -> AZDirichletNet:
    return AZDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=2,
        num_channels=8,
        num_blocks=2,
        dtype=jnp.float32,
        dirichlet_initial_concentration=3.1,
        rngs=nnx.Rngs(0),
    )


def _boardlaw_model() -> BoardlawDirichletNet:
    return BoardlawDirichletNet(
        num_actions=10,
        observation_shape=(3, 3, 4),
        num_outcomes=2,
        width=8,
        depth=2,
        dtype=jnp.float32,
        dirichlet_initial_concentration=3.1,
        rngs=nnx.Rngs(0),
    )


@pytest.mark.parametrize(
    ("model_factory", "expected_count", "expected_reduction", "expected_output"),
    [
        (_az_model, 4, (0, 1, 2), 3),
        (_boardlaw_model, 2, (0,), 1),
    ],
)
def test_muon_selects_only_hidden_block_kernels(
    model_factory,
    expected_count: int,
    expected_reduction: tuple[int, ...],
    expected_output: int,
):
    specs = _muon_weight_dimension_numbers(
        nnx.state(model_factory(), nnx.Param)
    )
    leaves = jax.tree_util.tree_flatten_with_path(
        specs,
        is_leaf=lambda value: value is None
        or isinstance(value, optax.contrib.MuonDimensionNumbers),
    )[0]
    selected = [(jax.tree_util.keystr(path), spec) for path, spec in leaves if spec is not None]

    assert len(selected) == expected_count
    assert all(path.startswith("['blocks']") for path, _ in selected)
    assert all("['kernel'].value" in path for path, _ in selected)
    assert all(spec.reduction_axis == expected_reduction for _, spec in selected)
    assert all(spec.output_axis == expected_output for _, spec in selected)


@pytest.mark.parametrize("model_factory", [_az_model, _boardlaw_model])
def test_hybrid_muon_optimizer_update_runs_under_nnx_jit(model_factory):
    model = model_factory()
    optimizer = _build_optimizer(model, Config(training=TrainingConfig()))
    grads = jax.tree.map(
        lambda value: 1e-3 * (jnp.sin(value * 7.0) + 0.1),
        nnx.state(model, nnx.Param),
    )
    update = nnx.jit(
        lambda current_model, current_optimizer, current_grads: (
            current_optimizer.update(current_model, current_grads)
        )
    )

    update(model, optimizer, grads)
    update(model, optimizer, grads)

    assert int(optimizer.step[...]) == 2
    for value in jax.tree.leaves(nnx.state(model, nnx.Param)):
        assert np.isfinite(np.asarray(value)).all()
