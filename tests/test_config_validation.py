import pytest
from pydantic import ValidationError

from scacchi.train import Config


def test_scalar_network_rejects_dirichlet_loss_weights():
    with pytest.raises(ValidationError, match="network='boardlaw_dirichlet'"):
        Config(
            network="boardlaw",
            value_dir_kl_weight=1.0,
            q_dir_kl_weight=0.0,
            value_outcome_weight=0.0,
            q_outcome_weight=0.0,
        )


def test_scalar_network_allows_zero_dirichlet_loss_weights():
    config = Config(
        network="boardlaw",
        value_dir_kl_weight=0.0,
        q_dir_kl_weight=0.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
    )

    assert config.network == "boardlaw"


def test_dirichlet_network_allows_dirichlet_loss_weights():
    config = Config(network="boardlaw_dirichlet", value_dir_kl_weight=1.0)

    assert config.network == "boardlaw_dirichlet"


def test_model_construction_context_allows_legacy_checkpoint_loss_weights():
    config = Config.model_validate(
        {
            "network": "boardlaw",
            "value_dir_kl_weight": 1.0,
            "q_dir_kl_weight": 1.0,
        },
        context={"model_construction_only": True},
    )

    assert config.network == "boardlaw"
