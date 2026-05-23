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


def test_num_search_blocks_must_be_positive():
    with pytest.raises(ValidationError):
        Config(network="boardlaw_dirichlet", num_search_blocks=0)


def test_grad_clip_norm_must_be_positive_when_set():
    with pytest.raises(ValidationError):
        Config(network="boardlaw_dirichlet", grad_clip_norm=0.0)


def test_grad_clip_norm_can_be_disabled():
    config = Config(network="boardlaw_dirichlet", grad_clip_norm=None)

    assert config.grad_clip_norm is None


def test_dirichlet_kl_loss_cutoff_must_be_positive():
    with pytest.raises(ValidationError):
        Config(network="boardlaw_dirichlet", dirichlet_kl_loss_cutoff=0.0)


def test_posterior_sample_action_source_is_valid():
    config = Config(
        network="boardlaw_dirichlet",
        selfplay_action_source="posterior_sample",
    )

    assert config.selfplay_action_source == "posterior_sample"


def test_selfplay_action_source_must_be_known():
    with pytest.raises(ValidationError, match="selfplay_action_source"):
        Config(network="boardlaw_dirichlet", selfplay_action_source="unknown")


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
