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


def test_posterior_sample_action_source_is_valid():
    config = Config(
        network="boardlaw_dirichlet",
        selfplay_action_source="posterior_sample",
    )

    assert config.selfplay_action_source == "posterior_sample"


def test_scalar_q_argmax_action_source_is_valid():
    config = Config(
        network="boardlaw_dirichlet",
        selfplay_action_source="scalar_q_argmax",
    )

    assert config.selfplay_action_source == "scalar_q_argmax"


def test_selfplay_action_source_must_be_known():
    with pytest.raises(ValidationError, match="selfplay_action_source"):
        Config(network="boardlaw_dirichlet", selfplay_action_source="unknown")


def test_search_policy_must_be_known():
    with pytest.raises(ValidationError, match="search_policy"):
        Config(network="boardlaw_dirichlet", search_policy="unknown")


def test_posterior_tree_search_requires_dirichlet_network_and_wdl3():
    with pytest.raises(ValidationError, match="boardlaw_dirichlet"):
        Config(
            network="boardlaw",
            search_policy="posterior_tree",
            value_dir_kl_weight=0.0,
            q_dir_kl_weight=0.0,
            value_outcome_weight=0.0,
            q_outcome_weight=0.0,
        )
    with pytest.raises(ValidationError, match="WDL3"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree",
            num_outcomes=2,
        )


def test_wavefront_posterior_tree_search_policy_is_valid():
    config = Config(network="boardlaw_dirichlet", search_policy="posterior_tree_wavefront")

    assert config.search_policy == "posterior_tree_wavefront"
    assert config.wavefront_backend == "arena"


def test_tree_node_training_requires_wavefront_policy():
    config = Config(
        network="boardlaw_dirichlet",
        search_policy="posterior_tree_wavefront",
        train_tree_nodes=True,
    )

    assert config.train_tree_nodes is True

    with pytest.raises(ValidationError, match="train_tree_nodes"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="gumbel",
            train_tree_nodes=True,
        )


def test_wavefront_knobs_are_validated():
    config = Config(
        network="boardlaw_dirichlet",
        search_policy="posterior_tree_wavefront",
        wavefront_num_lanes_per_root=2,
        wavefront_max_depth=32,
        wavefront_final_action_mode="posterior_sample",
    )

    assert config.wavefront_num_lanes_per_root == 2
    assert config.wavefront_max_depth == 32
    assert config.wavefront_final_action_mode == "posterior_sample"
    assert config.wavefront_pad_eval_batches is True
    assert config.wavefront_pad_jax_select is False
    assert config.wavefront_np_select_below == 1024
    assert config.wavefront_grouped_expansion is True
    assert config.wavefront_lane_indexed_step is True
    assert config.wavefront_stable_lane_batch is True
    assert config.wavefront_pad_pending_observation_gather is True

    assert (
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree_wavefront",
            wavefront_final_action_mode="scalar_q_argmax",
        ).wavefront_final_action_mode
        == "scalar_q_argmax"
    )
    assert (
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree_wavefront",
            wavefront_final_action_mode="argmax_q_mean",
        ).wavefront_final_action_mode
        == "argmax_q_mean"
    )

    with pytest.raises(ValidationError, match="wavefront_final_action_mode"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree_wavefront",
            wavefront_final_action_mode="unknown",
        )


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
