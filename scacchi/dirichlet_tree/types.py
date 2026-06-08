from __future__ import annotations

from typing import NamedTuple

import jax


class TreeTrainingData(NamedTuple):
    obs: jax.Array
    action_weights: jax.Array
    played_action: jax.Array
    legal_action_mask: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    value_tgt: jax.Array
    policy_loss_mask: jax.Array
    value_loss_mask: jax.Array
    search_loss_mask: jax.Array
    outcome_mask: jax.Array
    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None

class SearchDiagnostics(NamedTuple):
    path_depth_mean: jax.Array
    path_depth_p50: jax.Array
    path_depth_p90: jax.Array
    path_depth_max: jax.Array
    expanded_nodes: jax.Array
    terminal_fraction: jax.Array
    root_policy_entropy: jax.Array
    root_gamma: jax.Array
    root_downstream_eval_count: jax.Array
    root_q_concentration: jax.Array
