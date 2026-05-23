from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .types import NodeBlob


class PackedNodes(NamedTuple):
    edge_post_alpha: jnp.ndarray
    legal_actions: jnp.ndarray
    action_mask: jnp.ndarray


def pack_nodes_for_selection(nodes: list[NodeBlob]) -> PackedNodes:
    batch_size = len(nodes)
    max_actions = max((int(node.legal_actions.shape[0]) for node in nodes), default=1)
    max_actions = max(1, max_actions)
    num_outcomes = int(nodes[0].value_alpha.shape[0]) if nodes else 3
    alpha = np.ones((batch_size, max_actions, num_outcomes), dtype=np.float32)
    legal_actions = np.zeros((batch_size, max_actions), dtype=np.int32)
    action_mask = np.zeros((batch_size, max_actions), dtype=bool)
    for ix, node in enumerate(nodes):
        count = int(node.legal_actions.shape[0])
        if count == 0:
            continue
        alpha[ix, :count] = node.edge_post_alpha
        legal_actions[ix, :count] = node.legal_actions.astype(np.int32)
        action_mask[ix, :count] = True
    return PackedNodes(
        edge_post_alpha=jnp.asarray(alpha),
        legal_actions=jnp.asarray(legal_actions),
        action_mask=jnp.asarray(action_mask),
    )
