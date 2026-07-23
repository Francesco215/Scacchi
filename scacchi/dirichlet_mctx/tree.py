"""Fixed-capacity data for Dirichlet Thompson tree search."""

from __future__ import annotations

from typing import ClassVar, Protocol

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int8, Int32, Shaped

from .base import RecurrentState, RootFnOutput, StoredRecurrentState, UnbatchedRecurrentState
from .outcomes import NO_DISTANCE, NO_OUTCOME, align_outcome


@chex.dataclass(frozen=True)
class PosteriorUpdate:
    """Ephemeral result returned by one node-posterior repair.

    The persistent tree is deliberately flat.  This object only groups the
    three arrays produced by a configurable repair callback before search
    writes them back into their tree slots.
    """

    edge_alpha: Float[Array, "batch action outcome"]
    edge_payload: Int32[Array, "batch action"]
    value_alpha: Float[Array, "batch outcome"]


@chex.dataclass(frozen=True)
class NodeView:
    """The current node passed to a configurable posterior repair rule."""

    index: Int32[Array, "batch"]
    embedding: RecurrentState
    value_prior: Float[Array, "batch outcome"]
    value_alpha: Float[Array, "batch outcome"]
    node_payload: Int32[Array, "batch"]
    edge_alpha: Float[Array, "batch action outcome"]
    edge_payload: Int32[Array, "batch action"]
    edge_categorical_outcome: Int8[Array, "batch action"]
    to_play: Int32[Array, "batch"]
    invalid_actions: Bool[Array, "batch action"]


@chex.dataclass(frozen=True)
class ChildrenView:
    """Action-indexed child summaries needed to repair their parent.

    ``node_payload`` is support for unresolved children and distance for
    categorical children.  Consumers must inspect ``categorical_outcome``
    before interpreting it. ``embedding_table`` remains ungathered so large
    game states are not copied over the complete action axis.
    """

    index: Int32[Array, "batch action"]
    visited: Bool[Array, "batch action"]
    embedding_table: StoredRecurrentState
    value_prior: Float[Array, "batch action outcome"]
    value_alpha: Float[Array, "batch action outcome"]
    node_payload: Int32[Array, "batch action"]
    categorical_outcome: Int8[Array, "batch action"]
    to_play: Int32[Array, "batch action"]


@chex.dataclass(frozen=True)
class LeafView:
    """The selected leaf edge evaluated by the current simulation."""

    action: Int32[Array, "batch"]
    value_alpha: Float[Array, "batch outcome"]
    to_play: Int32[Array, "batch"]
    active: Bool[Array, "batch"]


@chex.dataclass(frozen=True)
class PosteriorUpdateContext:
    """Per-node input to ``PosteriorUpdateFn`` during bottom-up backup."""

    node: NodeView
    children: ChildrenView
    leaf: LeafView
    active: Bool[Array, "batch"]


@chex.dataclass(frozen=True)
class SearchSummary:
    """Root search targets decoded from the compact tree representation."""

    # Structural counts exist only for unresolved edges. Categorical entries
    # are zero because their shared payload stores distance instead.
    visit_counts: Float[Array, "batch action"]
    alpha: Float[Array, "batch action outcome"]
    value_alpha: Float[Array, "batch outcome"]
    q_categorical_outcome: Int8[Array, "batch action"]
    q_categorical_distance: Int32[Array, "batch action"]
    v_categorical_outcome: Int8[Array, "batch"]
    v_categorical_distance: Int32[Array, "batch"]


class UnbatchedTree(Protocol):
    """Shape-precise structural view of one ``Tree`` lane.

    JAX preserves the concrete ``Tree`` class when ``vmap`` removes its batch
    axis.  This protocol describes those transformed leaf shapes without
    introducing a second runtime tree type for action-selection callbacks.
    """

    parents: Int32[Array, "node"]
    children_index: Int32[Array, "node action"]
    node_to_play: Int32[Array, "node"]
    node_categorical_outcome: Int8[Array, "node"]
    node_payload: Int32[Array, "node"]
    edge_categorical_outcome: Int8[Array, "node action"]
    edge_payload: Int32[Array, "node action"]
    node_value_priors: Float[Array, "node outcome"]
    node_value_alpha: Float[Array, "node outcome"]
    edge_alpha: Float[Array, "node action outcome"]
    invalid_actions: Bool[Array, "node action"]
    simulation_active_count: Int32[Array, ""]
    executed_simulation_call_count: Int32[Array, ""]
    embeddings: UnbatchedRecurrentState

    ROOT_INDEX: ClassVar[int]
    NO_PARENT: ClassVar[int]
    UNVISITED: ClassVar[int]

    @property
    def num_actions(self) -> int:
        ...

    @property
    def searchable_actions(self) -> Bool[Array, "node action"]:
        ...

    def replace(self, *, invalid_actions: Bool[Array, "node action"]) -> UnbatchedTree:
        """Return the same concrete tree with a lane-local action mask."""

        ...


@chex.dataclass(frozen=True)
class Tree:
    """A compact JAX tree with state-dependent integer payloads.

    ``edge_payload`` is structural count ``R`` while the corresponding edge
    outcome is ``NO_OUTCOME`` and categorical distance otherwise.
    ``node_payload`` similarly stores total structural support ``n_down`` for
    unresolved nodes and categorical distance for solved nodes.  Outcome tags
    are therefore the mandatory discriminants for both arrays.

    ``edge_alpha`` contains the network Q fallback until repaired, then the
    latest unresolved edge message. ``node_value_priors`` is fixed and
    ``node_value_alpha`` is the mutable repaired cache.
    """

    parents: Int32[Array, "batch node"]
    children_index: Int32[Array, "batch node action"]
    node_to_play: Int32[Array, "batch node"]
    node_categorical_outcome: Int8[Array, "batch node"]
    node_payload: Int32[Array, "batch node"]
    edge_categorical_outcome: Int8[Array, "batch node action"]
    edge_payload: Int32[Array, "batch node action"]
    node_value_priors: Float[Array, "batch node outcome"]
    node_value_alpha: Float[Array, "batch node outcome"]
    edge_alpha: Float[Array, "batch node action outcome"]
    invalid_actions: Bool[Array, "batch node action"]
    # Instrumentation only: these counters are never consumed by selection,
    # expansion, posterior repair, or action commitment.
    simulation_active_count: Int32[Array, "batch"]
    executed_simulation_call_count: Int32[Array, "batch"]
    embeddings: StoredRecurrentState

    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1

    @property
    def num_actions(self) -> int:
        return self.children_index.shape[-1]

    @property
    def num_simulations(self) -> int:
        return self.children_index.shape[1] - 1

    @property
    def root_action_alpha(self) -> Float[Array, "batch action outcome"]:
        """Return fallback-aware effective action alphas at the root."""

        return self.summary().alpha

    @property
    def root_stored_edge_alpha(self) -> Float[Array, "batch action outcome"]:
        """Return the root's physical edge-alpha slots without fallbacks."""

        return self.edge_alpha[:, self.ROOT_INDEX]

    @property
    def root_edge_payload(self) -> Int32[Array, "batch action"]:
        return self.edge_payload[:, self.ROOT_INDEX]

    @property
    def root_value_alpha(self) -> Float[Array, "batch outcome"]:
        return self.node_value_alpha[:, self.ROOT_INDEX]

    @property
    def root_invalid_actions(self) -> Bool[Array, "batch action"]:
        return self.invalid_actions[:, self.ROOT_INDEX]

    @property
    def searchable_actions(self) -> Bool[Array, "batch node action"]:
        """Actions that are legal and do not have a categorical outcome."""

        return ~self.invalid_actions & (
            self.edge_categorical_outcome == int(NO_OUTCOME)
        )

    def summary(self) -> SearchSummary:
        root = self.ROOT_INDEX
        edge_outcome = self.edge_categorical_outcome[:, root]
        unresolved = edge_outcome == int(NO_OUTCOME)
        edge_payload = self.edge_payload[:, root]
        counts = jnp.where(unresolved, edge_payload, 0)

        child_index = self.children_index[:, root]
        visited = child_index != self.UNVISITED
        safe_child = jnp.where(visited, child_index, root)
        batch = jnp.arange(self.parents.shape[0])[:, None]
        stored = self.edge_alpha[:, root]
        child_prior = self.node_value_priors[batch, safe_child]
        child_player = self.node_to_play[batch, safe_child]
        root_player = self.node_to_play[:, root, None]
        child_prior = align_outcome(child_prior, child_player, root_player)
        fallback = jnp.where(visited[..., None], child_prior, stored)
        use_stored = ~unresolved | (counts > 0)
        alpha = jnp.where(use_stored[..., None], stored, fallback)

        edge_distance = jnp.where(unresolved, jnp.asarray(int(NO_DISTANCE), dtype=jnp.int32), edge_payload)
        node_outcome = self.node_categorical_outcome[:, root]
        node_distance = jnp.where(node_outcome == int(NO_OUTCOME), jnp.asarray(int(NO_DISTANCE), dtype=jnp.int32), self.node_payload[:, root])
        return SearchSummary(
            visit_counts=counts.astype(stored.dtype),
            alpha=alpha,
            value_alpha=self.node_value_alpha[:, root],
            q_categorical_outcome=edge_outcome,
            q_categorical_distance=edge_distance,
            v_categorical_outcome=node_outcome,
            v_categorical_distance=node_distance,
        )


def instantiate_tree_from_root(root: RootFnOutput, num_simulations: int, root_invalid_actions: Bool[Array, "batch action"]) -> Tree:
    """Allocate compact fixed-capacity storage and initialize the root."""

    chex.assert_rank(root.prior_logits, 2)
    batch_size, num_actions = root.prior_logits.shape
    num_outcomes = root.action_values.shape[-1]
    chex.assert_shape(root.value, (batch_size, num_outcomes))
    chex.assert_shape(root.action_values, (batch_size, num_actions, num_outcomes))
    chex.assert_shape(root.terminal_outcome, (batch_size,))
    chex.assert_shape(root_invalid_actions, (batch_size, num_actions))
    num_nodes = num_simulations + 1
    batch_node = (batch_size, num_nodes)
    batch_node_action = (batch_size, num_nodes, num_actions)

    def allocate_embedding(value: Shaped[Array, "batch *embedding_axes"]) -> Shaped[Array, "batch node *embedding_axes"]:
        table = jnp.zeros((batch_size, num_nodes, *value.shape[1:]), dtype=value.dtype)
        return table.at[:, Tree.ROOT_INDEX].set(value)

    edge_alpha = jnp.ones((*batch_node_action, num_outcomes), dtype=root.action_values.dtype).at[:, Tree.ROOT_INDEX].set(root.action_values)
    node_value_alpha = jnp.ones((*batch_node, num_outcomes), dtype=root.value.dtype).at[:, Tree.ROOT_INDEX].set(root.value)
    return Tree(
        parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        children_index=jnp.full(batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
        node_to_play=jnp.zeros(batch_node, dtype=root.to_play.dtype).at[:, Tree.ROOT_INDEX].set(root.to_play),
        node_categorical_outcome=jnp.full(batch_node, int(NO_OUTCOME), dtype=jnp.int8).at[:, Tree.ROOT_INDEX].set(root.terminal_outcome.astype(jnp.int8)),
        node_payload=jnp.zeros(batch_node, dtype=jnp.int32),
        edge_categorical_outcome=jnp.full(batch_node_action, int(NO_OUTCOME), dtype=jnp.int8),
        edge_payload=jnp.zeros(batch_node_action, dtype=jnp.int32),
        node_value_priors=node_value_alpha,
        node_value_alpha=node_value_alpha,
        edge_alpha=edge_alpha,
        invalid_actions=jnp.ones(batch_node_action, dtype=bool).at[:, Tree.ROOT_INDEX].set(root_invalid_actions),
        simulation_active_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        executed_simulation_call_count=jnp.zeros(
            (batch_size,),
            dtype=jnp.int32,
        ),
        embeddings=jax.tree.map(allocate_embedding, root.embedding),
    )
