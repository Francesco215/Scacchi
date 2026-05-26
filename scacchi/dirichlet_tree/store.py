from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .types import EVAL_EXPANDING, EVAL_INFLIGHT, NodeBlob, StateKey


@dataclass(slots=True)
class StoreStats:
    cache_hits: int = 0
    cache_misses: int = 0
    nodes_claimed: int = 0
    nodes_inflight: int = 0
    dirty_flushes: int = 0


@dataclass(frozen=True, slots=True)
class ClaimManyResult:
    claimed: tuple[StateKey, ...]
    expanded: tuple[StateKey, ...]
    inflight: tuple[StateKey, ...]


class NodeStore(Protocol):
    stats: StoreStats

    def get_many(self, keys: Iterable[StateKey]) -> dict[StateKey, NodeBlob | None]:
        ...

    def put_many(self, nodes: Iterable[NodeBlob]) -> None:
        ...

    def claim_many_inflight(
        self,
        keys: Iterable[StateKey],
    ) -> ClaimManyResult:
        ...

    def mark_dirty(self, key: StateKey) -> None:
        ...

    def flush_dirty(self) -> None:
        ...


class InMemoryNodeStore:
    def __init__(self) -> None:
        self.nodes: dict[StateKey, NodeBlob] = {}
        self.dirty_keys: set[StateKey] = set()
        self.stats = StoreStats()

    def get_many(self, keys: Iterable[StateKey]) -> dict[StateKey, NodeBlob | None]:
        result = {}
        for key in _dedupe(keys):
            node = self.nodes.get(key)
            if node is None:
                self.stats.cache_misses += 1
            else:
                self.stats.cache_hits += 1
            result[key] = node
        return result

    def put_many(self, nodes: Iterable[NodeBlob]) -> None:
        for node in nodes:
            self.nodes[node.key] = node
            self.dirty_keys.discard(node.key)

    def claim_many_inflight(
        self,
        keys: Iterable[StateKey],
    ) -> ClaimManyResult:
        claimed: list[StateKey] = []
        expanded: list[StateKey] = []
        inflight: list[StateKey] = []
        for key in _dedupe(keys):
            node = self.nodes.get(key)
            if node is None:
                self.nodes[key] = NodeBlob.inflight_node(key=key)
                claimed.append(key)
            elif node.status in (EVAL_INFLIGHT, EVAL_EXPANDING):
                inflight.append(key)
            else:
                expanded.append(key)
        self.stats.nodes_claimed += len(claimed)
        self.stats.nodes_inflight += len(inflight)
        return ClaimManyResult(tuple(claimed), tuple(expanded), tuple(inflight))

    def mark_dirty(self, key: StateKey) -> None:
        self.dirty_keys.add(key)

    def flush_dirty(self) -> None:
        if self.dirty_keys:
            self.stats.dirty_flushes += 1
        self.dirty_keys.clear()


def _dedupe(keys: Iterable[StateKey]) -> list[StateKey]:
    return list(dict.fromkeys(keys))
