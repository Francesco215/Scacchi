from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Protocol

from .codec import decode_node, encode_node
from .types import EVAL_EXPANDING, EVAL_INFLIGHT, NodeBlob, StateKey


@dataclass(slots=True)
class StoreStats:
    cache_hits: int = 0
    cache_misses: int = 0
    redis_mget: int = 0
    redis_mset: int = 0
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
        *,
        ttl_ms: int = 30000,
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
        *,
        ttl_ms: int = 30000,
    ) -> ClaimManyResult:
        del ttl_ms
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


class RedisNodeStore:
    def __init__(
        self,
        redis_client,
        *,
        namespace: str,
        cache_size: int = 100_000,
    ) -> None:
        self.redis = redis_client
        self.namespace = namespace.rstrip(":")
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[StateKey, NodeBlob] = OrderedDict()
        self.dirty_keys: set[StateKey] = set()
        self.stats = StoreStats()

    def redis_key(self, key: StateKey) -> str:
        return f"{self.namespace}:node:{key.redis_hex}"

    def get_many(self, keys: Iterable[StateKey]) -> dict[StateKey, NodeBlob | None]:
        unique = _dedupe(keys)
        result: dict[StateKey, NodeBlob | None] = {}
        misses: list[StateKey] = []
        for key in unique:
            node = self.cache.get(key)
            if node is None:
                misses.append(key)
                self.stats.cache_misses += 1
            else:
                self.cache.move_to_end(key)
                result[key] = node
                self.stats.cache_hits += 1
        if misses:
            self.stats.redis_mget += 1
            raw_values = self.redis.mget([self.redis_key(key) for key in misses])
            for key, raw in zip(misses, raw_values, strict=True):
                if raw is None:
                    result[key] = None
                    continue
                node = decode_node(raw)
                self._cache_put(node)
                result[key] = node
        return result

    def put_many(self, nodes: Iterable[NodeBlob]) -> None:
        node_list = list(nodes)
        if not node_list:
            return
        mapping = {self.redis_key(node.key): encode_node(node) for node in node_list}
        self.redis.mset(mapping)
        self.stats.redis_mset += 1
        for node in node_list:
            self._cache_put(node)
            self.dirty_keys.discard(node.key)

    def claim_many_inflight(
        self,
        keys: Iterable[StateKey],
        *,
        ttl_ms: int = 30000,
    ) -> ClaimManyResult:
        unique = _dedupe(keys)
        claimed: list[StateKey] = []
        expanded: list[StateKey] = []
        inflight: list[StateKey] = []
        uncached: list[tuple[StateKey, str, bytes]] = []
        for key in unique:
            cached = self.cache.get(key)
            if cached is not None:
                if cached.status in (EVAL_INFLIGHT, EVAL_EXPANDING):
                    inflight.append(key)
                else:
                    expanded.append(key)
                continue
            placeholder = NodeBlob.inflight_node(key=key)
            uncached.append((key, self.redis_key(key), encode_node(placeholder)))

        if uncached:
            pipe = self.redis.pipeline(transaction=False)
            for _, redis_key, encoded in uncached:
                pipe.set(redis_key, encoded, nx=True, px=int(ttl_ms))
            set_results = pipe.execute()
            misses: list[StateKey] = []
            for (key, _, encoded), ok in zip(uncached, set_results, strict=True):
                if ok:
                    claimed.append(key)
                    self._cache_put(decode_node(encoded))
                else:
                    misses.append(key)
            if misses:
                existing = self.get_many(misses)
                for key in misses:
                    node = existing[key]
                    if node is not None and node.status in (EVAL_INFLIGHT, EVAL_EXPANDING):
                        inflight.append(key)
                    else:
                        expanded.append(key)

        self.stats.nodes_claimed += len(claimed)
        self.stats.nodes_inflight += len(inflight)
        return ClaimManyResult(tuple(claimed), tuple(expanded), tuple(inflight))

    def mark_dirty(self, key: StateKey) -> None:
        self.dirty_keys.add(key)

    def flush_dirty(self) -> None:
        dirty = [self.cache[key] for key in self.dirty_keys if key in self.cache]
        if not dirty:
            return
        self.put_many(dirty)
        self.stats.dirty_flushes += 1
        self.dirty_keys.clear()

    def _cache_put(self, node: NodeBlob) -> None:
        self.cache[node.key] = node
        self.cache.move_to_end(node.key)
        while len(self.cache) > self.cache_size:
            key, _ = self.cache.popitem(last=False)
            self.dirty_keys.discard(key)


def _dedupe(keys: Iterable[StateKey]) -> list[StateKey]:
    return list(dict.fromkeys(keys))
