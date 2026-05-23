from __future__ import annotations

import argparse
import time

import numpy as np

from scacchi.dirichlet_tree.store import RedisNodeStore
from scacchi.dirichlet_tree.types import NodeBlob, StateKey


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.url:
        import redis

        client = redis.Redis.from_url(args.url)
    else:
        import fakeredis

        client = fakeredis.FakeRedis()

    store = RedisNodeStore(client, namespace="dqaz:bench")
    nodes = [
        NodeBlob.expanded_node(
            key=StateKey((ix, ix + 1, ix + 2, ix + 3)),
            current_player=0,
            legal_action_mask=np.ones((16,), dtype=bool),
            value_alpha=np.ones((3,), dtype=np.float32),
            policy_logits=np.zeros((16,), dtype=np.float32),
            q_alpha=np.ones((16, 3), dtype=np.float32),
        )
        for ix in range(args.batch)
    ]
    keys = [node.key for node in nodes]

    start = time.perf_counter()
    for _ in range(args.iters):
        store.put_many(nodes)
    mset_elapsed = time.perf_counter() - start
    store.cache.clear()
    start = time.perf_counter()
    for _ in range(args.iters):
        store.get_many(keys)
        store.cache.clear()
    mget_elapsed = time.perf_counter() - start
    print(f"mset_nodes_per_s={args.batch * args.iters / mset_elapsed:.1f}")
    print(f"mget_nodes_per_s={args.batch * args.iters / mget_elapsed:.1f}")
    print(f"redis_mset_round_trips={store.stats.redis_mset}")
    print(f"redis_mget_round_trips={store.stats.redis_mget}")


if __name__ == "__main__":
    main()
