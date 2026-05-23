from __future__ import annotations

import argparse
import time

import numpy as np

from scacchi.dirichlet_tree.codec import decode_node, encode_node
from scacchi.dirichlet_tree.types import NodeBlob, StateKey


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100_000)
    parser.add_argument("--actions", type=int, default=81)
    args = parser.parse_args()

    node = NodeBlob.expanded_node(
        key=StateKey((1, 2, 3, 4)),
        current_player=0,
        legal_action_mask=np.ones((args.actions,), dtype=bool),
        value_alpha=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        policy_logits=np.zeros((args.actions,), dtype=np.float32),
        q_alpha=np.ones((args.actions, 3), dtype=np.float32),
    )
    start = time.perf_counter()
    blobs = [encode_node(node) for _ in range(args.iters)]
    encode_s = time.perf_counter() - start
    start = time.perf_counter()
    for blob in blobs:
        decode_node(blob)
    decode_s = time.perf_counter() - start
    print(f"encode_nodes_per_s={args.iters / encode_s:.1f}")
    print(f"decode_nodes_per_s={args.iters / decode_s:.1f}")
    print(f"bytes_per_node={len(blobs[0])}")


if __name__ == "__main__":
    main()
