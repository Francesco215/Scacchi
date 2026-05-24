from __future__ import annotations

import struct

import numpy as np

from .types import KEY_WORDS, NodeBlob, StateKey


MAGIC = b"DTN1"
VERSION = 2
_HEADER = struct.Struct("<4sHBHQ4IbbHHQ4IiiiQBQQ")


def encode_node(node: NodeBlob) -> bytes:
    num_actions = int(node.legal_actions.shape[0])
    num_outcomes = int(node.value_alpha.shape[0])
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        int(node.status),
        int(node.game_id),
        int(node.model_id),
        *node.key.words,
        int(node.current_player),
        int(node.terminal_outcome),
        num_actions,
        num_outcomes,
        int(node.dirty_version),
        *node.parent_key.words,
        int(node.parent_node_id),
        int(node.parent_action),
        int(node.depth),
        int(node.downstream_eval_count),
        int(node.value_cache_status),
        int(node.value_cache_version),
        int(node.edge_epoch),
    )
    parts = [
        header,
        _le(node.value_alpha, "<f4").tobytes(order="C"),
        _le(node.value_cache_C, "<f4").tobytes(order="C"),
        _le(node.legal_actions, "<u4").tobytes(order="C"),
        _le(node.policy_logits, "<f2").tobytes(order="C"),
        _le(node.q_alpha, "<f4").tobytes(order="C"),
        _le(node.edge_base_alpha, "<f4").tobytes(order="C"),
        _le(node.edge_B, "<f4").tobytes(order="C"),
        _le(node.edge_has_post, "u1").tobytes(order="C"),
        _le(node.edge_eval_count_R, "<u4").tobytes(order="C"),
        _le(node.edge_version, "<u4").tobytes(order="C"),
        _le(node.edge_child_cache_version, "<i8").tobytes(order="C"),
        _le(node.child_keys, "<u4").tobytes(order="C"),
        _le(node.visits, "<u4").tobytes(order="C"),
        _le(node.pi_search, "<f2").tobytes(order="C"),
        _le(node.state_summary_alpha, "<f4").tobytes(order="C"),
    ]
    return b"".join(parts)


def decode_node(blob: bytes) -> NodeBlob:
    if len(blob) < _HEADER.size:
        raise ValueError("node blob is too short")
    (
        magic,
        version,
        status,
        game_id,
        model_id,
        w0,
        w1,
        w2,
        w3,
        current_player,
        terminal_outcome,
        num_actions,
        num_outcomes,
        dirty_version,
        pw0,
        pw1,
        pw2,
        pw3,
        parent_node_id,
        parent_action,
        depth,
        downstream_eval_count,
        value_cache_status,
        value_cache_version,
        edge_epoch,
    ) = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"invalid node blob magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported node blob version: {version}")

    offset = _HEADER.size
    value_alpha, offset = _read(blob, offset, "<f4", (num_outcomes,))
    value_cache_C, offset = _read(blob, offset, "<f4", (num_outcomes,))
    legal_actions, offset = _read(blob, offset, "<u4", (num_actions,))
    policy_logits, offset = _read(blob, offset, "<f2", (num_actions,))
    q_alpha, offset = _read(blob, offset, "<f4", (num_actions, num_outcomes))
    edge_base_alpha, offset = _read(blob, offset, "<f4", (num_actions, num_outcomes))
    edge_B, offset = _read(blob, offset, "<f4", (num_actions, num_outcomes))
    edge_has_post, offset = _read(blob, offset, "u1", (num_actions,))
    edge_eval_count_R, offset = _read(blob, offset, "<u4", (num_actions,))
    edge_version, offset = _read(blob, offset, "<u4", (num_actions,))
    edge_child_cache_version, offset = _read(blob, offset, "<i8", (num_actions,))
    child_keys, offset = _read(blob, offset, "<u4", (num_actions, KEY_WORDS))
    visits, offset = _read(blob, offset, "<u4", (num_actions,))
    pi_search, offset = _read(blob, offset, "<f2", (num_actions,))
    state_summary_alpha, offset = _read(blob, offset, "<f4", (num_outcomes,))
    if offset != len(blob):
        raise ValueError("node blob has trailing bytes")

    return NodeBlob(
        key=StateKey((w0, w1, w2, w3)),
        current_player=current_player,
        legal_actions=legal_actions,
        value_alpha=value_alpha,
        policy_logits=policy_logits,
        q_alpha=q_alpha,
        edge_base_alpha=edge_base_alpha,
        child_keys=child_keys,
        visits=visits,
        edge_B=edge_B,
        edge_has_post=edge_has_post.astype(bool),
        edge_eval_count_R=edge_eval_count_R,
        edge_version=edge_version,
        edge_child_cache_version=edge_child_cache_version,
        terminal_outcome=terminal_outcome,
        status=status,
        game_id=game_id,
        model_id=model_id,
        dirty_version=dirty_version,
        parent_key=StateKey((pw0, pw1, pw2, pw3)),
        parent_node_id=parent_node_id,
        parent_action=parent_action,
        depth=depth,
        value_cache_C=value_cache_C,
        downstream_eval_count=downstream_eval_count,
        value_cache_status=value_cache_status,
        value_cache_version=value_cache_version,
        edge_epoch=edge_epoch,
        pi_search=pi_search,
        state_summary_alpha=state_summary_alpha,
    )


def _le(array: np.ndarray | None, dtype: str) -> np.ndarray:
    return np.asarray(array, dtype=np.dtype(dtype))


def _read(
    blob: bytes,
    offset: int,
    dtype: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    np_dtype = np.dtype(dtype)
    size = int(np.prod(shape, dtype=np.int64)) * np_dtype.itemsize
    end = offset + size
    if end > len(blob):
        raise ValueError("node blob ended while reading an array")
    arr = np.frombuffer(blob, dtype=np_dtype, count=int(np.prod(shape)), offset=offset)
    return arr.reshape(shape).copy(), end
