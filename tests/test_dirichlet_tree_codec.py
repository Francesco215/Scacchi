import numpy as np
import pytest

from scacchi.dirichlet_tree.codec import decode_node, encode_node
from scacchi.dirichlet_tree.types import NodeBlob, StateKey


def _node():
    return NodeBlob(
        key=StateKey((1, 2, 3, 4)),
        current_player=1,
        legal_actions=np.array([3, 1], dtype=np.uint32),
        value_alpha=np.array([1.5, 2.5, 3.5], dtype=np.float32),
        policy_logits=np.array([0.25, -1.5], dtype=np.float16),
        q_alpha=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        edge_base_alpha=np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32),
        edge_evidence_E=np.array([[0.5, 0.0, 1.0], [2.0, 0.0, 0.25]], dtype=np.float32),
        child_keys=np.array([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=np.uint32),
        visits=np.array([7, 9], dtype=np.uint32),
        terminal_outcome=-1,
        status=1,
        game_id=42,
        model_id=123456789,
        dirty_version=99,
    )


def test_node_codec_round_trips_sparse_arrays_and_integer_fields():
    node = _node()

    decoded = decode_node(encode_node(node))

    assert decoded.key == node.key
    assert decoded.current_player == node.current_player
    assert decoded.status == node.status
    assert decoded.game_id == node.game_id
    assert decoded.model_id == node.model_id
    assert decoded.dirty_version == node.dirty_version
    assert decoded.terminal_outcome == node.terminal_outcome
    assert decoded.legal_actions.tolist() == [3, 1]
    assert np.array_equal(decoded.child_keys, node.child_keys)
    assert np.array_equal(decoded.visits, node.visits)
    assert np.allclose(decoded.value_alpha, node.value_alpha)
    assert np.allclose(decoded.policy_logits, node.policy_logits)
    assert np.allclose(decoded.q_alpha, node.q_alpha)
    assert np.allclose(decoded.edge_base_alpha, node.edge_base_alpha)
    assert np.allclose(decoded.edge_evidence_E, node.edge_evidence_E)


def test_node_codec_rejects_version_mismatch():
    blob = bytearray(encode_node(_node()))
    blob[4:6] = (999).to_bytes(2, "little")

    with pytest.raises(ValueError, match="version"):
        decode_node(bytes(blob))
