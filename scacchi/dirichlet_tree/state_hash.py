from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .types import KEY_WORDS, StateKey


_INIT = jnp.array(
    [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
    dtype=jnp.uint32,
)
_C = jnp.array(
    [0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F],
    dtype=jnp.uint32,
)


def canonical_state_key(state: Any) -> jax.Array:
    if _looks_like_hex_state(state):
        return _hex_state_key(state)
    words = []
    for leaf in jax.tree_util.tree_leaves(state):
        arr = jnp.asarray(leaf)
        words.append(_metadata_words(arr))
        flat = _leaf_words(arr)
        if flat.size:
            words.append(flat)
    if not words:
        payload = jnp.zeros((1,), dtype=jnp.uint32)
    else:
        payload = jnp.concatenate(words, axis=0)
    return _hash_words(payload)


def _looks_like_hex_state(state: Any) -> bool:
    game_state = getattr(state, "_x", None)
    return (
        game_state is not None
        and hasattr(game_state, "board")
        and hasattr(game_state, "step_count")
        and hasattr(game_state, "terminated")
        and hasattr(state, "_player_order")
        and hasattr(state, "current_player")
    )


def _hex_state_key(state: Any) -> jax.Array:
    game_state = state._x
    payload = jnp.concatenate(
        [
            jnp.asarray(game_state.board, dtype=jnp.uint32).reshape((-1,)),
            jnp.asarray(state.current_player, dtype=jnp.uint32).reshape((1,)),
            jnp.asarray(state._player_order, dtype=jnp.uint32).reshape((-1,)),
            jnp.asarray(game_state.step_count, dtype=jnp.uint32).reshape((1,)),
            jnp.asarray(game_state.terminated, dtype=jnp.uint32).reshape((1,)),
        ],
        axis=0,
    )
    return _hash_words(payload)


def batched_state_key_fn(state: Any) -> jax.Array:
    return jax.jit(jax.vmap(canonical_state_key))(state)


def state_keys_to_host(keys: Any) -> tuple[StateKey, ...]:
    arr = jax.device_get(keys)
    return tuple(StateKey.from_array(arr[ix]) for ix in range(arr.shape[0]))


def _metadata_words(arr: jax.Array) -> jax.Array:
    shape_words = jnp.asarray(arr.shape, dtype=jnp.uint32)
    dtype_word = jnp.asarray(_dtype_tag(arr.dtype), dtype=jnp.uint32).reshape((1,))
    ndim_word = jnp.asarray(arr.ndim, dtype=jnp.uint32).reshape((1,))
    return jnp.concatenate([dtype_word, ndim_word, shape_words], axis=0)


def _leaf_words(arr: jax.Array) -> jax.Array:
    arr = jnp.ravel(arr)
    if arr.dtype == jnp.bool_:
        return arr.astype(jnp.uint32)
    if jnp.issubdtype(arr.dtype, jnp.floating):
        return jax.lax.bitcast_convert_type(arr.astype(jnp.float32), jnp.uint32)
    return arr.astype(jnp.uint32)


def _hash_words(words: jax.Array) -> jax.Array:
    words = words.astype(jnp.uint32)

    def body(carry: jax.Array, word: jax.Array) -> tuple[jax.Array, None]:
        h = carry ^ (word + _C)
        h = _rotl32(h, jnp.array([13, 17, 7, 19], dtype=jnp.uint32))
        h = h * jnp.array([0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F, 0x165667B1], dtype=jnp.uint32)
        h = h ^ (h >> jnp.uint32(16))
        mixed_word = word * jnp.uint32(0x9E3779B9)
        return h + jnp.roll(h, 1) + mixed_word, None

    h, _ = jax.lax.scan(body, _INIT + jnp.asarray(words.size, dtype=jnp.uint32), words)
    return jnp.where(jnp.all(h == 0), jnp.ones((KEY_WORDS,), dtype=jnp.uint32), h)


def _rotl32(x: jax.Array, bits: jax.Array) -> jax.Array:
    return (x << bits) | (x >> (jnp.uint32(32) - bits))


def _dtype_tag(dtype) -> int:
    name = jnp.dtype(dtype).name
    tags = {
        "bool": 1,
        "int8": 2,
        "int16": 3,
        "int32": 4,
        "int64": 5,
        "uint8": 6,
        "uint16": 7,
        "uint32": 8,
        "uint64": 9,
        "float16": 10,
        "bfloat16": 11,
        "float32": 12,
        "float64": 13,
    }
    return tags.get(name, 255)
