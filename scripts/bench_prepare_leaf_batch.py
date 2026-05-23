from __future__ import annotations

import argparse
import gc
import time
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from scacchi.dirichlet_tree.arena_search import BatchedPosteriorArenaSearch
from scacchi.dirichlet_tree.search import run_wavefront_posterior_tree_search, search_config_from_any
from scacchi.dirichlet_tree.store import InMemoryNodeStore, RedisNodeStore
from scacchi.posterior_tree import run_posterior_tree_search


class ToyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    terminated: jax.Array
    rewards: jax.Array


class ToyEnv:
    def step(self, state: ToyState, action: jax.Array) -> ToyState:
        del action
        return ToyState(
            observation=state.observation + 1.0,
            legal_action_mask=jnp.array([True], dtype=jnp.bool_),
            current_player=1 - state.current_player,
            terminated=jnp.array(False),
            rewards=jnp.array([0.0, 0.0], dtype=jnp.float32),
        )


class TimedEvaluator:
    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.calls: list[dict[str, float | int]] = []
        self.root_done_s: float | None = None

    def __call__(self, obs: jax.Array):
        entry_s = time.perf_counter()
        batch = int(obs.shape[0])
        if self.calls:
            self.calls.append(
                {
                    "batch": batch,
                    "entry_s": entry_s,
                    "exit_s": entry_s,
                    "eval_s": 0.0,
                    "prep_since_root_s": (
                        -1.0 if self.root_done_s is None else entry_s - self.root_done_s
                    ),
                }
            )
            raise LeafBatchReady
        logits = jnp.zeros((batch, self.num_actions), dtype=jnp.float32)
        alpha_v = jnp.tile(jnp.array([[1.0, 1.0, 3.0]], dtype=jnp.float32), (batch, 1))
        alpha_q = jnp.ones((batch, self.num_actions, 3), dtype=jnp.float32)
        jax.block_until_ready((logits, alpha_v, alpha_q))
        exit_s = time.perf_counter()
        if not self.calls:
            self.root_done_s = exit_s
        prep_s = None if self.root_done_s is None else entry_s - self.root_done_s
        self.calls.append(
            {
                "batch": batch,
                "entry_s": entry_s,
                "exit_s": exit_s,
                "eval_s": exit_s - entry_s,
                "prep_since_root_s": -1.0 if prep_s is None else prep_s,
            }
        )
        return logits, alpha_v, alpha_q


class LeafBatchReady(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=262_144)
    parser.add_argument(
        "--policy",
        choices=["posterior_tree", "posterior_tree_wavefront", "both"],
        default="both",
    )
    parser.add_argument("--warmup-batch", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--store",
        choices=["arena", "inmemory", "fakeredis", "redis"],
        default="arena",
        help="Backend for posterior_tree_wavefront; ignored for posterior_tree.",
    )
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    args = parser.parse_args()

    policies = (
        ["posterior_tree", "posterior_tree_wavefront"]
        if args.policy == "both"
        else [args.policy]
    )

    for policy in policies:
        if args.warmup_batch > 0:
            _run(
                policy,
                _make_roots_for_policy(policy, args.store, args.warmup_batch),
                quiet=True,
                store_backend=args.store,
                redis_url=args.redis_url,
                run_id="warmup",
            )
            gc.collect()
        for repeat in range(args.repeats):
            result = _run(
                policy,
                _make_roots_for_policy(policy, args.store, args.batch),
                quiet=False,
                store_backend=args.store,
                redis_url=args.redis_url,
                run_id=f"batch{args.batch}:repeat{repeat + 1}",
            )
            print(
                "policy={policy} repeat={repeat} batch={batch} total_to_leaf_s={total:.6f} "
                "prep_to_leaf_s={prep:.6f} leaf_batch={leaf_batch} calls={calls} "
                "store={store} cache_hits={cache_hits} cache_misses={cache_misses} "
                "redis_mget={redis_mget} redis_mset={redis_mset} claimed={claimed}".format(
                    policy=policy,
                    repeat=repeat + 1,
                    batch=args.batch,
                    total=result["total_s"],
                    prep=result["prep_to_leaf_s"],
                    leaf_batch=result["leaf_batch"],
                    calls=result["calls"],
                    store=args.store if policy == "posterior_tree_wavefront" else "object_tree",
                    cache_hits=result["cache_hits"],
                    cache_misses=result["cache_misses"],
                    redis_mget=result["redis_mget"],
                    redis_mset=result["redis_mset"],
                    claimed=result["claimed"],
                ),
                flush=True,
            )
            gc.collect()


def _run(
    policy: str,
    root_states,
    *,
    quiet: bool,
    store_backend: str,
    redis_url: str,
    run_id: str,
) -> dict[str, float | int]:
    batch_size = _root_count(root_states)
    evaluator = TimedEvaluator(num_actions=1)
    config = SimpleNamespace(
        search_policy=policy,
        num_simulations=1,
        num_search_blocks=1,
        inflight_limit=1,
        search_eval_batch_size=batch_size,
        c_leaf=1.0,
        c_terminal=8.0,
        c_state=0.1,
        c_value_search=1.0,
        policy_mc_samples=1,
        backup_mc_samples=1,
        selfplay_action_source="scalar_q_argmax",
        wavefront_num_lanes_per_root=1,
        wavefront_max_depth=8,
        wavefront_final_action_mode="argmax_q_mean",
    )
    store = (
        _build_store(store_backend, redis_url, run_id, batch_size)
        if policy == "posterior_tree_wavefront" and store_backend != "arena"
        else None
    )
    start_s = time.perf_counter()
    try:
        if policy == "posterior_tree_wavefront" and store_backend == "arena" and not isinstance(root_states, list):
            search = BatchedPosteriorArenaSearch(env=ToyEnv(), rng_key=jax.random.PRNGKey(0))
            output = search.search_state_batch(
                root_states,
                evaluator,
                search_config_from_any(config, num_roots=batch_size),
            )
        elif policy == "posterior_tree_wavefront":
            output = run_wavefront_posterior_tree_search(
                env=ToyEnv(),
                root_states=root_states,
                leaf_evaluator=evaluator,
                rng_key=jax.random.PRNGKey(0),
                config=config,
                store=store,
            )
        else:
            output = run_posterior_tree_search(
                env=ToyEnv(),
                root_states=root_states,
                leaf_evaluator=evaluator,
                rng_key=jax.random.PRNGKey(0),
                config=config,
            )
        jax.block_until_ready(output.action)
    except LeafBatchReady:
        pass
    total_s = time.perf_counter() - start_s
    leaf_call = evaluator.calls[1] if len(evaluator.calls) > 1 else None
    if not quiet and leaf_call is None:
        raise RuntimeError(f"{policy} did not produce a leaf-eval batch")
    return {
        "total_s": total_s,
        "prep_to_leaf_s": -1.0 if leaf_call is None else float(leaf_call["prep_since_root_s"]),
        "leaf_batch": 0 if leaf_call is None else int(leaf_call["batch"]),
        "calls": len(evaluator.calls),
        "cache_hits": 0 if store is None else store.stats.cache_hits,
        "cache_misses": 0 if store is None else store.stats.cache_misses,
        "redis_mget": 0 if store is None else store.stats.redis_mget,
        "redis_mset": 0 if store is None else store.stats.redis_mset,
        "claimed": 0 if store is None else store.stats.nodes_claimed,
    }


def _build_store(store_backend: str, redis_url: str, run_id: str, batch: int):
    if store_backend == "arena":
        return None
    if store_backend == "inmemory":
        return InMemoryNodeStore()
    if store_backend == "fakeredis":
        import fakeredis

        return RedisNodeStore(
            fakeredis.FakeRedis(),
            namespace=f"dqaz:bench_prepare:{run_id}",
            cache_size=max(1024, batch * 4),
        )
    if store_backend == "redis":
        import redis

        client = redis.Redis.from_url(redis_url)
        client.ping()
        return RedisNodeStore(
            client,
            namespace=f"dqaz:bench_prepare:{run_id}:{time.time_ns()}",
            cache_size=max(1024, batch * 4),
        )
    raise ValueError(f"unknown store backend: {store_backend}")


def _make_roots_for_policy(policy: str, store_backend: str, batch: int):
    if policy == "posterior_tree_wavefront" and store_backend == "arena":
        return _make_root_state_batch(batch)
    return _make_root_states(batch)


def _root_count(root_states) -> int:
    if isinstance(root_states, list):
        return len(root_states)
    return int(root_states.current_player.shape[0])


def _make_root_state_batch(batch: int) -> ToyState:
    return ToyState(
        observation=jnp.arange(batch, dtype=jnp.float32).reshape((batch, 1)),
        legal_action_mask=jnp.ones((batch, 1), dtype=jnp.bool_),
        current_player=jnp.zeros((batch,), dtype=jnp.int32),
        terminated=jnp.zeros((batch,), dtype=jnp.bool_),
        rewards=jnp.zeros((batch, 2), dtype=jnp.float32),
    )


def _make_root_states(batch: int) -> list[ToyState]:
    state_batch = _make_root_state_batch(batch)
    return [
        ToyState(
            observation=state_batch.observation[ix],
            legal_action_mask=state_batch.legal_action_mask[ix],
            current_player=state_batch.current_player[ix],
            terminated=state_batch.terminated[ix],
            rewards=state_batch.rewards[ix],
        )
        for ix in range(batch)
    ]


if __name__ == "__main__":
    main()
