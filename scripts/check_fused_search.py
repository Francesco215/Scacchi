"""Compare the fused search (native_binned.search) with the JAX search, array by array.

Both run from the same positions with the same keys and the same network. The
reference is ``dirichlet_mctx.search.search`` with ``binned_search.action_selection``
and ``binned_search.update_posterior(kernel="cuda")``, i.e. the search that
``backup_kernel=cuda`` runs today. Every tree array is compared bitwise; the
script reports, per array, how many lanes differ and the first one.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL,+CONVOLUTION --xla_gpu_graph_min_graph_size=1",
)

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(REPO / "scripts"))
from profile_search import load_config, random_positions  # noqa: E402

FIELDS = [
    "parents", "children_index", "node_to_play", "node_categorical_outcome", "node_payload",
    "edge_categorical_outcome", "edge_payload", "node_value_priors", "node_value_alpha",
    "edge_alpha", "invalid_actions",
]


def build(config_name: str, batch: int, checkpoint: str | None, simulations: int | None):
    from scacchi.checkpoint import from_pretrained
    from scacchi.envs import make_env
    from scacchi.network import build_model

    extra = [] if simulations is None else [f"selfplay.search.binned_thompson.num_simulations={simulations}",
                                            f"selfplay.search.binned_thompson.max_depth={simulations}"]
    config = load_config(config_name, batch, extra)
    env = make_env(config.env.id, config.env.board_size)
    if checkpoint:
        model = from_pretrained(checkpoint, env, rngs=nnx.Rngs(0))
    else:
        # Untrained output layers are zero, so every belief would be the same;
        # perturb all parameters so the trees are not degenerate.
        model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(0))
        state = nnx.state(model, nnx.Param)
        leaves, treedef = jax.tree.flatten(state)
        keys = jax.random.split(jax.random.PRNGKey(17), len(leaves))
        leaves = [leaf + 0.05 * jax.random.normal(key, leaf.shape, leaf.dtype) for leaf, key in zip(leaves, keys)]
        nnx.update(model, jax.tree.unflatten(treedef, leaves))
    return config, env, model


def make_trees(env, model, search_config, cohorts=1):
    from scacchi import binned_search
    from scacchi.dirichlet_mctx import RootFnOutput
    from scacchi.dirichlet_mctx.search import search as tree_search
    from scacchi.dirichlet_q_search import make_dirichlet_expand_fn, terminal_outcome_from_reward
    from scacchi.native_binned.search import fused_search
    from scacchi.play_search import make_evaluator

    def roots(model, state):
        evaluator = make_evaluator(model)
        prediction = evaluator(state.observation)
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), state.current_player]
        root = RootFnOutput(
            prior_logits=prediction.logits, value=prediction.alpha_v, action_values=prediction.alpha_q,
            embedding=state, terminal_outcome=jnp.where(state.terminated, terminal_outcome_from_reward(reward, 2), jnp.int8(-1)),
            to_play=state.current_player,
        )
        return evaluator, prediction, root

    @nnx.jit
    def reference(model, state, key):
        evaluator, prediction, root = roots(model, state)
        expand = make_dirichlet_expand_fn(env, evaluator, num_outcomes=2)
        repair = functools.partial(binned_search.update_posterior, points=prediction.support, kernel="cuda")
        return tree_search(
            params=(), rng_key=key, root=root, recurrent_fn=expand, action_selection_fn=binned_search.action_selection,
            posterior_update=repair, num_simulations=search_config.num_simulations, max_depth=search_config.max_depth,
            invalid_actions=~state.legal_action_mask, num_outcomes=2,
        )

    @nnx.jit
    def fused(model, state, key):
        evaluator, prediction, root = roots(model, state)
        return fused_search(evaluator, state, root, key, points=prediction.support,
                            num_simulations=search_config.num_simulations, max_depth=search_config.max_depth,
                            cohorts=cohorts)

    return reference, fused


def compare(a, b):
    report = {}
    for name in FIELDS:
        x = np.asarray(getattr(a, name))
        y = np.asarray(getattr(b, name))
        same = (x.view(np.uint32) == y.view(np.uint32)) if x.dtype.kind == "f" else (x == y)
        lanes = ~same.reshape(same.shape[0], -1).all(axis=1)
        entry = {"lanes_differ": int(lanes.sum())}
        if lanes.any():
            lane = int(np.argmax(lanes))
            where = tuple(np.argwhere(~same[lane])[0].tolist())
            entry.update(first_lane=lane, at=list(where), reference=x[lane][where].item(), fused=y[lane][where].item())
            if x.dtype.kind == "f":
                entry["max_abs_diff"] = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        report[name] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="hex7")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--plies", type=int, nargs="+", default=[0, 6, 14, 24, 34])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cohorts", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config, env, model = build(args.config_name, args.batch, args.checkpoint, args.simulations)
    reference, fused = make_trees(env, model, config.selfplay.search.binned_thompson, args.cohorts)
    positions = jax.jit(random_positions, static_argnums=(0, 1, 2))
    results = []
    all_same = True
    for plies in args.plies:
        states = positions(env, args.batch, plies, jax.random.PRNGKey(1000 + plies))
        for seed in range(args.seeds):
            key = jax.random.PRNGKey(seed)
            started = time.perf_counter()
            a = jax.block_until_ready(reference(model, states, key))
            t_ref = time.perf_counter() - started
            started = time.perf_counter()
            b = jax.block_until_ready(fused(model, states, key))
            t_fused = time.perf_counter() - started
            report = compare(a, b)
            differ = {k: v for k, v in report.items() if v["lanes_differ"]}
            all_same &= not differ
            line = {"plies": plies, "seed": seed, "reference_s": round(t_ref, 3), "fused_s": round(t_fused, 3),
                    "differ": differ or "none"}
            print(json.dumps(line), flush=True)
            results.append(line)
    print("ALL IDENTICAL" if all_same else "MISMATCH")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
