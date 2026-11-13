"""How much is an old search target worth, and do its bits say so?

A replay pool stores a position with the target that search produced under the
network of that time. This script measures, without any pool, what such a row
is worth to the current network, as a function of the target's age and of the
bits the pool would store for it.

It trains the usual loop without a pool and keeps a copy of the network after
every iteration. For a current network B and an older one A it then

1. plays positions with A and keeps A's search targets ``t_A`` (the stored row);
2. searches the same positions twice with B: ``t_B`` and ``t_B2`` (two
   independent fresh targets);
3. evaluates B on them: ``p``.

Every target pulls the network's outputs along ``p - t`` (the gradient of the
NLL with respect to the logits of each head). A step on the stored row lowers
the loss against a fresh target to first order by

    use(A) = <p - t_A, p - t_B>,

and a step on an independent fresh target by ``use(B2) = <p - t_B2, p - t_B>``;
search noise is independent between searches and averages out of both. The
ratio of their means is the worth of the stored row relative to a fresh search
of the same position, and ``use(A) < 0`` marks a row that pulls the wrong way.

The same quantities binned by ``bits = KL(t_A || p)`` show whether rows with
more bits carry more usable signal or only more noise.

    .venv/bin/python scripts/pool_target_quality.py --out outputs/loss_pool/quality.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_gpu_enable_command_buffer=+WHILE,+CONDITIONAL,+CONVOLUTION --xla_gpu_graph_min_graph_size=1"
)

from flax import nnx
from hydra import compose, initialize_config_module
import jax
import jax.numpy as jnp
import numpy as np
from pgx.experimental import auto_reset

from scacchi.dirichlet_mctx.native_targets import TARGET_BINNED, TARGET_CATEGORICAL
from scacchi.envs import make_env
from scacchi.network import build_model
from scacchi.pipeline import build_optimizer, make_training_iteration
from scacchi.play_search import make_evaluator, make_search, make_search_player
from scacchi.types import load_config

HEADS = ("policy", "value", "q")


def as_distributions(posterior, legal):
    """Search targets as the probability vectors and weights the loss uses."""
    prediction, meta = posterior.prediction, posterior.metadata

    def belief(probs, kind, outcome, weight):
        kind = jnp.asarray(kind)
        exact = jax.nn.one_hot(jnp.asarray(outcome, jnp.int32) * (probs.shape[-1] - 1), probs.shape[-1], dtype=probs.dtype)
        target = jnp.where((kind == TARGET_CATEGORICAL)[..., None], exact, probs)
        valid = (kind == TARGET_BINNED) | (kind == TARGET_CATEGORICAL)
        return target, jnp.where(valid, weight, 0.0)

    row = meta.mask.astype(jnp.float32)
    value, value_weight = belief(prediction.alpha_v, meta.v_target_kind, meta.v_target_outcome, meta.v_target_weight)
    q, q_weight = belief(prediction.alpha_q, meta.q_target_kind, meta.q_target_outcome, meta.q_target_weight)
    q_weight = q_weight * (meta.q_supervision.selected & legal) * row[:, None]
    return {"policy": (prediction.policy, row), "value": (value, value_weight * row), "q": (q, q_weight)}


def network_distributions(output, legal):
    logits = jnp.where(legal, output.logits.astype(jnp.float32), -jnp.inf)
    return {
        "policy": jax.nn.softmax(logits, axis=-1),
        "value": output.value_probs.astype(jnp.float32),
        "q": output.action_probs.astype(jnp.float32),
    }


def pulls(targets, network, head_weights):
    """Per head, the weighted direction ``p - t`` a target pulls the outputs."""
    return {head: (network[head] - targets[head][0]) * targets[head][1][..., None] * head_weights[head] for head in HEADS}


def inner(a, b):
    """Per-row inner product of two pulls, per head."""
    return {head: jnp.sum((a[head] * b[head]).reshape(a[head].shape[0], -1), axis=-1) for head in HEADS}


def kl_bits(targets, network, head_weights):
    """Per-row KL(target || network) in bits, weighted like the loss."""
    total = 0.0
    for head in HEADS:
        target, weight = targets[head]
        tiny = jnp.finfo(jnp.float32).tiny
        terms = jnp.where(target > 0, target * (jnp.log(jnp.maximum(target, tiny)) - jnp.log(jnp.maximum(network[head], tiny))), 0.0)
        kl = jnp.sum(terms, axis=-1) * weight * head_weights[head]
        total = total + jnp.sum(kl.reshape(kl.shape[0], -1), axis=-1)
    return total / math.log(2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="hex7")
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--selfplay-batch", type=int, default=1024, help="games per training iteration")
    parser.add_argument("--probe-games", type=int, default=128, help="games played to collect probe positions")
    parser.add_argument("--now", type=int, nargs="+", default=[16, 32, 48, 64, 80], help="iterations at which to measure")
    parser.add_argument("--gaps", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64], help="target ages in iterations")
    parser.add_argument("--bins", type=int, default=8, help="quantile bins of the stored bits")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with initialize_config_module("scacchi.configs", version_base=None):
        config = load_config(compose(args.config, overrides=["+algorithm=binned", f"selfplay.batch_size={args.selfplay_batch}", *args.overrides]))
    env = make_env(config.env.id, config.env.board_size)
    model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(config.run.seed))
    optimizer = build_optimizer(model, config)
    training_iteration = make_training_iteration(env, config)
    steps = config.selfplay.max_num_steps
    updates_per_iteration = args.selfplay_batch * steps // config.training.batch_size
    losses = config.training.losses
    q_supervision = losses.q_supervision

    @nnx.jit
    def collect(model, key):
        """Positions from self-play with ``model`` and the targets of its searches."""
        player = make_search_player(env, model, config.selfplay.search, config.selfplay.action_commitment, q_supervision_config=q_supervision)
        init_key, step_key = jax.random.split(key)

        def step(state, key):
            player_key, reset_key = jax.random.split(key)
            state = state.replace(
                terminated=jnp.zeros_like(state.terminated), truncated=jnp.zeros_like(state.truncated), rewards=jnp.zeros_like(state.rewards)
            )
            output = player(state, player_key)
            following = jax.vmap(auto_reset(env.step, env.init))(state, output.action, jax.random.split(reset_key, args.probe_games))
            return following, (state, output.posterior)

        state = jax.vmap(env.init)(jax.random.split(init_key, args.probe_games))
        _, (states, posteriors) = jax.lax.scan(step, state, jax.random.split(step_key, steps))
        flat = lambda x: x.reshape((steps * args.probe_games,) + x.shape[2:])
        return jax.tree_util.tree_map(flat, states), jax.tree_util.tree_map(flat, posteriors)

    @nnx.jit
    def measure(model, states, stored, key):
        search = make_search(env, make_evaluator(model), config.selfplay.search, q_supervision)
        key_1, key_2 = jax.random.split(key)
        legal = states.legal_action_mask
        fresh_1 = as_distributions(search(states, key_1), legal)
        fresh_2 = as_distributions(search(states, key_2), legal)
        stored = as_distributions(stored, legal)
        network = network_distributions(model(states.observation, train=False), legal)
        pairs = jnp.maximum(jnp.mean(jnp.sum(fresh_1["q"][1] > 0, axis=-1)), 1e-6)
        head_weights = {"policy": losses.policy_weight, "value": losses.value_dir_kl_weight, "q": losses.q_dir_kl_weight / pairs}
        pull_stored, pull_1, pull_2 = (pulls(t, network, head_weights) for t in (stored, fresh_1, fresh_2))
        return {
            "use_stored": inner(pull_stored, pull_1),
            "use_fresh": inner(pull_2, pull_1),
            "norm_stored": inner(pull_stored, pull_stored),
            "norm_fresh": inner(pull_2, pull_2),
            "bits_stored": kl_bits(stored, network, head_weights),
            "bits_fresh": kl_bits(fresh_2, network, head_weights),
            # How undecided the stored search itself was about the position.
            "entropy_stored": -jnp.sum(jnp.where(stored["value"][0] > 0, stored["value"][0] * jnp.log(jnp.maximum(stored["value"][0], 1e-30)), 0.0), axis=-1),
            "row": stored["policy"][1] * fresh_1["policy"][1],
        }

    snapshots = [nnx.clone(model)]
    key = jax.random.PRNGKey(config.run.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as file:
        for iteration in range(1, args.iterations + 1):
            key, train_key = jax.random.split(key)
            jax.block_until_ready(training_iteration(model, optimizer, train_key))
            snapshots.append(nnx.clone(model))
            if iteration not in args.now:
                continue
            for gap in args.gaps:
                if gap > iteration:
                    continue
                key, collect_key, measure_key = jax.random.split(key, 3)
                states, stored = collect(snapshots[iteration - gap], collect_key)
                result = jax.device_get(measure(model, states, stored, measure_key))
                keep = np.asarray(result["row"]) > 0
                total = lambda name: sum(np.asarray(result[name][head])[keep] for head in HEADS)
                use_stored, use_fresh = total("use_stored"), total("use_fresh")
                bits_stored, bits_fresh = np.asarray(result["bits_stored"])[keep], np.asarray(result["bits_fresh"])[keep]
                record = {
                    "now_updates": iteration * updates_per_iteration,
                    "age_updates": gap * updates_per_iteration,
                    "rows": int(keep.sum()),
                    "worth": float(use_stored.mean() / use_fresh.mean()),
                    "use_stored": float(use_stored.mean()),
                    "use_fresh": float(use_fresh.mean()),
                    "wrong_way_stored": float((use_stored < 0).mean()),
                    "wrong_way_fresh": float((use_fresh < 0).mean()),
                    "bits_stored": float(bits_stored.mean()),
                    "bits_fresh": float(bits_fresh.mean()),
                    # Share of a fresh target's pull that a second search reproduces.
                    "signal_share_fresh": float(use_fresh.mean() / total("norm_fresh").mean()),
                    "signal_share_stored": float(use_stored.mean() / total("norm_stored").mean()),
                }
                for head in HEADS:
                    stored_head, fresh_head = np.asarray(result["use_stored"][head])[keep], np.asarray(result["use_fresh"][head])[keep]
                    record[f"worth/{head}"] = float(stored_head.mean() / fresh_head.mean())
                # Usable signal (use) and total pull (norm) against what the pool
                # could rank the row by: its bits, or the stored search's own entropy.
                norm_stored, norm_fresh = total("norm_stored"), total("norm_fresh")
                entropy = np.asarray(result["entropy_stored"])[keep]
                for name, by, use, norm in (
                    ("by_bits/stored", bits_stored, use_stored, norm_stored),
                    ("by_bits/fresh", bits_fresh, use_fresh, norm_fresh),
                    ("by_entropy/stored", entropy, use_stored, norm_stored),
                ):
                    edges = np.quantile(by, np.linspace(0, 1, args.bins + 1))
                    index = np.clip(np.searchsorted(edges, by, side="right") - 1, 0, args.bins - 1)
                    groups = [index == b for b in range(args.bins)]
                    record[f"{name}/x"] = [float(by[g].mean()) if g.any() else None for g in groups]
                    record[f"{name}/use"] = [float(use[g].mean()) if g.any() else None for g in groups]
                    record[f"{name}/norm"] = [float(norm[g].mean()) if g.any() else None for g in groups]
                    record[f"{name}/wrong_way"] = [float((use[g] < 0).mean()) if g.any() else None for g in groups]
                file.write(json.dumps(record) + "\n")
                file.flush()
                print(
                    f"now={record['now_updates']:5d} age={record['age_updates']:5d} worth={record['worth']:.2f} "
                    f"wrong_way={record['wrong_way_stored']:.2f}/{record['wrong_way_fresh']:.2f} "
                    f"bits={record['bits_stored']:.2f}/{record['bits_fresh']:.2f} signal_share={record['signal_share_fresh']:.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
