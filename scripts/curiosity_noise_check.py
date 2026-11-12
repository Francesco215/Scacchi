"""Does curiosity chase search noise? A check on a trained checkpoint.

A position's bits mix what the network could still learn with the noise of a
32-simulation search. If the curiosity head steered play towards positions
whose bits are mostly noise, the share of a target's pull that a second,
independent search reproduces would fall where the head predicts many bits.

The script plays positions with the checkpoint's network, without and with
the curiosity tilt, searches every position a second time, and reports

    signal share = <p - t_1, p - t_2> / mean(|p - t_1|^2, |p - t_2|^2)

(``p`` the network's outputs, ``t_1``, ``t_2`` the two search targets, with the
loss weights) overall and by octile of the bits the head predicted.

    .venv/bin/python scripts/curiosity_noise_check.py outputs/curiosity/h8/ckpt/curious_t1 --temperature 1.0 --out outputs/curiosity/h8/noise_t1.json
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
import jax
import jax.numpy as jnp
import numpy as np
from pgx.experimental import auto_reset

from scacchi.checkpoint import config_from_checkpoint, from_pretrained
from scacchi.curiosity import curious_action
from scacchi.dirichlet_mctx.native_targets import TARGET_BINNED, TARGET_CATEGORICAL
from scacchi.envs import make_env
from scacchi.play_search import make_evaluator, make_search, make_search_player

HEADS = ("policy", "value", "q")


def as_distributions(posterior, legal):
    """Search targets as the probability vectors and weights the loss uses."""
    prediction, meta = posterior.prediction, posterior.metadata

    def belief(probs, kind, outcome, weight):
        kind = jnp.asarray(kind)
        exact = jax.nn.one_hot(jnp.asarray(outcome, jnp.int32) * (probs.shape[-1] - 1), probs.shape[-1], dtype=probs.dtype)
        valid = (kind == TARGET_BINNED) | (kind == TARGET_CATEGORICAL)
        return jnp.where((kind == TARGET_CATEGORICAL)[..., None], exact, probs), jnp.where(valid, weight, 0.0)

    row = meta.mask.astype(jnp.float32)
    value, value_weight = belief(prediction.alpha_v, meta.v_target_kind, meta.v_target_outcome, meta.v_target_weight)
    q, q_weight = belief(prediction.alpha_q, meta.q_target_kind, meta.q_target_outcome, meta.q_target_weight)
    return {"policy": (prediction.policy, row), "value": (value, value_weight * row), "q": (q, q_weight * (meta.q_supervision.selected & legal) * row[:, None])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=1.0, help="curiosity temperature of the guided games")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()

    config = config_from_checkpoint(args.checkpoint)
    env = make_env(config.env.id, config.env.board_size)
    model = from_pretrained(args.checkpoint, env)
    steps, games, losses = int(config.selfplay.max_num_steps), args.games, config.training.losses
    q_supervision = losses.q_supervision

    def collect(temperature):
        @nnx.jit
        def run(model, key):
            player = make_search_player(env, model, config.selfplay.search, config.selfplay.action_commitment, q_supervision_config=q_supervision)
            search = make_search(env, make_evaluator(model), config.selfplay.search, q_supervision)
            init_key, step_key = jax.random.split(key)

            def step(carry, key):
                state, expected = carry  # expected: what the head predicted for this position, one move earlier
                player_key, action_key, second_key, reset_key = jax.random.split(key, 4)
                state = state.replace(
                    terminated=jnp.zeros_like(state.terminated), truncated=jnp.zeros_like(state.truncated), rewards=jnp.zeros_like(state.rewards)
                )
                first = player(state, player_key)
                second = search(state, second_key)
                output = model(state.observation, train=False)
                next_bits = output.next_bits
                action = first.action
                if temperature is not None:
                    action = curious_action(action_key, first.posterior.prediction.policy, state.legal_action_mask, next_bits, temperature)
                legal = state.legal_action_mask
                network = {
                    "policy": jax.nn.softmax(jnp.where(legal, output.logits.astype(jnp.float32), -jnp.inf), axis=-1),
                    "value": output.value_probs.astype(jnp.float32),
                    "q": output.action_probs.astype(jnp.float32),
                }
                targets = [as_distributions(posterior, legal) for posterior in (first.posterior, second)]
                pairs = jnp.maximum(jnp.mean(jnp.sum(targets[0]["q"][1] > 0, axis=-1)), 1e-6)
                weights = {"policy": losses.policy_weight, "value": losses.value_dir_kl_weight, "q": losses.q_dir_kl_weight / pairs}
                pulls = [{h: (network[h] - t[h][0]) * t[h][1][..., None] * weights[h] for h in HEADS} for t in targets]
                dot = lambda a, b: sum(jnp.sum((a[h] * b[h]).reshape(games, -1), axis=-1) for h in HEADS)
                bits = 0.0
                for h in HEADS:
                    target, weight = targets[0][h]
                    terms = jnp.where(target > 0, target * (jnp.log(jnp.maximum(target, 1e-30)) - jnp.log(jnp.maximum(network[h], 1e-30))), 0.0)
                    bits = bits + jnp.sum((jnp.sum(terms, axis=-1) * weight * weights[h]).reshape(games, -1), axis=-1)
                following = jax.vmap(auto_reset(env.step, env.init))(state, action, jax.random.split(reset_key, games))
                played = jnp.take_along_axis(next_bits, action[:, None], axis=-1)[:, 0]
                # How much the head can tell the candidate moves apart, and what a tilt gains from it.
                policy = first.posterior.prediction.policy
                support = legal & (policy > 0)
                plain_mean = jnp.sum(policy * jnp.where(support, next_bits, 0.0), axis=-1)
                spread = jnp.sqrt(jnp.sum(policy * jnp.where(support, jnp.square(next_bits - plain_mean[:, None]), 0.0), axis=-1))
                tilted = jax.nn.softmax(jnp.where(support, jnp.log(jnp.maximum(policy, 1e-30)) + next_bits / args.temperature, -jnp.inf), axis=-1)
                record = {
                    "spread": spread,
                    "tilt_gain": jnp.sum(tilted * jnp.where(support, next_bits, 0.0), axis=-1) - plain_mean,
                    "moves": jnp.sum(support, axis=-1).astype(jnp.float32),
                    "use": dot(pulls[0], pulls[1]),
                    "norm": 0.5 * (dot(pulls[0], pulls[0]) + dot(pulls[1], pulls[1])),
                    "bits": bits / math.log(2.0),
                    "expected": expected,
                    "row": targets[0]["policy"][1] * targets[1]["policy"][1],
                }
                # A new game starts with no prediction for its first position.
                return (following, jnp.where(following.terminated, jnp.nan, played)), record

            state = jax.vmap(env.init)(jax.random.split(init_key, games))
            _, records = jax.lax.scan(step, (state, jnp.full((games,), jnp.nan)), jax.random.split(step_key, steps))
            return records

        return run

    def summarise(records):
        flat = {name: np.asarray(value).reshape(-1) for name, value in records.items()}
        keep = flat["row"] > 0
        use, norm, bits, expected = (flat[name][keep] for name in ("use", "norm", "bits", "expected"))
        result = {"rows": int(keep.sum()), "signal_share": float(use.sum() / norm.sum()), "bits": float(bits.mean()), "wrong_way": float((use < 0).mean())}
        known = np.isfinite(expected)
        result["predicted_bits"] = float(expected[known].mean())
        result["spread_across_moves"] = float(flat["spread"][keep].mean())
        result["tilt_gain"] = float(flat["tilt_gain"][keep].mean())
        result["candidate_moves"] = float(flat["moves"][keep].mean())
        result["prediction_correlation"] = float(np.corrcoef(expected[known], bits[known])[0, 1])
        for name, by in (("by_predicted", expected), ("by_measured", bits)):
            rows = known if name == "by_predicted" else np.ones_like(known)
            edges = np.quantile(by[rows], np.linspace(0, 1, args.bins + 1))
            index = np.clip(np.searchsorted(edges, by, side="right") - 1, 0, args.bins - 1)
            groups = [rows & (index == b) for b in range(args.bins)]
            result[name] = {
                "x": [float(by[g].mean()) for g in groups],
                "measured_bits": [float(bits[g].mean()) for g in groups],
                "signal_share": [float(use[g].sum() / norm[g].sum()) for g in groups],
                "signal": [float(use[g].mean()) for g in groups],
            }
        return result

    report = {"checkpoint": str(args.checkpoint), "temperature": args.temperature}
    for index, (label, temperature) in enumerate((("plain", None), ("curious", args.temperature))):
        records = jax.device_get(collect(temperature)(model, jax.random.PRNGKey(index)))
        report[label] = summarise(records)
        r = report[label]
        print(f"{label:8s} rows={r['rows']} bits={r['bits']:.2f} signal_share={r['signal_share']:.2f} wrong_way={r['wrong_way']:.2f} corr(predicted, measured)={r['prediction_correlation']:.2f}")
        print(f"   candidate moves={r['candidate_moves']:.1f} spread of predicted bits across them={r['spread_across_moves']:.3f} expected gain of the tilt={r['tilt_gain']:.3f}")
        print("   by predicted bits:", " ".join(f"{x:.2f}:{s:.2f}" for x, s in zip(r["by_predicted"]["x"], r["by_predicted"]["signal_share"])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
