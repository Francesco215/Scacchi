# Scacchi — Curiosity-Driven Tree Search

Scacchi is the research implementation behind **Curiosity-Driven Tree
Search**, the second post in the *Mathematical Foundations of Curiosity*
series. It explores an AlphaZero-style self-play agent whose search operates on
explicit probability distributions over game outcomes, rather than only on
scalar value estimates and visit counts.

The article source is [`index.html`](index.html), for the more in-depth treatment, see [`math.md`](math.md).

> [!WARNING]
> This repository is a research artifact and reference implementation for the
> work described in the blog post. It is released as-is and is **not intended
> to be production-ready software**. The code, configuration, hardware
> assumptions, and internal APIs may contain rough edges and may change without
> compatibility guarantees.

## The idea

The network predicts three objects for each position:

- policy logits over moves;
- a state-value Dirichlet distribution over win/draw/loss outcomes;
- one action-value Dirichlet distribution for each legal move.

For binary games such as Hex, the outcome distribution is a Beta distribution,
which is the two-outcome special case of the Dirichlet distribution. Keeping a
full distribution allows the search to represent both its current estimate and
its uncertainty.

One search simulation has three stages:

1. **Sample downward.** Thompson sampling draws one possible value for every
   legal action and follows the best draw. An uncertain move is explored when
   it still has a meaningful probability of being the best move.
2. **Expand.** PGX advances the selected action and the network evaluates at
   most one new position.
3. **Repair upward.** Search recomputes the posterior-best policy and propagates
   the resulting Dirichlet messages back to the root.

At an unresolved node, the repaired belief has the form

\[
C=(1-\gamma)V+\gamma\sum_a \pi(a)Q_a,
\qquad
\gamma=\frac{n_{\mathrm{down}}}{\kappa+n_{\mathrm{down}}},
\]

where \(V\) is the network's value prior, \(Q_a\) is the current belief for
action \(a\), \(\pi(a)\) is the posterior probability that the action is best,
and \(n_{\mathrm{down}}\) is the amount of structural search support below the
node. `kappa` is the single prior-strength constant used by the repair rule.

Terminal results do not get converted into artificially concentrated
Dirichlet distributions. The tree stores exact categorical outcome and
distance certificates alongside unresolved Dirichlet beliefs. These
certificates are propagated with minimax semantics, prune solved branches, and
take precedence over sampled values during action selection.

## Detailed mathematical reference

[`math.md`](math.md) is the detailed, implementation-aligned mathematical
reference for the project. It develops the Dirichlet value and Q
parameterization, Thompson policy, bottom-up posterior repair, exact
categorical targets, supervision masks, and training losses used by the code.
The blog post provides the intuition; `math.md` records the precise definitions
and equations.

## Implementation details

The implementation is JAX-native and designed around large, parallel self-play
batches:

- **Environment:** PGX supplies batched game states and transitions. The
  experiments in the post train on Hex boards from 3×3 through 9×9.
- **Network:** Flax NNX implements a residual policy/value/Q model. The shared
  Hex recipe uses six 128-channel residual blocks and direct
  log-concentration Dirichlet heads.
- **Search:** `scacchi/dirichlet_mctx/` is a compact, fixed-capacity,
  MCTX-shaped tree backend with a `simulate -> expand -> backward` flow. Tree
  traversal and backup run in JAX control flow and are vectorized across batch
  lanes.
- **Posterior-best policies:** the general estimator uses populations of
  Thompson samples. Binary games may instead use the guarded `prefix_cdf`
  estimator, which evaluates the Beta densities and CDFs on an adaptive
  21-point grid and falls back lane-by-lane when a numerical guard fails.
- **Self-play and training:** search produces a policy target plus typed
  Dirichlet or exact categorical targets for the value and Q heads. The
  resulting trajectories are shuffled into minibatches and used to update the
  network with Muon for hidden block kernels and its auxiliary Adam path for
  other parameters.
- **Action commitment:** the policy written to replay and the policy used to
  choose the played move are separate. A run can reuse the search policy or
  construct a fresh posterior-sample/posterior-argmax policy from the searched
  root distributions.
- **Evaluation and checkpoints:** Hydra recipes define self-play, training,
  evaluation, logging, and checkpoint behavior. Orbax stores the model,
  optimizer, RNG, configuration, and progress metadata; W&B logging is
  configurable.

Q targets are assigned only to legal actions with positive search evidence or
to legal solved actions:

\[
M_{s,a}=\mathbf 1[\operatorname{legal}(s,a)\land
(\operatorname{evidence}_{s,a}>0\lor\operatorname{solved}_{s,a})].
\]

The default Q reduction is a mean over the selected state-action pairs. Search
evidence controls whether a pair is supervised; its magnitude does not scale
that pair's loss.

## Repository map

- `scacchi/train.py`: Hydra entry point, accelerator setup, evaluation,
  checkpointing, and the outer training loop.
- `scacchi/network.py`: residual networks and policy/value/Q output heads.
- `scacchi/play.py`: batched game and self-play loops.
- `scacchi/play_search.py`: evaluator, search, replay-target, and action
  commitment boundaries.
- `scacchi/dirichlet_q_search.py`: leaf expansion, terminal extraction, and Q
  supervision helpers.
- `scacchi/dirichlet_mctx/`: Dirichlet Thompson tree search, exact outcome
  propagation, posterior repair, and policy estimators.
- `scacchi/pipeline.py`: trajectory preparation, minibatching, and one complete
  self-play/training iteration.
- `scacchi/loss.py`: policy and typed Dirichlet/categorical losses.
- `scacchi/configs/`: shared and board-specific Hydra recipes, currently
  centered on Hex.
- `scripts/`: experiment launchers, benchmarks, checkpoint inspection, and
  plotting utilities.
- `tests/`: tests for configuration, losses, networks, play, checkpointing,
  search, categorical outcomes, and numerical posterior repair.
- `website/`: interactive article code and generated figures.

The lower-level search representation and callback contracts are documented in
[`scacchi/dirichlet_mctx/README.md`](scacchi/dirichlet_mctx/README.md).

## Running the reference code

The project currently requires Python 3.13 and uses `uv` for dependency and
command management. Training is intended for a GPU or TPU. The current
dependency set explicitly selects JAX's CUDA 13 build, so it is not portable to
every platform without editing the environment definition. The checked-in Hex
recipes also use the large batch sizes from the experiments and several of
them refer to external evaluation checkpoints that are not included in this
repository.

On a compatible environment:

```bash
uv sync
uv run pytest
```

To start a from-scratch Hex 6 run without the external evaluation checkpoint
or W&B logging:

```bash
uv run scacchi-train --config-name hex6 \
  eval.baseline=none eval.interval=0 logging.wandb.enabled=false
```

Useful experiment commands include:

```bash
scripts/train_all_hex.sh
uv run python scripts/checkpoint_moves.py checkpoints/<run-directory>
```

`train_all_hex.sh` runs the numbered Hex recipes sequentially. Training always
writes a final checkpoint; `checkpointing.max_to_keep: 0` disables periodic
retention but still keeps that final step. `checkpoint_moves.py` restores the
latest model in a run directory, plays one self-play game with the stored
search settings, and prints the moves as JSON. Pass `--seed` for a reproducible
game.

## Getting help

If you want to understand, reproduce, adapt, or build on any part of this code,
please get in touch. I am happy to help with its use despite the repository's
reference-only status.
