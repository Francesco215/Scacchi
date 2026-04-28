# Scacchi

Mesh-ready Gumbel AlphaZero for PGX chess, implemented in pure JAX.

Scacchi (Italian: "chess") trains a ResNet policy/value network via self-play and
Gumbel MCTS. The entire pipeline — self-play, search, training, evaluation, and
checkpointing — runs on JAX with first-class support for multi-device meshes.

## Features

- **Gumbel AlphaZero search** via [MCTX](https://github.com/google-deepmind/mctx) over exact PGX chess dynamics (no learned model)
- **AlphaZeroResNet** — configurable ResNetV2 trunk with separate policy and value heads
- **JAX device mesh** — batch-axis sharding ready for single- or multi-GPU/TPU training
- **Hydra configuration** — all hyperparameters overridable from the command line
- **Orbax checkpoints** — async save/restore; resume from latest checkpoint automatically
- **W&B logging** — optional; disabled by default
- **Optimizer choice** — AdamW, Adam, or Muon
- **PGX baseline evaluation** — periodic win/draw/loss reporting against the built-in chess baseline

## Installation

Requires Python 3.11, CUDA 13, and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Quick start

To run a smoke test, edit `scacchi/configs/config.yaml` to use small values (e.g. 2 iterations, tiny model), then launch training.

Full training run (default config):

```bash
uv run python -m scacchi.train
```

Useful overrides:

```bash
uv run python -m scacchi.train \
  train.num_iters=50000 \
  train.batch_size=512 \
  train.selfplay_batch_size=128 \
  train.search.num_simulations=64 \
  checkpoint.dir=runs/exp1 \
  logging.wandb_enabled=true
```

Checkpoints are saved under `checkpoint.dir` (default: `checkpoints/`) every
`checkpoint.save_interval_steps` steps and training resumes automatically from
the latest one. Set `checkpoint.max_to_keep=0` to disable checkpointing.

Evaluation logs win/draw/loss rates against the PGX baseline when
`pgx.make_baseline_model(env.id + "_v0")` is available.

## Configuration reference

All config keys can be overridden on the command line via Hydra (`key=value`).

| Section | Key knobs | Default |
|---|---|---|
| `env` | `id` | `chess` |
| `model` | `channels`, `blocks`, `resnet_v2`, `batch_norm` | 64 channels, 4 blocks, V2, BN on |
| `optimizer` | `name`, `learning_rate`, `weight_decay` | `adamw`, 1e-3, 1e-4 |
| `train` | `num_iters`, `batch_size`, `selfplay_batch_size`, `max_num_steps` | 10 000, 128, 32, 16 |
| `train.search` | `num_simulations`, `max_num_considered_actions`, `gumbel_scale` | 16, 16, 1.0 |
| `eval` | `enabled`, `interval`, `batch_size`, `max_num_steps` | true, every 10 iters, 4, 64 |
| `checkpoint` | `dir`, `max_to_keep`, `save_interval_steps`, `resume` | `checkpoints`, 5, 1000, true |
| `runtime` | `num_devices` | 1 |
| `logging` | `wandb_enabled`, `wandb_project` | false, `scacchi-az` |

## Architecture

```
Observation (8×8×119)
        │
   Conv + BN (trunk)
        │
  N × ResidualBlock
        │
   ┌────┴────┐
Policy head  Value head
(logits 4672)  (scalar ∈ [−1, 1])
```

**Self-play loop** (`scacchi/selfplay.py`): runs a batch of chess environments for
`max_num_steps` half-moves. At each step, Gumbel MCTS (`scacchi/search.py`) queries
the network and PGX dynamics to compute action weights, which become policy targets.

**Training** (`scacchi/training.py`): policy loss is softmax cross-entropy against
MCTS action weights; value loss is L2 against discounted terminal rewards, masked to
positions where a terminal value is available within the episode.

## Project structure

```
scacchi/
├── types.py        # SelfplayBatch, TrainingBatch namedtuples
├── config.py       # Dataclass configs with validation
├── runtime.py      # JAX mesh creation and NamedSharding helpers
├── models.py       # AlphaZeroResNet (ResidualBlock, policy/value heads)
├── optim.py        # Optimizer factory (AdamW, Adam, Muon)
├── search.py       # Gumbel MCTS over PGX dynamics
├── selfplay.py     # Batched self-play with jax.lax.scan
├── training.py     # Loss functions, minibatch update, iteration step
├── checkpoint.py   # Orbax async checkpoint manager
├── evaluation.py   # Baseline evaluation (win/draw/loss rates)
├── train.py        # Hydra entrypoint and main training loop
└── configs/
    └── config.yaml # Default configuration
tests/
├── test_smoke.py       # Integration tests (model, search, self-play, training)
├── test_checkpoint.py  # Checkpoint save/restore
├── test_config.py      # Config validation
└── test_evaluation.py  # Baseline evaluation
```

## Checks

```bash
uv run ty check
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q
```
