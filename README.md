# Chess-in-Jax

Mesh-ready Gumbel AlphaZero for PGX chess.

Run a tiny smoke training job:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -m scacchi.train \
  model.channels=4 \
  model.blocks=1 \
  model.policy_channels=1 \
  model.value_channels=1 \
  model.value_hidden=4 \
  train.num_iters=2 \
  train.selfplay_batch_size=2 \
  train.max_num_steps=2 \
  train.batch_size=4 \
  train.search.num_simulations=2 \
  train.search.max_num_considered_actions=4 \
  train.search.max_depth=2 \
  eval.interval=1 \
  eval.batch_size=2 \
  eval.max_num_steps=2
```

Evaluation logs PGX-baseline match stats when `pgx.make_baseline_model(env.id + "_v0")`
exists. Orbax checkpoints are saved under `checkpoint.dir` and can be disabled
with `checkpoint.max_to_keep=0`.

Run checks:

```bash
uv run ty check
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q
```
