# Chess-in-Jax

Mesh-ready Gumbel AlphaZero for PGX chess.

Run a tiny smoke training job:

```bash
uv run python -m scacchi.train model.channels=4 model.blocks=1 model.value_hidden=4 train.selfplay_batch_size=2 train.max_num_steps=2 train.batch_size=4 search.num_simulations=2 search.max_num_considered_actions=4 search.max_depth=2
```

Run checks:

```bash
uv run ty check
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest -q
```
