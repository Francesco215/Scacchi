# Mesh-Ready Gumbel AlphaZero For PGX Chess

## Summary

Implement a compact, fully JAX Gumbel AlphaZero training stack for PGX chess. Use PGX for exact chess dynamics, MCTX for tree search, Flax NNX for the policy/value network, Optax for optimizers, Hydra for config composition, `jaxtyping` for shape-aware annotations, and `ty check` for static analysis.

This is AlphaZero-style, not full MuZero: PGX supplies the real environment model, so we do not learn representation/dynamics/reward models in v1.

## Key Design Decisions

- Use `mctx.gumbel_muzero_policy` with PGX state as the MCTX recurrent embedding.
- Use `pgx.make("chess")`; expect observation `(8, 8, 119)` and action count `4672`.
- Use Flax NNX, not Haiku, so models are modular and future transformer-style architectures can share the same interface.
- Use mesh-based runtime abstractions from day one:
  - no `pmap`
  - current runtime is a one-device `Mesh(..., ("data",))`
  - future scaling uses `Mesh`, `NamedSharding`, `PartitionSpec`, `jax.jit` shardings, and optionally `shard_map`
- Keep all arrays in global shape form. Do not reshape into `[num_devices, per_device_batch, ...]`.

## Public Interfaces

- Hydra config groups:
  - `env=chess`
  - `model=resnet`
  - `optimizer=adamw|muon`
  - `search=gumbel`
  - `runtime=single_mesh`
  - `train=default`
- Model contract:
  - `model(obs, *, train: bool) -> tuple[policy_logits, value]`
  - accepts leading batch axes
  - returns logits over `env.num_actions` and scalar values in `[-1, 1]`
- Runtime helpers:
  - `create_mesh(cfg) -> jax.sharding.Mesh`
  - `data_sharding(mesh) -> NamedSharding`
  - `replicated_sharding(mesh) -> NamedSharding`
- Optimizer factory:
  - AdamW default
  - Muon via `optax.contrib.muon`, with AdamW fallback for non-Muon leaves

## Implementation Changes

- Add dependencies: `mctx`, `pgx>=2.6.0`, `flax`, `optax`, `jaxtyping`, `ty`, and `pytest`.
- Create a `scacchi/` package with modules for config schemas, runtime mesh helpers, model factories, optimizer factories, MCTX/PGX search adapter, self-play, training, checkpointing, and CLI entrypoint.
- Implement ResNet first: convolutional trunk, residual blocks, policy head, value head.
- Keep transformer support future-ready by making architecture selection a factory/config concern, not a training-loop concern.
- Use `jax.vmap(env.step)` inside the recurrent function; do not reimplement chess rules or MCTS.
- Use `jax.lax.scan` for self-play over time and vectorized batches over environments.
- Use `pgx.experimental.auto_reset` for self-play episode reset handling.
- Keep self-play data shape as `[time, batch, ...]`; training minibatches use `[batch, ...]`.
- Return metrics as JAX arrays from jitted code and reduce/log on host, so future mesh reductions are isolated to runtime wrappers.
- Validate config constraints such as `batch_size % mesh.shape["data"] == 0`, even for the one-device mesh.

## Search And Training Behavior

- Root creation:
  - network produces `prior_logits` and `value`
  - MCTX root embedding is the PGX state
  - invalid actions are `~state.legal_action_mask`
- Recurrent function:
  - applies PGX step
  - evaluates network on next observation
  - masks illegal next-action logits
  - uses reward from the acting player's perspective
  - uses `discount=-1.0` for nonterminal two-player zero-sum transitions
  - uses `discount=0.0` and value `0.0` at terminal states
- Search:
  - `mctx.gumbel_muzero_policy`
  - `mctx.qtransform_completed_by_mix_value`
  - training `gumbel_scale=1.0`
  - evaluation/search-only `gumbel_scale=0.0`
- Training:
  - policy target is `policy_output.action_weights`
  - value target is discounted terminal return from self-play
  - loss is policy softmax cross-entropy plus masked value MSE
  - v1 uses recent self-play batches only; no replay buffer, reanalyze, or learned MuZero dynamics yet

## CLI And Config Usage

- Main command: `python -m scacchi.train`
- Hydra overrides should work, for example:
  - `python -m scacchi.train train.num_iters=10 search.num_simulations=32`
  - `python -m scacchi.train optimizer=muon`
  - `python -m scacchi.train model=resnet train.batch_size=256`
- Hydra output directories hold logs, resolved configs, and checkpoints.

## Test Plan

- Run:
  - `uv run ty check`
  - `uv run pytest`
- Tests cover:
  - PGX chess shape/action assumptions
  - one-device mesh creation and sharding helpers
  - ResNet factory output shapes and finite values
  - AdamW and Muon optimizer factory creation
  - MCTX search returns legal actions on a tiny batch
  - tiny self-play scan returns expected tensors without NaNs
  - one train step returns finite policy/value losses and updates parameters
  - alternate batch size works, catching assumptions that would block future mesh sharding

## Assumptions

- No `pmap` path will be introduced.
- Full learned-dynamics MuZero is out of scope for v1.
- Transformer architecture is not implemented in v1, but the model interface and Hydra factory must make it a clean future swap.
- Multi-device execution is not implemented in v1, but the runtime layer must be mesh-based and batch-axis-sharding-ready.
