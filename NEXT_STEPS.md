# Next Steps

## Repo Hygiene

- Review and commit the current implementation files.
- Expand `README.md` with common Hydra overrides and expected hardware/runtime notes.

## Training Features

- Add W&B or structured metric logging.
- Add a replay buffer instead of training only on the latest self-play batch.
- Add stronger checkpoint metadata: PGX version, MCTX version, frames, and run identifiers.

## Evaluation

- Evaluate against previous checkpoints.
- Add tournament-style evaluation with deterministic search settings.
- Consider a baseline opponent, such as a PGX baseline if available for chess or an external engine wrapper later.
- Add confidence intervals or uncertainty estimates for Elo.

## Scaling

- Implement real mesh sharding for the batch axis with `Mesh`, `NamedSharding`, and `jax.jit` shardings.
- Keep avoiding `pmap`.
- Add tests for multi-device shape/sharding behavior when more devices are available.
- Profile memory and runtime for larger chess batches and search simulation counts.

## Model Roadmap

- Add a transformer or ViT-style chess model behind the existing model factory.
- Keep the same contract: `model(obs, *, train: bool) -> (policy_logits, value)`.
- Decide whether to introduce BatchNorm/LayerNorm/RMSNorm based on the architecture.

## Training Quality

- Run longer jobs with realistic configs and track whether policy/value losses move sensibly.
- Tune search settings: simulations, considered actions, depth, and Gumbel scale.
- Tune optimizer settings for AdamW and Muon.
- Add optional learning-rate schedules.

## Test Coverage

- Add Hydra override tests for model/optimizer/runtime configs.
