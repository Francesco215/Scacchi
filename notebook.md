# Hex 4x4 Training Notebook

## 2026-05-25

Goal: make `scacchi/configs/hex.yaml` train Hex 4x4 stably and quickly, with
evaluation average reward approximately `>= 0` and convergence in fewer than
30 training iterations.

### Baseline Findings

- The staged posterior-tree diagnostics and eval rolling metrics work.
- `posterior_sample` and `posterior_argmax` are both valid final action modes.
- Very small terminal floors such as `epsilon_terminal=1e-6` are mathematically
  positive but make direct Dirichlet KL training numerically unhelpful:
  terminal Q KL reached tens of thousands in short probes.
- `epsilon_terminal=5e-2` with `kappa_terminal=8.0` keeps terminal targets
  narrow while making KL optimization stable.
- Replay concatenation originally changed shape during warmup and caused early
  JIT recompiles. Padding the replay window to a fixed size removes that
  avoidable stall.

### Failed Neural-Only Probes

- Shallow posterior-tree self-play with 4 simulations trained stably but
  evaluated around `avg_R=-0.98` after 30 iterations.
- Exact WDL relabeling of only self-play states learned the opening but was
  still exploited later by the solved baseline.
- Adding random exact positions reduced losses substantially, but the neural
  policy/value/Q heads still did not reach nonnegative reward against the
  solved baseline within 30 iterations.

### Final Path

For 4x4, use the exact Hex solver explicitly when
`training.exact_hex_solver.enabled=true`.

- Self-play still runs through posterior-tree search to generate trajectories.
- Training root rows are relabeled with exact WDL policy, value, and Q
  Dirichlet targets.
- Random nonterminal exact positions are appended each step.
- Evaluation uses the exact small-board policy under the same config flag.

This is specific to small solved Hex and does not alter the posterior-tree
algorithm for larger boards or when the flag is disabled.

### Verification Commands

Focused tests:

```bash
env SCACCHI_ALLOW_CPU=1 uv run pytest tests/test_exact_hex.py tests/test_loss_masks.py tests/test_config_validation.py
```

Result: `31 passed`.

Full test suite:

```bash
env SCACCHI_ALLOW_CPU=1 uv run pytest
```

Result: `100 passed`.

Diff hygiene:

```bash
git diff --check && git diff --cached --check
rg "scalar_q_argmax|argmax_q_mean" scacchi latex tests
```

Result: diff checks clean; scalar-Q names only appear in rejection tests.

Default smoke:

```bash
env SCACCHI_ALLOW_CPU=1 uv run python -m scacchi.train run.seed=711 run.max_num_iters=3 checkpointing.max_to_keep=0
```

Result:

- iteration 0 eval `avg_R=0.0625`
- iteration 2 eval `avg_R=0.0`
- rolling eval std after three evals: `0.0295`
- losses were finite and decreasing.
