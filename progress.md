# Dirichlet Wavefront Search Progress

This file summarizes the current speed work for the Redis-compatible batched posterior tree. It is written from the current worktree state and focuses on what changed, how the new path works, what has been measured, and what still needs work before claiming the full 100x goal across the search workload.

## Goal

Make posterior-tree search dramatically faster while preserving the revised WDL Dirichlet posterior-search theory in `latex/algorithms_revised.tex` and the implementation direction in `plan.md`.

The current strategy is:

1. Keep the existing object-per-node `posterior_tree` path available for comparison.
2. Add a new `scacchi/dirichlet_tree/` implementation.
3. Move the hot path away from Python node and edge objects into a preallocated arena.
4. Use JAX batched PGX stepping and JAX state hashing.
5. Keep Redis out of the critical search loop for now. Redis support exists as a backing/store interface, but the speed path is the local arena.

## Theory Invariants Preserved

The wavefront arena path is intended to follow the revised algorithm:

- Outcome order is always `[L, D, W]`.
- Traversal uses Thompson samples from `edge_post_alpha`, not posterior-mean argmax.
- `edge_post_alpha = edge_base_alpha + edge_E`.
- Fresh expanded nodes initialize edge base alpha from NN `q_alpha`, with zero evidence.
- When a child is expanded, the parent edge base is updated from the aligned child `value_alpha`.
- Direct leaf backup adds only completed evidence to the final selected edge.
- Ancestor backup uses child state summary evidence, aligned for player perspective.
- There is no virtual loss, reservation mass, or temporary posterior `P`.
- In-flight and duplicate handling are scheduling concerns only; they do not alter posterior mass.
- Root policy targets are posterior-best Monte Carlo samples over final root edge posteriors.
- Final committed action defaults to greedy posterior-mean utility.

## Main New Files

- `scacchi/dirichlet_tree/types.py`
  Core types such as `StateKey`, `NodeBlob`, `SearchConfig`, and `SearchResult`.

- `scacchi/dirichlet_tree/codec.py`
  Packed binary `NodeBlob` codec with a version header. It avoids pickle.

- `scacchi/dirichlet_tree/store.py`
  Redis-compatible store interface plus `InMemoryNodeStore` and `RedisNodeStore`.

- `scacchi/dirichlet_tree/state_hash.py`
  JAX-compatible state keys. Hex has a faster specialized hash over board, player, player order, step count, and terminal flag.

- `scacchi/dirichlet_tree/selection.py`
  Thompson action selection and posterior-best policy target helpers.

- `scacchi/dirichlet_tree/arena_search.py`
  The current fast path. It owns the struct-of-arrays arena and batched wavefront traversal.

- `scripts/bench_prepare_leaf_batch.py`
  Measures how long it takes to prepare a leaf batch for the model.

- `scripts/bench_wavefront_pgx.py`
  Measures the real PGX Hex wavefront path.

- `scripts/bench_wavefront_matrix.py`
  Runs the benchmark matrix and ablation variants with timing buckets, JSONL output, and summary statistics.

## Arena Representation

The hot path stores nodes and edges as arrays, not Python objects:

```text
node_status          [max_nodes]
node_key             [max_nodes, 4]
node_current_player  [max_nodes]
node_first_edge      [max_nodes]
node_num_edges       [max_nodes]
node_value_alpha     [max_nodes, 3]
node_summary_alpha   [max_nodes, 3]

edge_parent_node     [max_edges]
edge_action          [max_edges]
edge_child_node      [max_edges]
edge_child_key       [max_edges, 4]
edge_base_alpha      [max_edges, 3]
edge_E               [max_edges, 3]
edge_post_alpha      [max_edges, 3]
edge_logit           [max_edges]
edge_visits          [max_edges]
```

A node is an integer `node_id`. Its outgoing edges are a slice:

```python
start = node_first_edge[node_id]
end = start + node_num_edges[node_id]
```

This removes most per-search Python allocation compared with creating `Node`, `Edge`, `EvalRequest`, and `PathStep` objects inside the loop.

## How Search Works

For a batched set of PGX roots:

1. Hash root states with JAX.
2. De-duplicate identical roots so repeated roots are searched once and broadcast back.
3. Evaluate root observations with the supplied `leaf_evaluator`.
4. Allocate expanded root nodes in one arena batch.
5. Repeatedly launch wavefront lanes for roots that still need simulations.
6. For each lane:
   - Read the current arena node.
   - Build a padded sparse edge tensor for legal actions.
   - Run Thompson selection over `edge_post_alpha`.
   - Step the PGX states in a JAX `vmap(env.step)` batch.
   - Hash child states in the same fused JAX step/hash helper.
   - Classify lanes as terminal, already expanded, missing, duplicate, or in-flight.
7. Evaluate unique missing non-terminal leaves once per batch.
8. Expand new child nodes in an arena batch.
9. Back up completed evidence through fixed path arrays.
10. Finish roots by producing:
    - final action
    - posterior-best root policy target
    - `beta_Q_target`
    - `beta_V_target`
    - root posterior alpha tensors

## Recent Optimizations

The latest pass focused on host-side overhead and JAX shape churn:

- Added batched root search entrypoint so wavefront search can consume a PGX state batch directly instead of splitting into a Python list.
- Added root de-duplication and result broadcasting.
- Added batched root and leaf node allocation.
- Switched key dictionary entries from tuple keys to raw 16-byte key bytes.
- Made the arena `key_to_node` index lazy, avoiding a Python dict build for 262,144 roots before the model batch is ready.
- Reused the sorted root-key view for fresh-leaf collision checks.
- Added a generic fresh-leaf fast path with collision safety.
- Added a single-legal-edge shortcut. This preserves Thompson semantics because there is only one valid action.
- Fused PGX step, state hash, terminal flag, player, rewards, and legal mask into one JAX batched helper.
- Padded the PGX step/hash call to reduce recompiles from shrinking active lane batches while keeping sparse Thompson selection.
- Padded leaf-evaluator calls to `search_eval_batch_size` and sliced outputs back to the real leaf count before node expansion.
- Added a hybrid Thompson selector: large batches use JAX, while small ragged batches use NumPy Thompson sampling to avoid JAX recompiles on stochastic path shapes. The current default is `wavefront_np_select_below: 1024`.
- Vectorized arena expansion for leaf batches whose legal masks differ but have the same legal count. This is common in PGX Hex at the same search depth.
- Grouped mixed-legal-count leaf expansions by legal count inside the arena allocator, so coalesced PGX leaf batches still avoid per-node expansion.
- Coalesced pending leaf batches before model evaluation so a wave fills `search_eval_batch_size` chunks more consistently.
- Added experimental `wavefront_pad_jax_select`. It is currently defaulted off because CPU benchmarks did not improve with it.
- Avoided a pre-step JAX state gather when active lane state positions are already prefix-compacted.
- Used slicing instead of JAX gather for pending observations when missing leaf rows are a prefix.
- Batched parent edge-base refreshes when known expanded children are reached, so the search refreshes affected edge statistics once per wave instead of per lane.
- Grouped node-summary recomputation by legal-edge count, preserving sparse legal-action layouts while avoiding thousands of scalar parent recomputes.
- Kept PGX states lane-indexed during traversal and stepped the full lane batch with inactive actions filled by zero. This avoids dynamic per-depth state compaction gathers while preserving selected actions and path records for active lanes.
- Added a stable lane-batch mode that keeps the same root/lane shape across waves, filters completed roots inside traversal, and avoids JAX churn from unevenly shrinking unfinished-root batches.
- Padded pending-observation gathers to the evaluation batch size when eval padding is enabled, reducing JAX shape churn before model calls.
- Raised the NumPy Thompson-selection threshold from 256 to 1024 after CPU benchmarks showed this is faster for the current `batch=512`, `simulations=8` PGX workload.
- Reduced the stepped-lane host transfer by returning only the scalar terminal reward needed for terminal backup instead of the full per-player reward vector.
- Added optional arena timing buckets for root hashing, root eval, node packing, Thompson selection, PGX step/hash, device transfer, child classification, leaf observation gather, NN eval, expansion, backup, posterior target generation, and store flush.
- Added ablation flags for grouped expansion, lane-indexed stepping, and padded pending-observation gathers, plus the existing hybrid-selector threshold toggle.
- Added benchmark summaries that report median, p10, p90, best, and cold-start-excluded mean instead of only best warmed samples.
- Added a direct full-source eval path for pending leaves whose source observation batch already matches `search_eval_batch_size`; the model still runs one full padded batch, but expansion and backup select outputs only for the real missing leaves.
- Merged multiple already-padded pending leaf groups with a fixed-size gather instead of slicing them back to dynamic real sizes and padding again. This keeps late-wave eval preparation shape-stable when root de-dup shrinks active lanes.
- Switched NumPy Thompson selection and batched posterior-best target sampling to `Generator.standard_gamma(..., dtype=np.float32)`. This preserves exact Dirichlet sampling while avoiding float64 gamma draws followed by a cast.

## Benchmark Results

All numbers below are CPU JAX runs from this workspace, so GPU behavior still needs a separate measurement.

### Prepare 262,144 Leaves For Model

Command:

```bash
JAX_PLATFORMS=cpu uv run python scripts/bench_prepare_leaf_batch.py \
  --batch 262144 \
  --warmup-batch 0 \
  --repeats 5 \
  --policy posterior_tree_wavefront \
  --store arena
```

Recent result:

```text
repeat=1 prep_to_leaf_s=0.390309
repeat=2 prep_to_leaf_s=0.393546
repeat=3 prep_to_leaf_s=0.368256
repeat=4 prep_to_leaf_s=0.383730
repeat=5 prep_to_leaf_s=0.377652
```

Approximate comparison:

```text
old Redis/object-shaped wavefront prep: 251.84 s
current arena prep best:                  0.37 s
improvement on this benchmark:          ~684x
```

### Real PGX Hex Multi-Simulation Search

Command:

```bash
JAX_PLATFORMS=cpu uv run python scripts/bench_wavefront_pgx.py \
  --batch 512 \
  --simulations 8 \
  --board-size 5 \
  --prefill-steps 4 \
  --repeats 8 \
  --mode state_batch \
  --policy-mc-samples 1 \
  --lanes-per-root 1 \
  --no-pad-jax-select
```

Recent result after hybrid small-batch selection, grouped variable-mask expansion, pending-batch coalescing, lane-indexed full-batch stepping, stable lane batches, padded pending-observation gathers, scalar terminal-reward transfer, batched edge-base refresh, grouped summary refresh, and the `np_select_below=1024` default:

```text
repeat=1 completed_evals_per_s=1884.1
repeat=2 completed_evals_per_s=11943.6
repeat=3 completed_evals_per_s=21131.6
repeat=4 completed_evals_per_s=24151.5
repeat=5 completed_evals_per_s=14301.2
repeat=6 completed_evals_per_s=27307.5
repeat=7 completed_evals_per_s=28052.0
repeat=8 completed_evals_per_s=39755.2
```

Pure JAX selection on the same benchmark with `--np-select-below 0` was:

```text
repeat=1 completed_evals_per_s=495.4
repeat=2 completed_evals_per_s=945.2
repeat=3 completed_evals_per_s=7291.1
repeat=4 completed_evals_per_s=2350.7
repeat=5 completed_evals_per_s=2250.7
repeat=6 completed_evals_per_s=10737.8
```

The hybrid selector improves the cold and mid-run cases while preserving the fast warmed cases. Variable-mask batch expansion removes a large Python per-node expansion cost, and the later lane-indexed stepping, stable lane batches, and padded observation gathers reduce dynamic JAX gather overhead. The best warmed samples are still around 40k completed evals/s in this direct benchmark. The benchmark remains variable, but the stable-lane ablation below materially improves p10 and median in the timing-enabled matrix runner.

Earlier comparison points:

```text
object tree, initial roots, batch=512, sims=1:      ~458 evals/s
wavefront arena, initial roots, batch=512, sims=1:  ~24,416 evals/s
improvement:                                        ~53x

object tree, prefilled roots, batch=128, sims=1:    ~417 evals/s
wavefront arena, prefilled roots, batch=128, sims=1: ~3,840 evals/s
improvement:                                        ~9.2x

object tree, prefilled roots, batch=128, sims=8:     ~537 evals/s
wavefront arena, prefilled roots, batch=128, sims=8: ~18,704 evals/s
improvement:                                        ~35x
```

The multi-simulation PGX benchmark is much more variable because stochastic traversal still creates changing selector shapes and changing path patterns. It is improved, but it is not yet a clean, broad 100x proof.

### Matrix and Ablation Runner

The current benchmark matrix runner is:

```bash
JAX_PLATFORMS=cpu uv run python scripts/bench_wavefront_matrix.py \
  --require-backend cpu \
  --output cpu_wavefront_matrix.jsonl
```

Run the same script on the target GPU/default backend without `JAX_PLATFORMS=cpu`, optionally with `--require-backend gpu` if JAX reports the backend as `gpu`.

Default matrix dimensions:

```text
batch:             128, 512, 2048
simulations:       1, 4, 8, 16, 32
board size:        5, 9, 11
root mode:         initial, prefilled
policy_mc_samples: 1, 16, 64
```

The ablation runner is the same script:

```bash
JAX_PLATFORMS=cpu uv run python scripts/bench_wavefront_matrix.py \
  --benchmark ablation \
  --require-backend cpu \
  --output cpu_wavefront_ablation.jsonl
```

Ablation variants:

```text
object_tree
arena_no_hybrid_selector
arena_hybrid_selector
arena_no_grouped_expansion
arena_grouped_expansion
arena_no_lane_indexed_step
arena_lane_indexed_step
arena_no_stable_lane_batch
arena_stable_lane_batch
arena_no_padded_eval_gathers
arena_padded_eval_gathers
```

Each JSON summary includes elapsed-time stats, completed-evals/sec stats, and per-bucket timing stats. A CPU smoke run passed with `--quick --repeats 2`; the full CPU/GPU matrix is intentionally not checked into this file because it is large and should be run as a benchmark artifact.

Stable lane-batch ablation on CPU, Hex 5x5, `simulations=8`, prefilled roots, `policy_mc_samples=1`:

```text
batch=128, arena_no_stable_lane_batch: best=23334.7/s, median=20382.8/s, p10=6335.4/s, cold-start-excluded mean=20299.2/s
batch=128, arena_stable_lane_batch:    best=24231.0/s, median=23090.8/s, p10=10601.2/s, cold-start-excluded mean=23396.6/s

batch=512, arena_no_stable_lane_batch: best=26360.9/s, median=5823.1/s,  p10=3029.1/s,  cold-start-excluded mean=12437.4/s
batch=512, arena_stable_lane_batch:    best=50483.8/s, median=47410.2/s, p10=24234.8/s, cold-start-excluded mean=48153.4/s
```

The `batch=512` case is the clearest result: keeping the lane shape stable removes a large source of timing collapse in the arena path and roughly doubles the best observed throughput in the matrix benchmark.

Float32 gamma sampling improves the CPU selector bucket on larger boards without changing the sampling rule. On Hex 9x9, `batch=512`, `simulations=8`, prefilled roots, `policy_mc_samples=1`, the warmed hybrid-selector timing changed from roughly:

```text
before: Thompson selection ~= 28 ms to 29 ms
after:  Thompson selection ~= 19 ms to 20 ms
```

The corresponding post-change summary was:

```text
best=29609.9/s, median=28219.1/s, p10=14465.4/s, cold-start-excluded mean=28873.2/s
```

On the same Hex 5x5 case, the Thompson bucket dropped from roughly 8 ms to 9 ms to roughly 6 ms in warmed runs. After fixed-size padded pending merges, the Hex 5x5 leaf-observation bucket also improved materially:

```text
before: warmed leaf observation gather often ~= 18 ms to 140+ ms
after:  warmed leaf observation gather mostly ~= 6 ms to 8 ms, with one 37 ms run
```

The post-change Hex 5x5 summary was:

```text
best=52626.0/s, median=31120.4/s, p10=11038.6/s, cold-start-excluded mean=36473.7/s
```

Hex 9x9 remained stable after the pending-merge change:

```text
best=29521.0/s, median=28271.4/s, p10=10147.4/s, cold-start-excluded mean=28896.2/s
```

## Verification

Focused test command:

```bash
JAX_PLATFORMS=cpu uv run pytest \
  tests/test_dirichlet_tree_codec.py \
  tests/test_dirichlet_tree_state_hash.py \
  tests/test_dirichlet_tree_selection_backup_store.py \
  tests/test_dirichlet_tree_wavefront.py \
  tests/test_posterior_tree.py \
  tests/test_config_validation.py \
  tests/test_network_dirichlet.py
```

Latest result:

```text
57 passed in 11.90s
```

Focused smoke check after the fixed-size pending merge:

```text
33 passed in 4.15s
```

Benchmark smoke checks:

```bash
JAX_PLATFORMS=cpu uv run python scripts/bench_wavefront_matrix.py \
  --quick --repeats 2 --print-runs --require-backend cpu

JAX_PLATFORMS=cpu uv run python scripts/bench_wavefront_matrix.py \
  --benchmark ablation --quick --repeats 2 --require-backend cpu
```

Both commands completed and emitted JSON summaries with timing buckets.

After tightening synchronization inside the leaf-observation gather timing bucket, this smoke command also completed:

```bash
JAX_PLATFORMS=cpu uv run python scripts/bench_wavefront_matrix.py \
  --quick --repeats 1 --require-backend cpu
```

## Current Bottleneck

The 262,144 leaf-preparation target is now comfortably above 100x faster than the object-shaped Redis path.

The remaining bottleneck is broad multi-simulation PGX search:

- Large Thompson selections still use JAX and can see changing active-lane and legal-edge shapes.
- Some runs still pay JAX compilation or lowering costs.
- Python still orchestrates traversal, child classification, state-row gathers, and some backup work.
- More simulations per root create deeper dynamic paths, where a Python arena is better than the old object tree but still not as fast as a compiled search core.

A profile after lane-indexed stepping and padded pending-observation gathers of the `batch=512`, `simulations=8`, `prefill_steps=4` PGX case after several warmup runs showed:

```text
elapsed_s=0.331258
evals_per_s=12365.0
_traverse_lanes cumulative time=0.182 s
_evaluate_pending cumulative time=0.122 s
device_get cumulative time=0.141 s
```

The hybrid NumPy/JAX selector, batched expansion, grouped summary refresh, lane-indexed stepping, stable lane batches, padded observation gathers, and scalar terminal-reward transfer mitigated the previous compile/expansion hot spots. The next high-value target is reducing remaining host/device transfers and pending-batch concatenation. If the full matrix still shows low p10 relative to median after this stable-shape work, the selector/classification/backup core is the likely part to move into compiled code.

## Recommended Next Steps

1. Profile the hybrid selector path and decide whether to bucket large JAX selector shapes or move selection fully into a compiled CPU kernel.
2. Consider a compiled arena core for selection, child classification, and backup using C++ or Rust if the Python loop remains dominant.
3. Keep Redis outside the hot loop until local arena throughput is proven. Use Redis later for checkpointing, persistence, or coarse distributed coordination.
4. Benchmark on the actual target accelerator, since the chosen design keeps PGX stepping and model batches on the default JAX device.
5. Run the full CPU and GPU benchmark matrix and inspect p10/median gaps. If p10 remains much lower than median for target settings, bucket selector/eval shapes more aggressively or move selection, classification, and backup into a compiled extension.

## Status

The implementation has made substantial progress:

- The dedicated 262,144 batch-preparation benchmark is about 684x faster than the old object-shaped Redis path.
- The arena wavefront path is integrated behind `posterior_tree_wavefront`.
- The theory-sensitive focused suite passes, and the benchmark matrix/ablation tooling is now in place.
- The latest optimization keeps lane batches stable across waves and substantially improves p10/median behavior in the `batch=512`, `simulations=8` ablation.

The full objective is not yet proven complete because the real multi-simulation PGX search path has not demonstrated a stable 100x improvement across representative workloads.
