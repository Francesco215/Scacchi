# Migration Plan: `algorithms_revised_evalcount_gamma`

## Summary
Migrate posterior-tree search from additive evidence `alpha_base + E` to completed edge posterior snapshots `B`, downstream evaluation counts `R`, and gamma-blended clean state caches `C^V`.

Primary implementation target: [arena_search.py](/workspace/worktrees/Scacchi/pecan-jackal/Scacchi/scacchi/dirichlet_tree/arena_search.py:79). Keep the legacy store-backed path in [posterior_tree.py](/workspace/worktrees/Scacchi/pecan-jackal/Scacchi/scacchi/posterior_tree.py:62) and [backup.py](/workspace/worktrees/Scacchi/pecan-jackal/Scacchi/scacchi/dirichlet_tree/backup.py:44) semantically aligned so valid search policies do not diverge.

## Public Interfaces And Types
- Update [types.py](/workspace/worktrees/Scacchi/pecan-jackal/Scacchi/scacchi/dirichlet_tree/types.py:55):
  - Replace `edge_evidence_E` semantics with completed edge posterior fields: `edge_B`, `edge_has_post`, `edge_eval_count_R`, `edge_version`, `edge_child_cache_version`.
  - Add node cache fields: `parent_key/node_id`, `parent_action`, `depth`, `value_cache_C`, `downstream_eval_count`, `value_cache_status`, `value_cache_version`, `edge_epoch`.
  - Keep `edge_post_alpha` as a derived compatibility property: `where(edge_has_post, edge_B, edge_base_alpha)`.
  - Bump [codec.py](/workspace/worktrees/Scacchi/pecan-jackal/Scacchi/scacchi/dirichlet_tree/codec.py:9) `VERSION` and encode/decode the new fields.
- Update config in [train.py](/workspace/worktrees/Scacchi/pecan-jackal/Scacchi/scacchi/train.py:180):
  - Add `leaf_value_mode: "alpha" | "mean" = "alpha"`.
  - Add `kappa_leaf = 1.0`, `kappa_terminal = 8.0`, `epsilon_terminal = 1e-6`, `state_posterior_kappa_n = 9.0`.
  - Keep old `c_leaf`, `c_terminal`, `c_state`, `c_value_search` as deprecated aliases during migration. Map `c_leaf -> kappa_leaf`, `c_terminal -> kappa_terminal`, and if only `c_state` is provided set `state_posterior_kappa_n = (1 - c_state) / c_state` for `c_state > 0`; use `1e9` for `c_state == 0`.
  - Treat `argmax_q_mean` as an alias for `scalar_q_argmax`.
- Rename training weight data from `q_evidence_mass` to `q_loss_weight` across `SearchResult`, `PosteriorTreeBatchOutput`, `TreeTrainingData`, `SelfplayOutput`, and `Sample`.
- Add `search_loss_mask` for root rows. Use it for search policy/value Dirichlet KL masks; keep `outcome_mask` only for eventual game-outcome losses.

## Function-By-Function Migration
- `EdgeBase` / `EdgePosterior`:
  - In `PosteriorArena`, `NodeBlob`, and `PosteriorTree`, stop computing posteriors as `edge_base + edge_E`.
  - `EdgePosterior` must return `edge_B` when `edge_has_post` is true, otherwise the current base prior.
  - Existing child-value base refresh remains only a fallback for edges without completed `B`.

- `ThompsonSelect`:
  - Keep [selection.py](/workspace/worktrees/Scacchi/pecan-jackal/Scacchi/scacchi/dirichlet_tree/selection.py:8) sampling logic.
  - Change callers in `_traverse_lanes` and `PosteriorTree.thompson_select` to pass only available actions, excluding edges whose child is `inflight` or `expanding`.

- `ComputeStateSearchPosterior`:
  - Replace `recompute_summary`, `_state_search_posterior_batch`, and `PosteriorTree.state_search_posterior`.
  - Compute `pi_search` over all legal actions, including unvisited fallback actions.
  - Compute `E^V = sum(pi_search[a] * EdgePosterior(a))`, `N_down = sum(R[a])`, `gamma = N_down / (state_posterior_kappa_n + N_down)`, and `C^V = (1 - gamma) * node_value_alpha + gamma * E^V`.

- `RefreshEdgeFromChild`:
  - Add arena and store-backed helpers that refresh a parent edge only when the child is expanded, non-terminal, has completed child evidence, and has a clean `C^V`.
  - Publish `B_parent = align(child.C^V)`, `R_parent = 1 + child.N_down`, set `edge_has_post`, update child-cache version, increment edge version, and mark the parent dirty.
  - Do not refresh from newly expanded neural leaves with no completed outgoing child evidence.

- `TryRepairNode`, `RepairPathToRoot`, `RepairDirtyFrontier`:
  - Add repair methods to `BatchedPosteriorArenaSearch` and equivalent store/legacy implementations.
  - Repair children before parents, reject stale repairs when `edge_epoch` changes, and publish clean `C^V` plus `N_down`.
  - In the single-threaded arena backend, implement repair tokens as simple status checks; keep the state machine structure so later parallelization remains compatible.

- `BackupPath`:
  - Replace additive backup in `_backup_path`, `_backup_pending_rows`, `backup.backup_path`, and `PosteriorTree.backup_path`.
  - Final edge publishes `B = align(beta_leaf)`, `R = 1`, `edge_has_post = true`; it does not add to previous edge mass.
  - After publishing the final edge, mark the final parent dirty and call `RepairPathToRoot`.

- `InitializePosteriorTree`:
  - Initialize root node cache to `alpha_V`, `value_cache_status = clean`, versions/counters to zero, edge `B` to absent, and child-cache version to `-1`.
  - For terminal nodes, store terminal status and a positive terminal value vector.

- `TryGetOrCreateChild`, `EvaluateReachedLeaf`, `NextRequest`:
  - Add explicit `unexpanded -> inflight -> expanding -> expanded` evaluation status.
  - Allocate unique child placeholders before evaluation; duplicate lanes must observe the same child and not emit duplicate requests.
  - Terminal leaves produce `epsilon_terminal * ones + kappa_terminal * one_hot(outcome)` and back up immediately.
  - Non-terminal leaves emit one request only after successfully marking `inflight`.

- `ConsumeResult`:
  - Install logits, `alpha_V`, and `alpha_Q`; initialize `C^V = alpha_V`, `N_down = 0`, clean cache version incremented, then publish `expanded`.
  - Back up `beta_leaf = alpha_V` for `leaf_value_mode="alpha"`.
  - For `leaf_value_mode="mean"`, back up `kappa_leaf * mean(alpha_V)`.

- `BuildEvalBatch` / `RunPosteriorTreeSearch`:
  - Stop issuing new requests for a root when `done + inflight >= num_simulations`.
  - Drain outstanding in-flight results before export.
  - Run opportunistic repair during the loop and final `RepairDirtyFrontier` before `finish_search`.

- `FinishSearch` / `MakeTargets`:
  - In `_finish_search_dense`, `finish_search`, and `PosteriorTree.finish`, run final repair first.
  - Root `beta_Q_target` is `EdgePosterior` for every legal action.
  - Root `beta_V_target` is root `C^V`, not `root.alpha_v + c_value_search * value_proxy`.
  - `q_loss_weight` is posterior-best `pi_search`, not evidence mass.
  - Tree export in `_build_tree_training_data` includes only clean expanded non-terminal nodes with child evidence. Terminal leaves and newly expanded neural leaves are excluded.
  - Root appears exactly once through the root `SearchResult`; tree-data export always skips roots.

- Loss pipeline:
  - In [loss.py](/workspace/worktrees/Scacchi/pecan-jackal/Scacchi/scacchi/loss.py:45), use `q_loss_weight` directly:
    `sum(w_Q * KL_Q) / max(sum(w_Q), eps)`.
  - Use `search_loss_mask` for policy and value Dirichlet KL rows.
  - Keep eventual-return value/outcome losses controlled only by `outcome_mask`.

## Tests
- Update existing tests that assert additive `edge_E` behavior to assert snapshot overwrite behavior, `edge_has_post`, and `R`.
- Add tests for gamma cache computation: unvisited legal actions included, `N_down` controls gamma, and `C^V` matches the formula.
- Add tests for child refresh gating: no refresh from dirty children or neural leaves without child evidence; refresh from clean interior child sets `B`, `R = 1 + N_down`, and version.
- Add tests for terminal and neural leaf backup vectors: terminal uses positive narrow Dirichlet; neural alpha mode backs up raw `alpha_V`.
- Add wavefront duplicate/in-flight tests proving duplicate lanes do not create duplicate requests or duplicate posterior writes.
- Update training/export tests so Q KL is weighted by `pi_search`, value target is `C^V`, terminal tree rows are excluded, and root is not duplicated.
- Run at minimum: `pytest tests/test_dirichlet_tree_selection_backup_store.py tests/test_dirichlet_tree_wavefront.py tests/test_posterior_tree.py tests/test_loss_masks.py tests/test_dirichlet_tree_codec.py tests/test_config_validation.py`.

## Assumptions
- `posterior_tree_wavefront` with arena backend remains the production path.
- Legacy `posterior_tree` and store-backed wavefront stay supported, but performance work is limited to the arena path.
- `dirichlet_q_search.py` is a separate MCTX policy and is not migrated beyond any required output-field rename.
- Backward compatibility is source-level only; serialized node blobs require codec version migration or regeneration.
