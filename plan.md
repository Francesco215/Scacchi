# Native Categorical Posterior Tree Implementation Plan

## Summary

Implement `latex/algorithms_categorical_full.tex` as the new posterior-tree contract. Terminal and solved outcomes must be represented as native categorical certificates, not as high-concentration Dirichlet proxies. Categorical nodes and edges are absorbing, search never descends through categorical edges, and learner rows carry tagged native Dirichlet/Categorical/Mixture targets.

The production path is `posterior_tree_wavefront` with the arena backend, but the object tree and store-backed wavefront path should stay semantically aligned for comparison and tests.

## 1. Native Target Representation

Edit:

- `scacchi/dirichlet_tree/native.py` (new file, about 250 lines)
- `scacchi/dirichlet_tree/types.py` (about 120 lines)
- `scacchi/train.py` and YAML configs (about 50 lines)

Implement:

- Target tags: `TARGET_PAD`, `TARGET_DIRICHLET`, `TARGET_CATEGORICAL`.
- Outcome constants for WDL order `[L, D, W]`.
- Shape-compatible native sidecar arrays:
  - Dirichlet payload: positive alpha vector.
  - Categorical payload: outcome index plus certified distance.
- Native helpers for:
  - constructing Dirichlet and categorical targets,
  - aligning targets across player perspective,
  - sampling utility from native targets,
  - computing deterministic/mean utility for diagnostics.
- Add result/training fields:
  - `q_target_kind`, `q_target_weight`, `q_target_outcome`, `q_target_distance`,
  - `v_target_kind`, `v_target_weight`, `v_target_outcome`, `v_target_distance`.
- Keep the existing `beta_Q_target[..., action, outcome]` and
  `beta_V_target[..., outcome]` tensor shapes for replay/model compatibility.
  Categorical status is carried by the native sidecar fields, and the legacy
  beta arrays are compatibility payloads only for categorical rows.
- Add config fields:
  - `categorical_epsilon`, default small positive value such as `1e-4`,
  - `categorical_draw_rule`, default `policy_prior`.

Compatibility:

- Provide an upgrade helper that converts old Dirichlet-only arrays into native
  Dirichlet sidecar fields.
- Keep `kappa_terminal` and `epsilon_terminal` for legacy non-native code paths only; posterior-tree native categorical code must not use them to encode terminal outcomes.

## 2. Object `PosteriorTree` Path

Edit:

- `scacchi/posterior_tree.py` (about 500 lines)
- `tests/test_posterior_tree.py` (about 220 lines)

Implement:

- Add node fields:
  - `cat_outcome`,
  - `cat_distance`,
  - `cat_action`.
- Add edge fields:
  - `edge_cat_outcome`,
  - `edge_cat_distance`,
  - native edge target arrays.
- Replace scalar `edge_posterior` logic with:
  - `native_edge_posterior`,
  - compatibility `edge_posterior` that only returns Dirichlet alpha for Dirichlet targets.
- Implement:
  - `refresh_categorical_edge`,
  - `try_categorize_node`,
  - `propagate_categorical`,
  - categorical-aware `repair_path_to_root`,
  - categorical-aware `repair_dirty_frontier`.
- Terminal leaf behavior:
  - terminal node publishes `CatObj(outcome, 0)`,
  - parent edge publishes aligned `CatObj(outcome, 1)`,
  - no `_terminal_beta` is used for posterior-tree terminal backups.
- Selection behavior:
  - categorical nodes emit no request,
  - categorical edges are excluded from Thompson selection,
  - unresolved non-blocked edges continue to use Dirichlet or native utility sampling.
- Finish behavior:
  - root categorical node returns deterministic policy and committed action,
  - non-categorical root computes posterior-best policy from native edge objects,
  - output includes native Q/V targets and legacy-compatible arrays.

## 3. Arena Wavefront Search

Edit:

- `scacchi/dirichlet_tree/arena_search.py` (about 1,100 lines)
- `scacchi/dirichlet_tree/selection.py` (about 160 lines)
- `tests/test_dirichlet_tree_wavefront.py` (about 300 lines)

Implement:

- Add arena arrays:
  - node categorical outcome, distance, committed action,
  - edge categorical outcome and distance,
  - native value/Q sidecar fields for exported training rows.
- Store host-side node states in the active arena when needed for direct legal-action terminal scans. This is required to run `RefreshCategoricalEdge` over every legal action when proving losses/draws.
- Implement batched and scalar helpers:
  - publish categorical edge,
  - publish categorical node,
  - refresh one-step terminal categorical edges,
  - refresh categorical edge from categorical child,
  - try categorize node,
  - propagate categorical certificate to ancestors.
- Traversal changes:
  - skip roots already categorical,
  - skip categorical nodes during descent,
  - exclude categorical edges from selectable action masks,
  - stop issuing new work when root becomes categorical,
  - ignore or cancel stale neural results below a categorical ancestor.
- Backup and repair changes:
  - neural leaf backup publishes native Dirichlet target,
  - terminal leaf backup publishes native categorical target,
  - child refresh publishes categorical parent edge when child is categorical,
  - dirty repair tries categorical rules before recomputing native state cache.
- State-cache changes:
  - compute policy over native edge objects,
  - use categorical-aware compatibility payloads for search-weighted child summaries,
  - blend with node value prior,
  - keep solved/categorical node values native through the sidecar fields.
- Finish/export changes:
  - root categorical finish uses deterministic action and policy,
  - root and tree-node rows export native Q/V target arrays,
  - `alpha_root` remains diagnostic/compatibility only.

## 4. Store-Backed And Redis Scope

Per the current implementation direction, Redis/store-backed blob migration is out
of scope for this plan.

Do not edit:

- `scacchi/dirichlet_tree/codec.py`
- Redis serialization details in `scacchi/dirichlet_tree/store.py`
- Redis-specific schema or migration code

Keep the store-backed tests compatible where they touch the arena or shared
helpers, but do not implement categorical codec persistence or Redis PGX-state
handling in this pass.

## 5. Training Loss And Data Plumbing

Edit:

- `scacchi/loss.py` (about 420 lines)
- `scacchi/play.py` (about 140 lines)
- `scacchi/pipeline.py` (about 80 lines)
- `scacchi/train.py` logging/validation (about 80 lines)
- `tests/test_loss_masks.py` and `tests/test_play.py` (about 260 lines)

Implement:

- `CatPoint(z)` using `categorical_epsilon`.
- `DirNLL(alpha, z; categorical_epsilon)`:
  - no target Dirichlet is constructed,
  - evaluate predicted Dirichlet density at the smoothed categorical point.
- Native recursive loss:
  - Dirichlet targets use existing Dirichlet KL,
  - categorical targets use `DirNLL`,
  - padded targets contribute zero loss.
- Q loss:
  - use native Q targets per legal action,
  - use `q_loss_weight` as the default action weight,
  - mask illegal/padded rows.
- V loss:
  - use native V target per row,
  - categorical rows use `DirNLL`,
  - non-categorical rows use KL.
- Data plumbing:
  - stack and concatenate native target fields in self-play,
  - flatten root/tree rows in `make_compute_loss_input`,
  - preserve replay buffer concatenation and minibatching with new fields,
  - active-row detection should include native policy/value/search masks.
- Metrics:
  - keep existing `value_dir_kl_loss` and `q_dir_kl_loss`,
  - add or repurpose categorical/native metrics such as `value_cat_nll_loss`, `q_cat_nll_loss`, `value_native_loss`, and `q_native_loss`.

## 6. Exact Hex Relabeler

Edit:

- `scacchi/exact_hex.py` (about 140 lines)
- `tests/test_exact_hex.py` (about 80 lines)

Implement:

- Emit exact solved WDL labels as native categorical V/Q targets.
- Do not call `_dirichlet_outcome` for posterior-tree native data.
- Use `INF_DISTANCE` unless the exact solver is extended to provide proof depth.
- Keep exact policy targets and Q weights unchanged.
- Preserve compatibility for any non-native legacy data path by using the native upgrade helper.

## 7. Tests And Acceptance Criteria

Edit:

- Existing tests listed above plus new focused tests as needed (about 900 lines total).

Add tests for:

- Terminal leaf stores categorical node/edge, not terminal Dirichlet.
- Categorical edge is never selected for further search.
- Win rule chooses the shortest certified winning edge.
- Loss rule requires every legal action to be categorical loss and chooses the longest loss.
- Draw rule requires every legal action to be draw/loss and selects only from draw edges.
- Categorical child propagates aligned parent edge outcome and distance.
- Root categorical finish returns deterministic action and policy.
- Stale GPU result below categorical ancestor is ignored.
- Solved categorical value/cache rows remain native and use categorical sidecar fields.
- Native `DirNLL` is used for categorical Q/V targets.
- All-Dirichlet behavior remains compatible with previous posterior-tree tests.
- Arena tree export includes categorical non-terminal rows and excludes dirty/in-flight rows.
- Codec roundtrip preserves categorical fields.
- Exact Hex relabeling emits categorical targets.

Run:

```bash
uv run pytest tests/test_posterior_tree.py \
  tests/test_dirichlet_tree_selection_backup_store.py \
  tests/test_dirichlet_tree_wavefront.py \
  tests/test_loss_masks.py \
  tests/test_exact_hex.py \
  tests/test_config_validation.py
```

Then run:

```bash
uv run pytest tests/test_tictactoe_dumb_search.py tests/test_play.py tests/test_dirichlet_q_search.py
uv run pytest
```

Acceptance criteria:

- No posterior-tree terminal or exact categorical target is encoded as a Dirichlet proxy.
- Categorical certificates are absorbing.
- Categorical roots stop search early and commit the certified action.
- Non-categorical nodes may contain categorical edges, and those edges participate in policy/loss targets through exact categorical utility and `DirNLL`.
- Existing all-Dirichlet posterior behavior remains unchanged when no categorical edge or node is present.

## Assumptions

- Outcome order remains `[L, D, W]`.
- `posterior_tree_wavefront` with arena backend is the production implementation.
- Object and store-backed paths remain available for comparison and tests, but arena throughput is the priority.
- Native categorical support is required for root-only and tree-node training rows.
- Categorical smoothing is loss-side only and never appears in CPU posterior-tree storage.
