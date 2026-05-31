# DQAZ Search Architecture Plan

## Goal

Make DQAZ search use one consistent architecture:

- categorical solving is always enabled
- traversal strategy is explicit and independent from categorical solving
- multi-trajectory batching per root is explicit and safe
- JAX backup is the only numeric backup path
- shared-prefix and split trajectories are handled deterministically
- the backup layout prioritizes accelerator parallelism

The current `solve_categorical` flag mixes two separate concerns:

1. whether categorical outcomes are solved and propagated
2. whether Rust may issue multiple pending leaf requests from the same tree

That coupling should be removed.

## Current Issues

The current JAX backup is semantically close to the desired behavior: path-row
updates are scattered into shared node/edge arrays before node values are
recomputed.

However, there are architectural footguns:

- `solve_categorical=False` limits each tree to one pending request.
- `solve_categorical=True` switches to a different request generator.
- duplicate `(node, edge)` updates in one JAX step rely on `.set(...)` being fed
  identical values.
- numeric backup has a Rust fallback path that can drift from JAX semantics.
- the JAX layout is node-table based and recomputes over broad padded slabs,
  which is correct but not maximally parallel.

## Target Search Structure

Categorical solving should always be maintained by the tree. It should not be a
switch that changes the meaning of search.

Rust should select leaves with Thompson sampling over unresolved posterior
edges. During this traversal, solved parts of the tree are pruned:

- if the current node is solved, stop descending because there is no information
  to gain below it
- ignore solved categorical edges as expansion candidates
- sample only unresolved posterior-valued legal edges
- if the root becomes solved, search for that root is complete

The number of pending leaf requests per root should be controlled independently
from categorical solving. A root may contribute one or many Thompson-selected
leaf requests to a transition batch, but this should be a batching choice, not a
change in search semantics.

JAX should receive the paths Rust selected and run numeric backup only. It
should not know why the paths were selected, and it should not own categorical
solving.

## Target JAX Backup Model

Use one JAX numeric backup function for all DQAZ searches.

The backup tensors should share the same leading axes:

```text
B = roots in the backup batch
D = padded path depth
T = trajectories per root in this wave
A = action size
```

For this architecture pass, the depth-local node width is equal to the
trajectory count:

```text
W = T
```

The path tensors are depth-major so their leading axes align with the node
tensors:

```text
path_slots:   [B, D, T]
path_edges:   [B, D, T]
path_mask:    [B, D, T]
leaf_alpha:   [B, T, 3]
leaf_players: [B, T]
```

`path_slots[b, d, t]` is the depth-local node slot used by trajectory `t` at
depth `d`. Shared-prefix trajectories point to the same slot.

The leading axes are intentionally aligned as `[B, D, T]`. In path tensors, the
third axis is the trajectory lane. In node tensors, the third axis is the
depth-local node slot, sized to `T`. `path_slots` maps trajectory lanes onto
node slots, so shared trajectories can collide intentionally while the tensor
shape stays aligned.

The numeric tree state is stored in depth buckets:

```text
edge_b:         [B, D, T, A, 3]
edge_completed: [B, D, T, A]
edge_r_count:   [B, D, T, A]
q_alpha:        [B, D, T, A, 3]
value_alpha:    [B, D, T, 3]
legal_mask:     [B, D, T, A]
node_players:   [B, D, T]
c_v:            [B, D, T, 3]
policy:         [B, D, T, A]
n_down:         [B, D, T]
```

At a fixed depth, JAX works on a contiguous slice:

```text
edge_b[:, depth, :, :, :]
```

This prioritizes regular accelerator work. Transposition sharing across depths
is out of scope for this layout; if the same logical state appears at different
depths, the backup export may duplicate it for this wave.

## Backup Step Semantics

For each reverse depth:

1. Gather active trajectory updates:

   ```text
   slot = path_slots[:, depth, :]    # [B, T]
   edge = path_edges[:, depth, :]    # [B, T]
   beta = current trajectory beta    # [B, T, 3]
   ```

2. Align `beta` into the parent player perspective.

3. Coalesce updates by `(root, depth, slot, edge)` before writing:

   ```text
   hit_count = scatter_add(active)   # [B, T, A]
   beta_sum = scatter_add(beta)      # [B, T, A, 3]
   beta_out = beta_sum / hit_count
   ```

4. Update edge tables:

   ```text
   edge_b[:, depth, :, :, :] = beta_out for touched slots/actions
   edge_completed[:, depth, :, :] = true for touched slots/actions
   edge_r_count[:, depth, :, :] += hit_count
   ```

5. Recompute depth-local node values over the action axis:

   ```text
   edge_posterior = where(edge_completed, edge_b, q_alpha)
   policy = posterior_best(edge_posterior, legal_mask)  # over A
   evidence = sum(policy * edge_posterior over A)
   n_down = sum(edge_r_count over legal A)
   c_v = blend(value_alpha, evidence, n_down)
   ```

6. Gather updated `c_v` back to trajectory beta:

   ```text
   beta[:, :] = c_v[:, depth, slot]
   beta_players[:, :] = node_players[:, depth, slot]
   ```

This makes the split-node behavior explicit:

```text
two trajectories update different child edges
-> scatter/coalesce writes both edges into the same depth-local slot
-> node posterior is recomputed from the full updated edge row
-> both trajectories carry the same node summary upward
```

It also makes duplicate same-edge behavior deterministic:

```text
two trajectories update the same (node, edge)
-> count increments by two
-> beta is coalesced before edge_b is written
```

## Categorical Metadata

JAX remains responsible only for numeric posterior backup.

Rust remains responsible for categorical metadata:

- terminal outcome tags
- categorical edge tags
- solved node tags
- solved action and distance

After JAX applies numeric results, Rust runs a metadata-only categorical pass.
This pass must not redo numeric backup and must not increment numeric counts.

## Rust Responsibilities

Rust should own:

- tree storage
- optional transposition bookkeeping outside the first depth-bucketed JAX backup
- request scheduling
- path recording
- child insertion/reuse
- categorical metadata
- final target export

Rust should not own:

- whole-path numeric backup
- posterior-best policy recomputation for numeric nodes on the backed-up paths

## Python/JAX Responsibilities

Python/JAX should own:

- PGX batched stepping
- model evaluation
- numeric path backup
- posterior-best policy sampling for numeric node caches

PGX transition batches should stay flat and accelerator-friendly:

```text
parent_states: [P, ...]
actions:       [P]
```

The backup input can keep root/trajectory structure:

```text
P = B * T
path tensors: [B, D, T]
```

## Migration Plan

1. Untangle search structure.

   - Make categorical solving always enabled.
   - Keep one Thompson leaf-selection path in Rust.
   - Make the Thompson selector skip solved nodes and solved edges.
   - Control per-root leaf batching independently from categorical solving.

2. Preserve useful current behavior without coupling it to solving.

   - The old one-pending-request behavior is just a batching limit of one leaf
     per root.
   - The high-parallelism `fig_8.py` behavior is Thompson selection with many
     pending leaves per root.
   - Any deterministic proof-frontier walk should be kept only as optional
     research/debug code, not as the normal way to enable categorical solving.

3. Add tests for current JAX backup invariants.

   - Two trajectories share prefix then split.
   - Both child edges update before split-node value recompute.
   - Shared parent edge above split receives two count increments.
   - Duplicate same `(node, edge)` updates are deterministic.

4. Introduce explicit coalescing in JAX backup.

   - Replace reliance on duplicate `.set(...)` values.
   - Use `scatter_add` for hit counts and beta sums.
   - Write averaged/coalesced beta to `edge_b`.

5. Move from flat path rows to root-major trajectory tensors.

   - Export aligned `[B, D, T]` path tensors from Rust.
   - Export depth-bucketed `[B, D, T, A, ...]` node/action tensors.
   - Flatten only at JAX call boundaries if needed for implementation.

6. Recompute only touched nodes.

   - Start with full depth-slice recompute for correctness.
   - Add touched-slot masks per depth.
   - Benchmark full depth-slice vs touched-slot recompute.

7. Remove Rust numeric backup from the production path.

   - Keep it only as a test reference if useful.
   - Verify final targets match JAX backup for small deterministic fixtures.

8. Update `fig_8.py` and configs.

   - Stop setting `solve_categorical`.
   - Use always-on categorical solving.
   - Set the desired per-root leaf batching directly.
   - Keep `search_jax_backup=True`.

## Required Tests

Add small scalar-style fixtures first, then WDL fixtures:

- one path, no split
- two paths, shared prefix then split
- two paths, shared prefix then same child edge
- terminal child plus nonterminal sibling
- categorical solved child metadata after JAX backup
- `finish()` reuses JAX cached policy for numeric roots
- solved categorical root uses solved policy

For the split test, assert this sequence:

```text
L1 backs up to A.x
L2 backs up to A.y
A.c_v is computed from both A.x and A.y
R.a receives A.c_v
R.a count increments by 2
```

## Performance Direction

The priority is accelerator parallelism.

Prefer regular, padded tensors over Python/Rust loops when the semantic merge is
clear. Keep the merge explicit with scatter/coalesce operations rather than
relying on accidental duplicate write behavior.

Key benchmarks:

- backup time vs current node-table backup
- JAX compile count
- PGX step/eval utilization
- tree-size scaling for `fig_8.py`
- p10/median latency gap across repeated search calls

## Non-Goals

Do not change the training target semantics:

- WDL Dirichlet heads remain numeric posterior targets.
- categorical targets remain native categorical fields.
- categorical outcomes are not converted into synthetic Dirichlet target alphas.

Do not move categorical solving into JAX until the numeric backup path is stable.

Do not preserve transposition sharing inside the first depth-bucketed JAX backup.
Rust may continue to use transpositions internally, but the exported backup batch
can duplicate a logical state across depth buckets if needed.
