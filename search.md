# Rust + JAX DQAZ Search

This document describes the current split between Rust and JAX for DQAZ search.

## Overview

Rust owns the tree. JAX owns the numeric backward backup.

Rust decides which leaf edges to expand, creates or reuses tree nodes, stores the
tree state, and applies final metadata updates. JAX receives selected paths and
does the batched Dirichlet backup along those paths.

## Search Loop

For each root state, Rust starts at the root node and walks down the tree.

At each decision node, Rust uses Thompson sampling over the action posteriors:

```text
sample each legal action's outcome distribution
compute utility = win probability - loss probability
choose the action with highest sampled utility
```

If the chosen edge already has a child, Rust follows that child and continues.
If the chosen edge has no child, Rust stops. That edge is a leaf expansion
request.

Rust repeats this across roots until it has a transition batch.

## Leaf Evaluation

Python receives the requested parent states and actions. PGX applies the actions
to produce child states. The model evaluates those child states and returns:

```text
policy logits
value alpha
Q alpha for actions
terminal information
```

Python packs legal actions into compact arrays and sends the transition results
back to Rust.

## Tree Update

Rust consumes the transition batch token and updates the tree:

```text
create or reuse the child node
attach it to the selected parent edge
clear the pending request
record the selected path from root to leaf
```

Terminal children are inserted as terminal nodes. Nonterminal children are
inserted as decision nodes.

## JAX Backup Batch

Rust exports the affected nodes and paths to JAX. The batch contains:

```text
edge evidence and visit counts
model priors for affected nodes
legal action masks
node players
selected paths
leaf alphas
leaf players
```

The arrays are padded to stable shapes. Paths are processed in fixed-depth
blocks of 32 so the JAX backup kernel can reuse one compiled executable.

## JAX Backup

JAX walks each selected path backward, from leaf to root.

For each active path step, JAX:

```text
aligns the child alpha to the parent player
writes that alpha to the selected edge
marks the edge completed
increments the edge visit count
recomputes the node posterior policy
recomputes the node cached value
passes that value upward as the next beta
```

JAX returns the updated numeric node and edge arrays.

## Applying Results

Rust applies the JAX results back into the tree:

```text
edge evidence
edge completion flags
edge visit counts
cached node values
cached search policies
```

On the JAX path, Rust does not perform a whole-path numeric backup.

## Categorical Finishing

Rust still owns categorical metadata because JAX does not handle categorical
solving natively.

After JAX has applied the numeric backup, Rust performs only categorical
finishing touches:

```text
mark categorical edges
mark solved categorical nodes
record categorical outcomes and distances
record solved actions
```

This metadata pass does not replace the JAX numeric backup and does not redo the
whole path backup in Rust.

## Finish

When search is complete, Rust reads the root data and returns:

```text
selected action
search policy
Q targets
V target
native categorical target fields
```

For normal numeric roots, `finish()` reuses the cached search policy produced by
the JAX backup. For categorical roots, Rust uses the solved categorical policy.
