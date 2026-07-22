# Dirichlet MCTX

`scacchi.dirichlet_mctx` is the Thompson-search sister of MCTX.  Its public
flow deliberately matches MCTX:

```python
import functools

root = dirichlet_mctx.RootFnOutput(
    prior_logits=prediction.logits,
    value=prediction.alpha_v,
    action_values=prediction.alpha_q,
    embedding=env_state,
    terminal_outcome=root_terminal_outcome,
    to_play=env_state.current_player,
)
policy_output = dirichlet_mctx.dirichlet_thompson_policy(
    params=(),
    rng_key=rng_key,
    root=root,
    recurrent_fn=expand_fn,
    num_simulations=num_simulations,
    invalid_actions=~env_state.legal_action_mask,
    posterior_update=functools.partial(
        dirichlet_mctx.update_posterior,
        kappa=4.0,
    ),
)
```

The module map is also parallel to MCTX:

- `base.py`: root, expansion, and policy-output contracts.
- `tree.py`: compact fixed-capacity `Tree` and update-view types.
- `outcomes.py`: outcome sentinels, perspective alignment, and utilities.
- `native_targets.py`: native target tags, smoothing, and categorical density NLL.
- `action_selection.py`: one node-local Thompson selector used everywhere.
- `search.py`: `simulate -> expand -> bottom-up repair`.
- `policies.py`: the public `dirichlet_thompson_policy` wrapper.
- `posterior_updates.py`: the replaceable node-posterior repair rule.

The stored search state follows `tictactoe-demo/app.js`, but disjoint logical
values share physical buffers. Every edge owns one alpha slot containing its
network Q fallback before repair and message `B` afterwards. An `int32` edge
payload is count `R` while the edge outcome tag is `NO_OUTCOME`, then certified
distance after the edge becomes categorical. The node `int32` payload likewise
holds `n_down` while unresolved and distance once categorical. The `int8`
outcome tags are the mandatory discriminants. Thompson selection reads `B`
when unresolved `R > 0`, an expanded child's value prior otherwise, and the
Q fallback last. Counts are structural and never alter alpha concentration.

Expansion has one native terminal contract. `RecurrentFnOutput.value` and
`action_values` are the child's model alphas, while `terminal_outcome` is an
exact outcome index from the child's `to_play` perspective, or `NO_OUTCOME`
for a non-terminal child. A terminal expansion publishes the exact tag and
distance zero on the child plus the aligned distance-one certificate on its
incoming edge. It does not synthesize a concentrated terminal alpha.

Exact outcomes remain separate `int8` tags, while categorical distances reuse
the former support/count payload. A terminal node is exactly a categorical
node with payload zero; an unresolved node can also have zero support, so the
tag must always be checked. Its incoming edge has distance one. Certificates
are authoritative—a categorical value is never inferred from an alpha vector.
A certified edge is excluded from traversal but uses exact `-1/0/+1` utility
when a mixed categorical/Dirichlet policy is read out.

Bottom-up repair makes a node categorical when any edge is a certified win,
or when every legal edge has been certified and the minimax result is a draw
or loss. Wins use the shortest currently certified winning edge; losses delay
defeat for the longest certified distance. All certified draw edges are
equivalent. The search samples uniformly among equally good certified edges,
including equal-distance win/loss ties, using its existing RNG. The selected
action is derived from immutable edge certificates rather than stored on the
node, and no policy reevaluation is needed. A categorical root returns a
one-hot policy on the sampled certified action, and remaining static-loop
iterations become inactive.

Backward contains no uncertain-posterior formula. At every path node it gathers
a `NodeView`, a `ChildrenView`, and a `LeafView`, calls
`posterior_update(rng_key, context)`, stores the ephemeral `PosteriorUpdate`,
and repeats toward the root. The persistent `Tree` itself is flat.
`LeafView.active` is true only at the deepest node, so the same callback owns
the direct unresolved leaf message and every child-to-parent repair. A
replacement rule can inspect the current embedding and child summaries—or
lazily gather child embeddings—without changing traversal. Terminal detection,
support accounting, and categorical minimax propagation remain fixed search
semantics outside that callback.

The default update recomputes `pi_search` from a fresh population over the
node's current, post-repair native action objects: exact utility for categorical
edges and a Thompson draw from the learned alpha for unresolved edges. The
internal population size is an estimator budget and may be smaller than the
public root-policy population; even one draw is an unbiased estimate of the
same posterior-best policy. Its production sampler is a
bounded-work, vectorized Marsaglia--Tsang Dirichlet draw matching the demo's
accept/reject mathematics. It evaluates a small proposal population in
parallel so tiny learned components do not stall nested search loops, while
keeping the sampling rule identical at traversal, node repair, and the public
root. It does not use visits, an independent Gaussian-utility rule, or a
historical policy average. While an edge is unresolved, structural `R` affects
only `n_down`, the prior/search mixing weight, and child propagation. Before a
certificate overwrites `R` with distance, its final count delta is committed
to the unresolved parent's `n_down`.

There is one scalar search constant:

```text
gamma = n_down / (kappa + n_down)
```

`kappa` is the strength of the node's learned value prior against repaired
descendants. It is not terminal evidence and does not scale leaf alphas. To
form the numeric value-cache mixture when categorical and unresolved edges
coexist, the update temporarily projects a categorical edge to
`sum(A_edge) * one_hot(outcome)`. This preserves that edge's existing learned
effective-alpha mass (from its Q fallback, child prior, or earlier message)
while imposing the sidecar's exact direction. The projection does not mutate
the stored alpha; exact sidecars still own selection, propagation, and targets.

The backend returns the raw app-style unresolved caches together with native
categorical tags. Scacchi trains unresolved Q/V targets toward the
effective alphas and cache. Categorical Q/V targets instead use the negative
log density of the predicted Dirichlet at an epsilon-interior categorical
point; neither structural support nor the temporary cache projection becomes a
target. Individual categorical-edge counts no longer exist. Evidence-mass Q
weighting reads counts only for unresolved rows; categorical rows receive their
explicit categorical weight.

Static mathematical choices belong to the update callable rather than the
tree. For example, callers can pass
`functools.partial(update_posterior, kappa=4, policy_samples=128)` or a wholly
different `(rng_key, context) -> PosteriorUpdate` function. Every configured
update uses the same complete leaf/node/children backup path.

The simulate/expand/backward organization is derived from DeepMind's MCTX,
which is distributed under the Apache License 2.0; the stored state and backup
semantics here are specialized for Dirichlet message passing.
