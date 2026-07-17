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
    terminal=env_state.terminated,
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
    categorical_draw_rule="policy_prior",
)
```

The module map is also parallel to MCTX:

- `base.py`: root, expansion, and policy-output contracts.
- `tree.py`: fixed-capacity `Tree`, node posterior, and update-view types.
- `categorical.py`: native target tags, smoothing, and categorical density NLL.
- `action_selection.py`: one node-local Thompson selector used everywhere.
- `search.py`: `simulate -> expand -> bottom-up repair`.
- `policies.py`: the public `dirichlet_thompson_policy` wrapper.
- `posterior_updates.py`: the replaceable node-posterior repair rule.

The stored search state follows `tictactoe-demo/app.js`. Every edge owns a full
Dirichlet message `B` and downstream count `R`. The demo's message-present bit
is exactly `R > 0`, so the tree does not duplicate it. Every node also caches a
searched value Dirichlet. Thompson selection reads `B` when `R > 0`, an
expanded child's value prior otherwise, and the node's Q-head prior as the
final fallback. These learned alphas represent unresolved leaves and caches;
`R` is structural and is never added to their concentration.

Expansion has one native terminal contract. `RecurrentFnOutput.value` and
`action_values` are the child's model alphas, while `terminal_outcome` is an
exact outcome index from the child's `to_play` perspective, or `NO_OUTCOME`
for a non-terminal child. A terminal expansion publishes the exact tag and
distance zero on the child plus the aligned distance-one certificate on its
incoming edge. It does not synthesize a concentrated terminal alpha.

Exact outcomes are stored beside those uncertain Dirichlet objects. Every node
and edge has categorical outcome/distance sidecars; `-1` means unresolved, a
terminal node has distance zero, and its incoming edge has distance one. These
sidecars are authoritative—a categorical certificate is never inferred from
an alpha vector. A certified edge is excluded from later traversal but uses
its exact `-1/0/+1` utility when a mixed categorical/Dirichlet policy is read
out.

Bottom-up repair makes a node categorical when any edge is a certified win,
or when every legal edge has been certified and the minimax result is a draw
or loss. Wins use the shortest currently certified winning edge; losses delay
defeat for the longest certified distance. Draws use the configured
`policy_prior`, `fastest_draw`, `slowest_draw`, or `fixed_order` rule. The
selected action is derived from immutable edge certificates rather than stored
on the node. A categorical root returns a deterministic one-hot policy, and
remaining static-loop iterations become inactive.

Backward contains no uncertain-posterior formula. At every path node it gathers a
`NodeView`, a `ChildrenView`, and a `LeafView`, calls
`posterior_update(rng_key, context)`, stores the returned `NodePosterior`, and
repeats toward the root. `LeafView.active` is true only at the deepest node, so
the same callback owns the direct unresolved leaf message and every child-to-parent
repair. A replacement rule can inspect the current embedding and all child
summaries—or lazily gather child embeddings—without changing traversal.
Terminal detection and categorical minimax propagation remain fixed search
semantics outside that replaceable callback.

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
historical policy average. Structural `R` affects only `n_down`, the
prior/search mixing weight, and child propagation.

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
categorical sidecars. Scacchi trains unresolved Q/V targets toward the
effective alphas and cache. Categorical Q/V targets instead use the negative
log density of the predicted Dirichlet at an epsilon-interior categorical
point; neither `R` nor the temporary cache projection becomes a target.

Static mathematical choices belong to the update callable rather than the
tree. For example, callers can pass
`functools.partial(update_posterior, kappa=4, policy_samples=128)` or a wholly
different `(rng_key, context) -> NodePosterior` function. Every configured
update uses the same complete leaf/node/children backup path.

The simulate/expand/backward organization is derived from DeepMind's MCTX,
which is distributed under the Apache License 2.0; the stored state and backup
semantics here are specialized for Dirichlet message passing.
