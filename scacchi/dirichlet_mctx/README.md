# Dirichlet MCTX

`scacchi.dirichlet_mctx` is the Thompson-search sister of MCTX.  Its public
flow deliberately matches MCTX:

```python
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
    posterior_update=dirichlet_mctx.update_posterior,
)
```

The module map is also parallel to MCTX:

- `base.py`: root, expansion, and policy-output contracts.
- `tree.py`: fixed-capacity `Tree`, node posterior, and update-view types.
- `action_selection.py`: one node-local Thompson selector used everywhere.
- `search.py`: `simulate -> expand -> bottom-up repair`.
- `policies.py`: the public `dirichlet_thompson_policy` wrapper.
- `posterior_updates.py`: the replaceable node-posterior repair rule.

The stored search state follows `tictactoe-demo/app.js`. Every edge owns a full
Dirichlet message `B` and downstream count `R`. The demo's message-present bit
is exactly `R > 0`, so the tree does not duplicate it. Every node also caches a
searched value Dirichlet. Thompson selection reads `B` when `R > 0`, an
expanded child's value prior otherwise, and the node's Q-head prior as the
final fallback.

Backward contains no posterior formula. At every path node it gathers a
`NodeView`, a `ChildrenView`, and a `LeafView`, calls
`posterior_update(rng_key, context)`, stores the returned `NodePosterior`, and
repeats toward the root. `LeafView.active` is true only at the deepest node, so
the same callback owns the direct leaf message and every child-to-parent
repair. A replacement rule can inspect the current embedding and all child
summaries—or lazily gather child embeddings—without changing traversal.

The default update recomputes `pi_search` from a fresh population over the
node's current, post-repair action alphas. The internal population size is an
estimator budget and may be smaller than the public root-policy population;
even one draw is an unbiased estimate of the same posterior-best policy. Its
production sampler is a
bounded-work, vectorized Marsaglia--Tsang Dirichlet draw matching the demo's
accept/reject mathematics. It evaluates a small proposal population in
parallel so tiny terminal components do not stall nested search loops, while
keeping the sampling rule identical at traversal, node repair, and the public
root. It does not use visits, an independent Gaussian-utility rule, or a
historical policy average. Structural `R` affects only `n_down`, the
prior/search mixing weight, and child propagation.

The backend returns the raw app-style caches. Scacchi trains Q toward the root
effective action alphas and V toward the root value cache; `R` is not added to
either concentration.

Static mathematical choices belong to the update callable rather than the
tree. For example, callers can pass
`functools.partial(update_posterior, kappa_n=4, policy_samples=128)` or a wholly
different `(rng_key, context) -> NodePosterior` function. Every configured
update uses the same complete leaf/node/children backup path.

The simulate/expand/backward organization is derived from DeepMind's MCTX,
which is distributed under the Apache License 2.0; the stored state and backup
semantics here are specialized for Dirichlet message passing.
