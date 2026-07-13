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
- `tree.py`: fixed-capacity `Tree`, root `Posterior`, and summary types.
- `action_selection.py`: root Thompson and interior policy-prior selection.
- `search.py`: `simulate -> expand -> backward`.
- `policies.py`: the public `dirichlet_thompson_policy` wrapper.
- `posterior_updates.py`: the replace-prior-plus-evidence Bayesian update.

The tree stores only topology, edge visits, policy priors, state embeddings,
player/terminal metadata, and the root posterior.  Scalar-MCTS rewards,
discounts, value averages, and duplicate visit tables are intentionally absent.
Evidence is added once per simulation during backup, already aligned to the
root player's perspective.

The simulate/expand/backward organization is derived from DeepMind's MCTX,
which is distributed under the Apache License 2.0; the stored state and backup
semantics here are specialized for Dirichlet evidence.
