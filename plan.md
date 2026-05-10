# Store Posterior Dirichlet Targets Directly

## Summary

- Self-play/search produces fixed posterior targets and training consumes them directly.
- The train step does not reconstruct targets from current model outputs.
- Root Thompson selection keeps live posterior updates inside the MCTX block.
- Non-terminal leaves contribute `c_leaf` value-head pseudo-evidence.

## Key Changes

- Change value and Q Dirichlet heads to `alpha = exp(t) * softmax(r)`, with no additive base prior.
- Keep root selection using `alpha_base + current_tree_evidence`.
- Use terminal evidence as `c_terminal * one_hot(outcome)` and non-terminal evidence as `c_leaf * value_head_mean`.
- Add optional repeated MCTX root-search blocks with default `num_search_blocks = 1`, carrying posterior alphas between blocks.

## Target Data Flow

- Compute `beta_Q_target = alpha_Q_search_prior + q_evidence_sum` during self-play.
- Compute `v_evidence_sum = sum_a policy_target[a] * q_evidence_sum[a]`.
- Compute `beta_V_target = alpha_V_search_prior + v_evidence_sum`.
- Carry `beta_V_target`, `beta_Q_target`, the value evidence mask, and the Q evidence mask into training samples.

## Losses

- Train policy with NLL against the posterior-best policy target.
- Train value with `KL(Dir(stopgrad(beta_V_target)) || Dir(alpha_V_current))`.
- Train Q with `KL(Dir(stopgrad(beta_Q_target)) || Dir(alpha_Q_current))` over actions with evidence.
- Remove temporary outcome cross-entropy, search-mean cross-entropy, entropy loss, and `1 + evidence` targets.

## Tests

- Verify root selection uses `alpha_base + evidence`.
- Verify terminal and non-terminal evidence weights and perspective flips.
- Verify stored `beta_*_target` values are used directly by the loss.
- Verify the network Dirichlet parameterization is `exp(t) * softmax(r)`.
- Run `uv run pytest -q`.
