# Search and Play Refactor Plan

## Goal

Refactor search and gameplay around these concepts:

```python
nn_evaluator = make_nn_evaluator(nn_config)
search_algo = make_search_algo(search_config)
player = Player(nn_evaluator, search_algo, target_builder, action_committer)
output = play(env, player_1, player_2, ...)
```

For self-play, `player_1` and `player_2` are the same learner player. For
evaluation, `player_1` is the model being evaluated and `player_2` is the
baseline/opponent player.

The main architectural rule is:

- search improves root estimates
- target building converts search evidence into training targets
- action commitment decides which move is actually played
- play advances the environment and records trajectory data

Search should not own action commitment or loss weighting.

## Current Surface Area

The relevant current files are:

```text
scacchi/play.py            274 lines
scacchi/play_search.py    1497 lines
scacchi/evaluations.py     195 lines
scacchi/pipeline.py        205 lines
scacchi/loss.py            657 lines
scacchi/types.py           382 lines
```

Current self-play path:

```text
make_training_iteration
-> make_selfplay
-> model(obs, train=False)
-> _run_model_search(...)
-> _SearchStepOutput with played_action and targets
-> auto_reset(env.step, env.init)
-> SelfplayOutput
-> make_compute_input_for_lossfn
-> train_minibatches
```

Current eval path:

```text
make_mcts_evaluate
-> _model_eval_action for learner
-> _model_eval_action for baseline
-> _make_model_mcts_policy
-> _run_model_search(...)
-> choose action by env_state.current_player
-> _step_active_eval_rows
-> returns
```

Current DQAZ path:

```text
_run_posterior_tree_search_step
-> _run_dqaz_posterior_tree_search
-> dqaz.SearchEngine
-> PGX transition batches
-> optional JAX backup
-> engine.finish(commit=...)
-> _dqaz_output_to_posterior_batch
```

Important external callers:

- `tests/test_play.py` imports `_run_scalar_gumbel_search`,
  `_select_played_action`, and `_legalize_played_action`.
- `tests/test_play_search_tictactoe.py` and
  `tests/test_play_search_hex_dqaz.py` import `_run_posterior_tree_search_step`.
- `scripts/fig_8.py` imports `_make_model_mcts_policy` and
  `_run_posterior_tree_search_step`.

So the refactor is not only local to `play.py`; tests and `fig_8.py` need either
compatibility shims or direct migration.

## Current Couplings To Break

`_run_model_search` currently does too much:

- reads `config.search.kind`
- dispatches between scalar gumbel, Dirichlet gumbel, and Dirichlet Thompson
- reads active search config values dynamically
- computes posterior policy targets
- computes `beta_Q_target` and `beta_V_target`
- computes `q_loss_weight`
- selects `played_action`

`_run_dqaz_posterior_tree_search` also commits actions internally:

```python
commit = _selfplay_action_source(config)
results = engine.finish(tree_ids, commit=commit)
```

This mixes search result export with the acting policy. That should become an
adapter detail or be moved into external action commitment.

`SelfplayOutput` is also serving multiple roles:

- trajectory data (`obs`, `reward`, `terminated`, `discount`)
- action data (`played_action`, `legal_action_mask`)
- policy targets (`action_weights`, `search_loss_mask`)
- Dirichlet targets (`beta_Q_target`, `beta_V_target`, native target fields)
- DQAZ tree training data and diagnostics

That output can stay as the training-facing compatibility type for a while, but
internally it should be assembled from smaller stage outputs.

## Config Boundary

Search config should only describe search expansion and search backend behavior.

Keep in `search.*`:

- `kind`
- gumbel: `num_simulations`, `gumbel_scale`, search constants
- Dirichlet Thompson: `num_simulations`, `num_blocks`, search constants
- DQAZ: `num_simulations`, `inflight_limit`, `state_posterior_kappa_n`,
  `eval_batch_size`, `pad_to_eval_batch`, `jax_backup`, `debug`,
  `epsilon_terminal`, search constants

Move out of search:

- `selfplay.action_source`
- `search.*.policy_samples`
- `search.*.policy_sample_chunk_size`
- `training.losses.q_loss_weight_mode` stays in training/loss config

Proposed replacements:

```text
selfplay.action_commitment_type:
  posterior_argmax
  posterior_sample
  search_action

training.targets.posterior_policy_samples:
  int, with 0 meaning "use the search policy directly"

training.targets.posterior_policy_sample_chunk_size:
  int | None
```

`posterior_best` and `posterior_argmax` should collapse to one public name. The
old spelling can remain as a compatibility alias for one migration.

## Stage Contracts

### NN Evaluator

The evaluator owns model invocation and output normalization, not search.

```python
class EvaluatorOutput(NamedTuple):
    logits: jax.Array
    value: jax.Array | None = None
    alpha_v: jax.Array | None = None
    alpha_q: jax.Array | None = None
```

Expected behavior:

- NNX models are called as `model(obs, train=False)`.
- PGX baseline models are called as `model(obs)`.
- Scalar networks fill `logits` and `value`.
- Dirichlet networks fill `logits`, `alpha_v`, and `alpha_q`.

This can start as helper functions rather than a class:

```python
predict(model, obs) -> raw model output
normalize_model_output(raw) -> EvaluatorOutput
```

### Search Algorithm

The search algorithm is built from the active search config once.

```python
search_algo = make_search_algo(active_search_config)
```

Runtime signature:

```python
search_result = search_algo(
    root_state=env_state,
    root_eval=evaluator_output,
    transition_evaluator=transition_evaluator,
    rng_key=search_key,
)
```

`transition_evaluator(states, actions)` returns child states and child
`EvaluatorOutput`. For MCTX search this wraps `jax.vmap(env.step)` and the NN
evaluator. For DQAZ this is also the natural transition batch interface.

Search returns raw improved estimates:

```python
class SearchResult(NamedTuple):
    search_action: jax.Array
    search_policy: jax.Array
    search_loss_mask: jax.Array

    q_evidence_sum: jax.Array | None = None
    action_value_target_prior: jax.Array | None = None
    action_alpha_post: jax.Array | None = None

    beta_Q_target: jax.Array | None = None
    beta_V_target: jax.Array | None = None

    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None

    tree_data: TreeTrainingData | None = None
    diagnostics: SearchDiagnostics | None = None
```

Field meaning:

- `search_action`: raw action selected by the search algorithm.
- `search_policy`: visit policy, Gumbel policy, posterior policy, or DQAZ
  exported root policy.
- `q_evidence_sum`: evidence accumulated by Dirichlet search per root action.
- `action_value_target_prior`: action Dirichlet prior after root child prior
  correction.
- `action_alpha_post`: action posterior after prior plus evidence.
- `beta_Q_target` and `beta_V_target`: allowed for DQAZ because the native
  engine already exports target-shaped data. JAX/MCTX paths can leave these
  empty and let target building compute them.
- native target fields: DQAZ categorical/Dirichlet metadata.
- `tree_data`: optional extra rows for training.
- `diagnostics`: search metrics only.

Search does not return `played_action` and does not return `q_loss_weight`.

### Target Builder

Target building converts search evidence into training labels.

```python
targets = target_builder(
    root_eval=evaluator_output,
    search_result=search_result,
    legal_action_mask=env_state.legal_action_mask,
    rng_key=target_key,
)
```

Target output:

```python
class TrainingTargets(NamedTuple):
    action_weights: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    search_loss_mask: jax.Array

    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None
```

Behavior by search/network type:

- Scalar gumbel:
  - `action_weights = search_result.search_policy`
  - `beta_Q_target`, `beta_V_target`, and `q_loss_weight` are zero-shaped
    Dirichlet compatibility targets.

- Dirichlet gumbel:
  - `search_policy = mctx.action_weights`
  - `q_evidence_sum = q_evidence_sum_from_tree(search_tree)`
  - posterior policy target is sampled from `action_alpha_post`, unless
    `posterior_policy_samples == 0`, in which case use `search_policy`.
  - `beta_Q_target`, `beta_V_target` are computed with `posterior_targets`.

- Dirichlet Thompson:
  - search expands with `num_simulations` and `num_blocks`
  - posterior policy target is sampled from `action_alpha_post`, unless
    `posterior_policy_samples == 0`, in which case use `search_policy`
  - `beta_Q_target`, `beta_V_target` are computed with `posterior_targets`.

- DQAZ:
  - start by passing through native exported `action_weights`, `beta_Q_target`,
    `beta_V_target`, and native target fields
  - longer term, make the Rust finish/export API return raw posterior data so
    target building can own the final target policy consistently.

`q_loss_weight` is built here because it is training-facing:

```text
q_loss_weight_mode = evidence_mass -> sum(q_evidence_sum, axis=-1)
q_loss_weight_mode = policy        -> action_weights
```

For DQAZ, keep the current sparse-policy behavior initially:

```text
q_loss_weight = exported root policy over legal actions
```

Then revisit whether this should become evidence mass once native export can
provide the right evidence tensor.

### Action Committer

Action commitment decides the move to play.

```python
action = action_committer(
    search_result=search_result,
    targets=targets,
    legal_action_mask=env_state.legal_action_mask,
    rng_key=commit_key,
)
```

Supported commitment types:

```text
posterior_argmax -> argmax(targets.action_weights)
posterior_sample -> sample(targets.action_weights)
search_action    -> search_result.search_action
```

Every mode goes through `legalize_action(action, legal_action_mask)`.

This replaces `_select_played_action`; the old function can remain as a shim
temporarily for tests.

### Player

The player composes evaluator, search, target building, and action commitment.

```python
class PlayerOutput(NamedTuple):
    action: jax.Array
    action_weights: jax.Array | None = None
    beta_Q_target: jax.Array | None = None
    beta_V_target: jax.Array | None = None
    q_loss_weight: jax.Array | None = None
    search_loss_mask: jax.Array | None = None
    tree_data: TreeTrainingData | None = None
    diagnostics: SearchDiagnostics | None = None
    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None
```

For training self-play, all target fields are present. For eval, only `action`
is required; policy/target fields may be `None`.

Because batched positions can contain both current players, `play` usually has
to evaluate both players for the whole batch and select row-wise by
`env_state.current_player`. This means both players should return the same
PyTree structure for a given play mode. Eval players can both use a minimal
`PlayerOutput(action=...)` schema.

To keep NNX simple, prefer a player spec/factory over storing model state in the
player:

```python
learner_player = make_model_player(env, search_algo, target_builder, committer)
output = learner_player(model, env_state, key)
```

The baseline player can also use this signature, with `baseline_model` passed as
its model argument. A simple policy-only baseline player can ignore search and
commit from masked logits.

### Play

Generic play should own environment progression only.

```python
play_output = play(
    env,
    player_1,
    player_2,
    model_1,
    model_2,
    rng_key,
    batch_size=batch_size,
    max_steps=max_steps,
    reset_mode="auto_reset" | "none",
    stop_mode="fixed_steps" | "all_done",
    collect_mode="training" | "returns",
)
```

Step behavior:

1. Record pre-step observation and legal mask if training data is being
   collected.
2. Build a searchable state. For eval/all-done mode, terminal rows get a dummy
   legal action because their actions are discarded.
3. Split keys for player 1, player 2, targets, commitment, and reset.
4. Evaluate both players on the searchable state.
5. Select action row-wise with `env_state.current_player`.
6. Validate/legalize action.
7. Step active rows.
8. Auto-reset rows only in fixed-horizon self-play mode.
9. Record reward, discount, termination, selected action, and selected player
   training fields.

Training/self-play output should remain compatible with `SelfplayOutput`:

```python
class TrainingPlayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    action_weights: jax.Array
    played_action: jax.Array
    legal_action_mask: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    discount: jax.Array
    tree_data: TreeTrainingData | None = None
    search_loss_mask: jax.Array | None = None
    search_diagnostics: SearchDiagnostics | None = None
    native target fields...
```

Evaluation output can be smaller:

```python
class EvalPlayOutput(NamedTuple):
    returns: jax.Array
    invalid_action: jax.Array
    num_steps: jax.Array | None = None
```

`make_selfplay` becomes a wrapper around `play(learner, learner, ...)`.

`make_mcts_evaluate` becomes a wrapper around
`play(learner, baseline, ..., collect_mode="returns")`.

## Function Reduction Estimate

High-confidence functions/classes to remove or replace after the first complete
migration:

```text
1.  _search_value
2.  _search_constant
3.  _search_kind
4.  _selfplay_action_source
5.  _run_model_search
6.  _make_model_mcts_policy
7.  _model_eval_action
8.  _with_eval_num_simulations
9.  _stack_optional_tree
10. _stack_selfplay_frames
11. _concat_selfplay_time
12. _concat_selfplay_outputs
13. _fixed_replay_window
14. _SearchStepOutput
15. PosteriorTreeBatchOutput
```

Likely moved/renamed rather than deleted:

```text
_select_played_action        -> commit_action
_legalize_played_action      -> legalize_action
_q_loss_weight_from_mode     -> build_q_loss_weight
_search_loss_mask            -> target/search utility
_empty_posterior_targets     -> scalar target builder utility
make_recurrent_fn            -> search/evaluator transition helper
make_dirichlet_recurrent_fn  -> search/evaluator transition helper
_searchable_eval_state       -> play utility
_step_active_eval_rows       -> play utility
_poison_eval_returns         -> eval collector utility
_run_posterior_tree_search_step -> DQAZ search factory or compatibility shim
```

Estimated deletion count:

- first pass with compatibility shims: remove 8 to 10 functions
- final cleanup after tests/scripts migrate: remove 15 to 18 functions/classes

Estimated line-count impact:

```text
play.py:
  current: 274 lines
  expected: 220 to 300 lines depending on whether generic play lives here
  net: -50 to +30

evaluations.py:
  current: 195 lines
  expected: 60 to 90 lines
  net: -105 to -135

play_search.py:
  current: 1497 lines
  expected: 1250 to 1350 lines if target/action dispatch moves out
  net in file: -150 to -250
  note: much of this is DQAZ backend code and remains

new player/search/target modules:
  expected: +250 to +450 lines

pipeline.py:
  current: 205 lines
  expected: 155 to 190 lines if unused replay helpers are removed
  net: -15 to -50

types.py/configs/tests:
  expected: +30 to +80 lines for renamed config fields and aliases
```

Net estimate:

- compatibility phase: roughly -50 to +150 lines
- final cleanup phase: roughly -100 to -250 lines

The main benefit is not raw line-count reduction. The main benefit is removing
semantic coupling and making the data flow testable stage by stage.

## Migration Plan

### Phase 1: Introduce Stage Types And Utilities

Add the new `SearchResult`, `TrainingTargets`, and `PlayerOutput` types.

Move or wrap these helpers without changing behavior:

- `legalize_action`
- `commit_action`
- `build_q_loss_weight`
- scalar empty Dirichlet compatibility targets
- posterior policy target builder

Keep old names as compatibility shims for tests.

### Phase 2: Search Factory

Add:

```python
make_search_algo(search_cfg)
```

Dispatch by concrete config type:

```python
GumbelSearchConfig -> gumbel search closure
DirichletThompsonSearchConfig -> Dirichlet Thompson closure
DQAZSearchConfig -> DQAZ search closure
```

The returned search function should not inspect the global `Config` object.
Static config values should be closed over once.

This phase removes `_search_value`, `_search_kind`, and most dynamic active
config lookups from the hot path.

### Phase 3: Target Builder And Action Commitment

Add:

```python
make_target_builder(target_cfg, loss_cfg, env_cfg)
make_action_committer(commitment_cfg)
```

Move `policy_samples` and `policy_sample_chunk_size` into target config.

Rename:

```text
selfplay.action_source -> selfplay.action_commitment_type
```

Keep old YAML support temporarily:

```text
action_source maps to action_commitment_type with a deprecation note
posterior_best maps to posterior_argmax
```

### Phase 4: Model Player

Add:

```python
make_model_player(env, evaluator, search_algo, target_builder, action_committer)
```

The model player should return complete training fields in self-play mode and
minimal fields in eval mode.

Also add a policy-only player for PGX baselines if we want eval without MCTS for
the opponent. If current behavior must be preserved, the baseline player should
use the same search player path as the learner.

### Phase 5: Generic Play

Implement generic play with two players.

Required modes:

```text
self-play:
  players: learner, learner
  reset_mode: auto_reset
  stop_mode: fixed_steps
  collect_mode: training

eval:
  players: learner, baseline
  reset_mode: none
  stop_mode: all_done
  collect_mode: returns
```

Keep `make_selfplay` and `make_mcts_evaluate` as public wrappers first. They can
be simplified after callers move to `play`.

### Phase 6: Tests And Scripts

Update tests in this order:

1. Add unit tests for `commit_action` and `legalize_action`.
2. Add unit tests for target building with scalar gumbel and Dirichlet
   Thompson.
3. Update self-play tests to call the player/play API.
4. Update eval tests to call generic play utilities.
5. Update DQAZ tests to use `make_search_algo(DQAZSearchConfig(...))` or keep a
   compatibility wrapper until the native export contract is cleaned up.
6. Update `scripts/fig_8.py` last. It currently has a custom eval loop that the
   new player/play API should replace cleanly.

## DQAZ-Specific Caveats

DQAZ is the hardest part to make conceptually clean because the native engine
currently exports target-shaped data and selects actions in `engine.finish`.

Initial compromise:

- `make_dqaz_search` may still call the native finish/export path.
- Treat exported action as `search_action`.
- Keep external action commitment in Python for self-play/eval where possible.
- Pass native `action_weights`, `beta_Q_target`, `beta_V_target`, and target
  metadata through the target builder.

Longer-term cleanup:

- Change native export so `finish` does not need a commitment type.
- Export root posterior/policy data independently from committed action.
- Let Python action commitment choose the final move for all search algorithms.

## Risks

JAX/NNX tracing:

- closures over static config are fine, but changing `num_simulations`,
  `num_blocks`, or posterior sample counts recompiles
- returned PyTree schemas must be stable across `jax.lax.scan` and
  `nnx.while_loop`

Heterogeneous players:

- batched play may contain rows for both players at once
- easiest implementation evaluates both players every step and selects row-wise
- both player outputs need the same structure for fields that are collected

Terminal eval rows:

- current eval gives terminal rows a dummy legal action and discards their moves
- generic play needs to preserve that behavior

Training compatibility:

- `loss.py` currently consumes `SelfplayOutput`
- keep a compatibility output with the same fields until the loss input builder
  is refactored separately

Config compatibility:

- configs and tests currently use `selfplay.action_source`
- keep alias handling for one migration to avoid breaking every config at once

## Recommended First Patch

Do not start by rewriting `play`.

Start with the search boundary:

1. Add `SearchResult`.
2. Add `TrainingTargets`.
3. Add `make_search_algo(active_search_cfg)`.
4. Add `make_target_builder(...)`.
5. Add `commit_action(...)`.
6. Make current `make_selfplay` use these pieces while still returning
   `SelfplayOutput`.

Once self-play behavior is unchanged under the new stages, introduce the
two-player `play` loop and migrate eval.
