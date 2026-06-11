# Search and Play Refactor Plan

## Goal

Refactor search and gameplay around these concepts:

```python
evaluator = make_evaluator(model)
search = make_search(env, evaluator, play_mode.search)
player = Player(search, action_committer)
output = play(env, player_1, player_2, ...)
```

For self-play, `player_1` and `player_2` are the same learner player. For
evaluation, `player_1` is the model being evaluated and `player_2` is the
baseline/opponent player.

The main architectural rule is:

- search improves root estimates
- search converts model priors into posterior targets
- action commitment decides which move is actually played
- play advances the environment and records trajectory data

Search should not own action commitment or training loss reduction choices.

## Initial Scope

Focus the first implementation on the JAX/MCTX search paths:

- scalar Gumbel
- Dirichlet Gumbel
- Dirichlet Thompson

Keep DQAZ on the existing compatibility path for now. DQAZ has extra native
engine/export semantics, so it should be migrated after the search/play
boundary is working for Gumbel and Dirichlet search.

## Current Surface Area

The relevant current files are:

```text
scacchi/play.py            274 lines
scacchi/play_search.py    1497 lines
scacchi/evaluations.py     218 lines
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
- Dirichlet targets (`beta_Q_target`, `beta_V_target`, target metadata fields)
- DQAZ tree training data and diagnostics

That output can stay as the training-facing compatibility type for a while, but
internally it should be assembled from smaller stage outputs.

## Config Boundary

`SearchConfig` should be a reusable subsection of each play mode, not the single
top-level source of truth. Self-play and evaluation can legitimately use
different search settings, and the eval baseline can use a third search setting.

Search config should only describe search expansion and posterior construction
behavior.

Keep in each `SearchConfig`:

- `kind`
- gumbel: `num_simulations`, `gumbel_scale`, search constants
- Dirichlet Thompson: `num_simulations`, `num_blocks`, search constants
- DQAZ: `num_simulations`, `inflight_limit`, `state_posterior_kappa_n`,
  `eval_batch_size`, `pad_to_eval_batch`, `jax_backup`, `debug`,
  `epsilon_terminal`, search constants

Move out of search:

- action commitment
- `training.losses.q_loss_weight_mode` stays in training/loss config

Proposed play config shape:

```python
@dataclass
class SelfplayConfig:
    batch_size: int
    max_num_steps: int
    search: SearchConfig
    action_commitment_type: ActionCommitmentType


@dataclass
class EvalConfig:
    interval: int
    batch_size: int
    player_search: SearchConfig
    baseline_search: SearchConfig
    player_action_commitment_type: ActionCommitmentType
    baseline_action_commitment_type: ActionCommitmentType
    baseline: EvalBaseline
    baseline_id: str | None = None
```

Example YAML shape:

```yaml
selfplay:
  batch_size: 16384
  max_num_steps: 128
  action_commitment_type: posterior_sample
  search:
    kind: dirichlet_thompson
    dirichlet_thompson:
      num_simulations: 4
      num_blocks: 4
      posterior_policy:
        samples: 0
        sample_chunk_size: 32
      constants:
        kappa_leaf: 1.0
        kappa_terminal: 8.0

eval:
  interval: 10
  batch_size: 1024
  baseline: pgx
  baseline_id: gardner_chess_v0
  player_action_commitment_type: posterior_argmax
  baseline_action_commitment_type: posterior_argmax
  player_search:
    kind: gumbel
    gumbel:
      num_simulations: 64
      gumbel_scale: 1.0
  baseline_search:
    kind: gumbel
    gumbel:
      num_simulations: 32
      gumbel_scale: 1.0
```

The posterior policy sample fields can also stay inside the active search config
if we treat them as part of producing the posterior policy target. The important
boundary is that action commitment and loss reduction choices do not live in
search.

`posterior_best` and `posterior_argmax` should collapse to one public name. The
old spelling can remain as a compatibility alias for one migration. The current
top-level `config.search` can also stay as a temporary compatibility alias, but
new code should read `config.selfplay.search`, `config.eval.player_search`, and
`config.eval.baseline_search`.

Construction then becomes:

```python
selfplay_search = make_search(env, learner_evaluator, config.selfplay.search)
selfplay_player = make_player(selfplay_search, selfplay_committer)

eval_player_search = make_search(env, learner_evaluator, config.eval.player_search)
eval_player = make_player(eval_player_search, eval_player_committer)

baseline_search = make_search(env, baseline_evaluator, config.eval.baseline_search)
baseline_player = make_player(baseline_search, baseline_committer)
```

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

This can start as a model-bound helper function rather than a class:

```python
evaluator(obs) -> EvaluatorOutput
```

### Search Function

The bound search function is built from the environment, a model-bound
evaluator, and the selected search config for the current play mode.

```python
search = make_search(env, evaluator, play_mode_search_config)
```

Runtime signature:

```python
search_output = search(
    root_state=env_state,
    rng_key=search_key,
)
```

`make_search(...)` bakes in the expansion logic. For the current MCTX paths,
that means it builds the same recurrent functions already used in the code:

```python
make_recurrent_fn(env, evaluator)
make_dirichlet_recurrent_fn_from_constants(env, evaluator, search_cfg.constants)
```

With the new evaluator contract, these factories keep their role but consume
`EvaluatorOutput` fields instead of unpacking raw model tuples directly. The
Dirichlet recurrent builder should take the active search constants directly,
not the full global config.

The caller should not pass a separate raw `predict_fn` or recurrent function.
`make_search(...)` uses `search_cfg` to choose the scalar, Dirichlet, or later
DQAZ evaluator/search adapter internally.

Scalar Gumbel search uses the scalar recurrent function. Dirichlet Gumbel and
Dirichlet Thompson search use the Dirichlet recurrent function. DQAZ is the
exception for now: it is not an MCTX recurrent search, so its search closure can
keep adapting to the existing native posterior-tree transition/evaluation path
behind the DQAZ adapter.

Search evaluates the root position itself, runs the configured search, and
returns improved predictions shaped like the model heads plus training metadata.

```python
class SearchOutput(NamedTuple):
    posterior: PosteriorTargets
    diagnostics: SearchDiagnostics | None = None


class PosteriorTargets(NamedTuple):
    prediction: PosteriorPrediction
    metadata: TargetMetadata | None = None


class PosteriorPrediction(NamedTuple):
    policy: jax.Array
    value: jax.Array | None = None
    alpha_v: jax.Array | None = None
    alpha_q: jax.Array | None = None


class TargetMetadata(NamedTuple):
    mask: jax.Array | None = None
    q_weight: jax.Array | None = None
    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None
    tree_data: TreeTrainingData | None = None
```

Field meaning:

- `posterior.prediction.policy` is `pi_search`. It is a policy distribution,
  not logits.
- For scalar policy/value models, `posterior.prediction.value` is the searched
  scalar value target when available.
- For Dirichlet models, `posterior.prediction.alpha_v` and
  `posterior.prediction.alpha_q` are the searched Dirichlet posteriors for the
  value and Q heads.
- `metadata.mask` marks rows/actions with valid search targets.
- `metadata.q_weight` is the Q-head loss weight.
- target kind/outcome/distance/weight fields are categorical-vs-Dirichlet
  target metadata used by the current DQAZ/terminal-target loss path.
- `metadata.tree_data` contains optional extra tree rows for training.
- `diagnostics` contains search metrics only.

Search does not return `played_action`.

Behavior by search/network type:

- Scalar gumbel:
  - `posterior.prediction.policy = pi_search`
  - `posterior.prediction.value` may hold a searched scalar value target if we
    want scalar value targets from search; otherwise scalar value training can
    keep using trajectory returns.

- Dirichlet gumbel:
  - `posterior.prediction.policy = pi_search`
  - `posterior.prediction.alpha_v = beta_V_target`
  - `posterior.prediction.alpha_q = beta_Q_target`

- Dirichlet Thompson:
  - search expands with `num_simulations` and `num_blocks`
  - `posterior.prediction.policy = pi_search`
  - `posterior.prediction.alpha_v = beta_V_target`
  - `posterior.prediction.alpha_q = beta_Q_target`

- DQAZ:
  - `posterior.prediction.policy = pi_search`
  - `posterior.prediction.alpha_v = beta_V_target`
  - `posterior.prediction.alpha_q = beta_Q_target`
  - categorical solved outcomes and target weights go in `metadata`

Intermediate values like `q_evidence_sum`, `action_alpha_post`, and
`action_value_target_prior` stay inside the search implementation. They are not
part of the public play/training contract unless we explicitly add them as
diagnostics.

### Action Committer

Action commitment decides the move to play.

```python
action = action_committer(
    posterior=search_output.posterior,
    legal_action_mask=env_state.legal_action_mask,
    rng_key=commit_key,
)
```

Supported commitment types:

```text
posterior_argmax -> argmax(posterior.prediction.policy)
posterior_sample -> sample(posterior.prediction.policy)
```

The old `search_action` mode needs a migration decision. Either drop it, or keep
it as a compatibility-only mode by adding an optional backend action field to
metadata for backends that expose one. The core `SearchOutput` shape should not
grow raw search internals just for this mode.

Every mode goes through `legalize_action(action, legal_action_mask)`.

This replaces `_select_played_action`; the old function can remain as a shim
temporarily for tests.

### Player

The player composes a bound search function and action commitment. The model,
environment, evaluator, and search config are already baked into `search`.

```python
class PlayerOutput(NamedTuple):
    action: jax.Array
    posterior: PosteriorTargets | None = None
    diagnostics: SearchDiagnostics | None = None
```

For training self-play, `posterior` is present. For eval, only `action` is
required; posterior fields may be `None`.

Because batched positions can contain both current players, `play` usually has
to evaluate both players for the whole batch and select row-wise by
`env_state.current_player`. This means both players should return the same
PyTree structure for a given play mode. Eval players can both use a minimal
`PlayerOutput(action=...)` schema.

The player API can be small:

```python
player = make_player(search, committer)
output = player(env_state, key)
```

The player calls `search(env_state, search_key)`, commits from
`search_output.posterior.prediction.policy`, and stores the full posterior for
training self-play. A simple policy-only baseline can still be represented as a
different bound search function that returns a `SearchOutput` with only a policy
posterior.

### Play

Generic play should own environment progression only.

```python
training_samples, eval_metrics = play(
    env,
    player_1,
    player_2,
    rng_key,
    play_config,
)
```

`play_config` owns batch size, maximum steps, reset behavior, stop behavior, and
whether training samples or eval metrics are populated.

Step behavior:

1. Record pre-step observation and legal mask if training data is being
   collected.
2. Build a searchable state. For eval/all-done mode, terminal rows get a dummy
   legal action because their actions are discarded.
3. Split keys for player 1, player 2, commitment, and reset.
4. Evaluate both players on the searchable state.
5. Select action row-wise with `env_state.current_player`.
6. Validate/legalize action.
7. Step active rows.
8. Auto-reset rows only in fixed-horizon self-play mode.
9. Record reward, discount, termination, selected action, and selected player
   training fields.

The base output has both sides of the collection contract:

```python
class TrainingSamples(NamedTuple):
    obs: jax.Array
    posterior: PosteriorTargets
    legal_action_mask: jax.Array
    played_action: jax.Array | None = None


class EvalMetrics(NamedTuple):
    avg_return: jax.Array
    win_rate: jax.Array
    draw_rate: jax.Array
    lose_rate: jax.Array


play(...) -> tuple[TrainingSamples | None, EvalMetrics | None]
```

Then expose thin wrappers:

```python
play_training(...) -> TrainingSamples
play_eval(...) -> EvalMetrics
```

For the first implementation, keep a compatibility adapter from
`TrainingSamples` to the existing `SelfplayOutput` fields. Longer term,
`loss.py` should consume `samples.obs`, `samples.posterior.prediction`, and
`samples.posterior.metadata` directly.

`make_selfplay` becomes a wrapper around `play_training(learner, learner, ...)`.

`make_mcts_evaluate` becomes a wrapper around
`play_eval(learner, baseline, ...)`.

## Function Reduction Estimate

The estimate changes now that the first implementation deliberately keeps DQAZ
on the compatibility path. Several old helpers are ugly, but they are still
used by `_run_posterior_tree_search_step` and the native DQAZ export path, so
they should not be counted as first-pass deletions.

Current measured surface:

```text
core files:
  scacchi/play.py            274 lines
  scacchi/play_search.py    1497 lines
  scacchi/evaluations.py     218 lines
  scacchi/pipeline.py        205 lines
  scacchi/loss.py            657 lines
  scacchi/types.py           382 lines

nearby tests:
  tests/test_play.py                 180 lines
  tests/test_evaluations.py          161 lines
  tests/test_config_validation.py    353 lines
```

First-pass scope:

- add `make_search(env, evaluator, search_cfg)` for scalar Gumbel, Dirichlet
  Gumbel, and Dirichlet Thompson
- add `make_player(search, action_committer)`
- add generic `play(...)`, `play_training(...)`, and `play_eval(...)`
- move search outputs to `PosteriorTargets`
- nest search configs under `selfplay` and `eval`
- keep adapters for `SelfplayOutput`, current loss input, DQAZ, and old tests

High-confidence first-pass deletes:

```text
1. _stack_optional_tree
2. _stack_selfplay_frames
3. _concat_selfplay_time
4. _make_model_mcts_policy
5. _model_eval_action
6. _with_eval_num_simulations
7. _with_eval_search_kind
8. _baseline_eval_search_config
```

Likely first-pass replacements or shims, not hard deletes:

```text
_run_model_search             -> make_search-backed compatibility wrapper
_SearchStepOutput             -> SearchOutput adapter for old callers
_run_scalar_gumbel_search     -> scalar gumbel search closure or shim
_run_dirichlet_search         -> Dirichlet search closure or shim
_select_played_action         -> commit_action shim
_legalize_played_action       -> legalize_action shim
_search_loss_mask             -> target metadata utility
_empty_posterior_targets      -> scalar/Dirichlet posterior utility
_q_loss_weight_from_mode      -> target metadata or loss adapter utility
_native_target_kwargs_from_output -> DQAZ compatibility adapter
_searchable_eval_state        -> generic play utility
_step_active_eval_rows        -> generic play utility
_poison_eval_returns          -> eval collector utility
_concat_selfplay_outputs      -> TrainingSamples/SelfplayOutput adapter
_fixed_replay_window          -> replay buffer utility or delete if unused
```

Must stay during the DQAZ compatibility phase:

```text
_search_value
_search_constant
_search_kind
_selfplay_action_source
PosteriorTreeBatchOutput
_run_posterior_tree_search_step
_run_dqaz_posterior_tree_search
_dqaz_output_to_posterior_batch
```

Estimated deletion count:

- first pass: hard-delete about 6 to 8 functions and replace another 8 to 12
  with shims or moved utilities
- final cleanup after loss, tests, scripts, and DQAZ migrate: remove about 18 to
  25 functions/classes

Estimated first-pass line-count impact:

```text
play.py:
  current: 274 lines
  expected: 230 to 320 lines
  net: -40 to +50
  reason: old self-play helpers go away, generic play and wrappers arrive

evaluations.py:
  current: 218 lines
  expected: 45 to 80 lines
  net: -140 to -175
  reason: eval becomes a thin player/play wrapper

play_search.py:
  current: 1497 lines
  expected: 1450 to 1580 lines
  net: -50 to +80
  reason: Gumbel/Dirichlet get a cleaner factory, but DQAZ and old shims stay

new player/search/play modules:
  expected: +180 to +320 lines if split out
  note: this is absorbed into existing files if we keep the refactor local

pipeline.py:
  current: 205 lines
  expected: 180 to 240 lines
  net: -25 to +35
  reason: replay/loss adapters may temporarily offset helper deletion

loss.py:
  current: 657 lines
  expected first pass: 650 to 730 lines
  net: -10 to +70
  reason: likely keeps current Sample/SelfplayOutput path through an adapter

types.py/configs:
  current: 382 lines
  expected: 450 to 520 lines
  net: +70 to +140
  reason: nested play search configs plus temporary aliases/validation

tests:
  current nearby tests: 694 lines
  expected: 740 to 850 lines
  net: +45 to +155
  reason: new factory/player/play tests while compatibility tests still exist
```

Net first-pass estimate:

- if split into new modules: roughly +100 to +300 lines
- if kept mostly in existing files: roughly +50 to +220 lines

This pass is likely a temporary line-count increase. That is acceptable because
we are deliberately buying compatibility while changing the boundaries.

Final cleanup estimate after the compatibility layer is removed:

```text
delete old SelfplayOutput adapters and old _SearchStepOutput path:   -80 to -140
delete old eval-specific MCTS action path:                           -120 to -175
delete top-level search config aliases and old action_source names:   -30 to -70
let loss consume TrainingSamples/PosteriorTargets directly:           -40 to -120
migrate or split DQAZ helpers from generic play_search.py:            -80 to -200
```

Expected final net versus the current codebase:

- conservative cleanup: about -150 to -350 lines
- if DQAZ is split cleanly and native export adapters shrink: about -250 to
  -500 lines

The main benefit is still not raw line-count reduction. The main benefit is that
the hard boundaries become testable:

- evaluator: model output normalization
- search: prior to posterior conversion
- player: posterior to action commitment
- play: environment stepping and collection
- loss/eval: consume the appropriate collected output

## Migration Plan

### Phase 1: Introduce Stage Types And Utilities

Add the new `SearchOutput`, `PosteriorTargets`, `PosteriorPrediction`,
`TargetMetadata`, and `PlayerOutput` types.

Move or wrap these helpers without changing behavior:

- `legalize_action`
- `commit_action`
- scalar empty Dirichlet compatibility targets
- posterior policy target builder

Keep old names as compatibility shims for tests.

### Phase 2: Search Factory

Add:

```python
make_search(env, evaluator, search_cfg)
```

Dispatch by concrete config type:

```python
GumbelSearchConfig -> bound gumbel search closure
DirichletThompsonSearchConfig -> bound Dirichlet Thompson closure
```

The returned search function should not inspect the global `Config` object and
should not receive raw prediction or recurrent functions at call time. Static
config values, environment stepping, and model evaluation are closed over once.

This phase removes `_search_value`, `_search_kind`, and most dynamic active
config lookups from the new Gumbel/Dirichlet hot path. Those helpers can stay
in `play_search.py` temporarily for the DQAZ compatibility path.

### Phase 3: Posterior Metadata And Action Commitment

Add:

```python
make_action_committer(commitment_cfg)
```

Make each search backend populate `PosteriorTargets` and `TargetMetadata`
directly. The current `q_loss_weight`, `search_loss_mask`, and categorical
target fields become metadata. Keep a compatibility adapter that maps
`TrainingSamples` back to the existing `SelfplayOutput` field names until
`loss.py` is migrated.

Rename:

```text
selfplay.action_source -> selfplay.action_commitment_type
```

Keep old YAML support temporarily:

```text
action_source maps to action_commitment_type with a deprecation note
posterior_best maps to posterior_argmax
```

### Phase 4: Player

Add:

```python
make_player(search, action_committer)
```

The player should return complete posterior fields in self-play mode and
minimal fields in eval mode. Policy-only baselines can be implemented as bound
search functions that skip tree search and return masked policy predictions.

### Phase 5: Generic Play

Implement generic play with two players.

Required modes:

```text
self-play:
  players: learner, learner
  reset_mode: auto_reset
  stop_mode: fixed_steps
  wrapper: play_training(...) -> TrainingSamples

eval:
  players: learner, baseline
  reset_mode: none
  stop_mode: all_done
  wrapper: play_eval(...) -> EvalMetrics
```

Keep `make_selfplay` and `make_mcts_evaluate` as public wrappers first. They can
be simplified after callers move to `play_training` and `play_eval`.

### Phase 6: Tests And Scripts

Update tests in this order:

1. Add unit tests for `commit_action` and `legalize_action`.
2. Add unit tests for posterior construction with scalar gumbel and Dirichlet
   Thompson.
3. Update self-play tests to call the player/play API.
4. Update eval tests to call generic play utilities.
5. Leave DQAZ tests on the compatibility wrapper for the first pass.
6. Update `scripts/fig_8.py` later. It currently has a custom eval loop that the
   new player/play API should replace cleanly.

## DQAZ-Specific Caveats

DQAZ is the hardest part to make conceptually clean because the native engine
currently exports target-shaped data and selects actions in `engine.finish`.

Initial compromise:

- `make_dqaz_search` may still call the native finish/export path.
- Treat exported action as `search_action`.
- Keep external action commitment in Python for self-play/eval where possible.
- Map native exported `action_weights`, `beta_Q_target`, `beta_V_target`, and
  target metadata into `PosteriorTargets`.

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

1. Add `SearchOutput`, `PosteriorTargets`, `PosteriorPrediction`, and
   `TargetMetadata`.
2. Add a compatibility adapter from `TrainingSamples` to current
   `SelfplayOutput` target fields.
3. Add `make_search(env, evaluator, play_mode_search_cfg)`.
4. Add `commit_action(...)`.
5. Add `make_player(search, action_committer)`.
6. Make current `make_selfplay` use these pieces while still returning
   `SelfplayOutput`.

Once self-play behavior is unchanged under the new stages, introduce the
two-player `play` loop and migrate eval.
