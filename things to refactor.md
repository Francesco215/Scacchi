# Things To Refactor

## Literal/config audit

- Runtime config now uses nested dataclasses loaded through Hydra/OmegaConf structured config. The old flat config field table, manual type coercion tables, and `normalize_config_dict()` shim are gone.
- Runtime call sites now pass sub-configs at the main boundaries: model, logger, checkpointing, evaluation, self-play, training, and loss setup.
- `NetworkName`: keep `boardlaw_dirichlet`. Removed runtime support for `aznet` and `boardlaw`; those do not produce the policy, value-Dirichlet, and Q-Dirichlet outputs required by `algorithms.tex`.
- `SearchPolicy`: keep `posterior_tree_wavefront`. Removed runtime support for `gumbel`, `dirichlet_thompson`, and the old non-arena `posterior_tree` path.
- `SelfplayActionSource`: keep `search_action`. The search driver already returns the committed action from `FinishSearch`; a second self-play action selector is not part of the algorithm.
- `LeafValueMode`: keep `alpha` and `mean`. Section 7.2 explicitly allows `mean` as an ablation, with `alpha` as the Bayesian/default mode.
- `FinalActionMode`: keep `posterior_argmax` and `posterior_sample`. Section 15 lists exactly these non-categorical committed-action modes.
- `CategoricalDrawRule`: keep `policy_prior`, `fastest_draw`, `slowest_draw`, and `fixed_order`. Section 7.4 lists these as valid draw-action rules.
- `PolicyTargetMode`: removed. `winner_action` is incompatible with Section 17.1; policy targets are categorical one-hot for solved nodes or posterior-best search targets otherwise.
- `DuplicateLeafMode`: removed. In-flight leaves are scheduler state only; duplicate leaf handling should not be a Bayesian/search semantic option.
- `training.tree.*`: removed from runtime config. Section 17 defines the export set directly: clean non-terminal interior nodes plus categorical non-terminal nodes, with terminal leaves excluded.
- The non-posterior self-play branch and Pydantic-era evaluation config mutation path have been removed.
- The dirichlet-tree search boundary now consumes the nested runtime `SearchConfig` directly. The flat `search_config_from_any()` compatibility bridge and `SimpleNamespace` test configs are removed.
- Deleted the old jitted Dirichlet-Q search, list-backed posterior-tree wrapper, exact Hex helper, and their tests. Runtime self-play/evaluation now call the native dirichlet-tree driver directly.
- External key-value storage, its tests, benchmark scripts, and package dependencies are removed. The retained search path is the in-process wavefront arena.

## Remaining implementation refactors

- Remove terminal Dirichlet proxy functions and fields from `scacchi/dirichlet_tree/arena_search.py`, `scacchi/dirichlet_tree/search.py`, and `scacchi/dirichlet_tree/backup.py`; terminal outcomes must stay native categorical objects.
- Replace internal `backup_mc_samples` plumbing with the single posterior-best sample count `policy_mc_samples`/`M_pi`.
- Rename remaining `wavefront_*` function/type names once the old non-wavefront path is gone. The algorithm is now the native posterior-tree driver, not a selectable wavefront variant.
- Remove scalar network branches (`AZNet`, `BoardlawNet`) from normal training code after migrated/evaluation checkpoints no longer need them.
