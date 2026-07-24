from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, cast
import warnings

from omegaconf import DictConfig, OmegaConf


class Network(StrEnum):
    aznet = "aznet"
    aznet_dirichlet = "aznet_dirichlet"
    boardlaw = "boardlaw"
    boardlaw_dirichlet = "boardlaw_dirichlet"


class RezeroKernelInit(StrEnum):
    variance_scaling = "variance_scaling"
    orthogonal = "orthogonal"


class ActionCommitmentType(StrEnum):
    posterior_argmax = "posterior_argmax"
    posterior_sample = "posterior_sample"
    search_action = "search_action"


class SearchKind(StrEnum):
    policy = "policy"
    gumbel = "gumbel"
    dirichlet_thompson = "dirichlet_thompson"


class PosteriorUpdateKind(StrEnum):
    monte_carlo = "monte_carlo"
    numerical = "numerical"


class QActionSet(StrEnum):
    positive_search_evidence_or_solved = (
        "positive_search_evidence_or_solved"
    )
    positive_posterior_policy_or_solved = (
        "positive_posterior_policy_or_solved"
    )


class QPairReduction(StrEnum):
    mean_over_selected_state_action_pairs = (
        "mean_over_selected_state_action_pairs"
    )
    # Deprecated: retained only so migrated historical configurations and
    # checkpoint metadata reproduce their original normalized weighting.
    legacy_normalized_source_weighted_mean = (
        "legacy_normalized_source_weighted_mean"
    )


class DirichletLossMode(StrEnum):
    full = "full"
    mean = "mean"


class LossMaskMode(StrEnum):
    search = "search"
    value = "value"
    # Replicate pgx examples/alphazero/train.py exactly: policy CE unmasked
    # (softmax over all actions, all frames), value loss mean(l2 * value_mask).
    pgx = "pgx"


class PolicyTargetMode(StrEnum):
    search = "search"
    winner_action = "winner_action"


class EvalBaseline(StrEnum):
    checkpoint = "checkpoint"
    pgx = "pgx"
    none = "none"
    random = "random"


def _require_ge(name: str, value: int | None, minimum: int) -> None:
    if value is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}.")


def _require_gt(name: str, value: float | int | None, minimum: float) -> None:
    if value is not None and value <= minimum:
        raise ValueError(f"{name} must be > {minimum}; got {value}.")


def _require_range(name: str, value: float, *, lower: float, upper: float) -> None:
    if not lower < value < upper:
        raise ValueError(f"{name} must be > {lower} and < {upper}; got {value}.")


@dataclass
class RunConfig:
    seed: int = 0
    max_num_iters: int = 400


@dataclass
class EnvConfig:
    id: str = "go_9x9"
    board_size: int | None = None
    num_outcomes: int | None = None

@dataclass
class ModelConfig:
    network: Network = Network.aznet
    num_channels: int = 128
    num_layers: int = 6
    compute_dtype: str = "float32"
    # TODO: remove the three config below and settle on some defaults in the code
    resnet_v2: bool = True
    legacy_dirichlet_head_init: bool = False
    dirichlet_concentration_floor: float | None = None
    dirichlet_initial_concentration: float | None = None
    rezero_kernel_init: RezeroKernelInit = RezeroKernelInit.variance_scaling

    def __post_init__(self) -> None:
        _require_gt(
            "model.dirichlet_concentration_floor",
            self.dirichlet_concentration_floor,
            0.0,
        )
        _require_gt(
            "model.dirichlet_initial_concentration",
            self.dirichlet_initial_concentration,
            0.0,
        )



@dataclass
class PolicySearchConfig:
    temperature: float = 1.0

    def __post_init__(self) -> None:
        _require_gt("search.policy.temperature", self.temperature, 0.0)


@dataclass
class GumbelSearchConfig:
    num_simulations: int = 32
    gumbel_scale: float = 1.0

    def __post_init__(self) -> None:
        _require_ge("search.gumbel.num_simulations", self.num_simulations, 1)
        _require_gt("search.gumbel.gumbel_scale", self.gumbel_scale, 0.0)


@dataclass
class MonteCarloPosteriorUpdateConfig:
    # Prior mass in gamma = n / (kappa + n) for bottom-up cache repair.
    kappa: float = 4.0
    # Shared Monte Carlo budget for internal repair and the root readout.
    policy_samples: int = 32
    policy_sample_chunk_size: int | None = 32

    def __post_init__(self) -> None:
        if not math.isfinite(self.kappa):
            raise ValueError(
                "search.dirichlet_thompson.posterior_update.monte_carlo.kappa "
                f"must be finite; got {self.kappa}."
            )
        _require_gt(
            "search.dirichlet_thompson.posterior_update.monte_carlo.kappa",
            self.kappa,
            0.0,
        )
        _require_ge(
            "search.dirichlet_thompson.posterior_update.monte_carlo."
            "policy_samples",
            self.policy_samples,
            1,
        )
        _require_ge(
            "search.dirichlet_thompson.posterior_update.monte_carlo."
            "policy_sample_chunk_size",
            self.policy_sample_chunk_size,
            1,
        )
        warnings.warn(
            "Monte Carlo posterior updates are usually sub-optimal; use them "
            "when their lower computational cost is worth the approximation.",
            UserWarning,
            stacklevel=2,
        )


@dataclass
class NumericalPosteriorUpdateConfig:
    # Prior mass in gamma = n / (kappa + n) for bottom-up cache repair.
    kappa: float = 4.0
    half_width: int = 10
    tail_scale: float = 8.0
    min_half_range: float = 6.0
    max_half_range: float = 11.0
    # An unsafe quadrature estimate falls back to winner Monte Carlo.
    fallback_policy_samples: int = 32
    fallback_policy_sample_chunk_size: int | None = 32

    def __post_init__(self) -> None:
        if not math.isfinite(self.kappa):
            raise ValueError(
                "search.dirichlet_thompson.posterior_update.numerical.kappa "
                f"must be finite; got {self.kappa}."
            )
        _require_gt(
            "search.dirichlet_thompson.posterior_update.numerical.kappa",
            self.kappa,
            0.0,
        )
        _require_ge(
            "search.dirichlet_thompson.posterior_update.numerical."
            "fallback_policy_samples",
            self.fallback_policy_samples,
            1,
        )
        _require_ge(
            "search.dirichlet_thompson.posterior_update.numerical."
            "fallback_policy_sample_chunk_size",
            self.fallback_policy_sample_chunk_size,
            1,
        )
        _require_ge(
            "search.dirichlet_thompson.posterior_update.numerical.half_width",
            self.half_width,
            1,
        )
        for name, value in (
            (
                "search.dirichlet_thompson.posterior_update.numerical."
                "tail_scale",
                self.tail_scale,
            ),
            (
                "search.dirichlet_thompson.posterior_update.numerical."
                "min_half_range",
                self.min_half_range,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0; got {value}.")
        if (
            not math.isfinite(self.max_half_range)
            or self.max_half_range < self.min_half_range
        ):
            raise ValueError(
                "search.dirichlet_thompson.posterior_update.numerical."
                "max_half_range must be finite and >= min_half_range; got "
                f"{self.max_half_range} and {self.min_half_range}."
            )


@dataclass
class PosteriorUpdateConfig:
    kind: PosteriorUpdateKind = PosteriorUpdateKind.monte_carlo
    monte_carlo: MonteCarloPosteriorUpdateConfig = field(
        default_factory=MonteCarloPosteriorUpdateConfig
    )
    numerical: NumericalPosteriorUpdateConfig = field(
        default_factory=NumericalPosteriorUpdateConfig
    )

    def select(
        self,
        kind: PosteriorUpdateKind | None = None,
    ) -> MonteCarloPosteriorUpdateConfig | NumericalPosteriorUpdateConfig:
        selected = self.kind if kind is None else kind
        return cast(
            MonteCarloPosteriorUpdateConfig | NumericalPosteriorUpdateConfig,
            getattr(self, str(selected)),
        )

    def active(
        self,
    ) -> MonteCarloPosteriorUpdateConfig | NumericalPosteriorUpdateConfig:
        return self.select()


@dataclass
class DirichletThompsonSearchConfig:
    num_simulations: int = 32
    max_depth: int | None = None
    posterior_update: PosteriorUpdateConfig = field(
        default_factory=PosteriorUpdateConfig
    )

    def __post_init__(self) -> None:
        _require_ge(
            "search.dirichlet_thompson.num_simulations",
            self.num_simulations,
            0,
        )
        if self.max_depth is None:
            self.max_depth = self.num_simulations
        minimum_depth = 1 if self.num_simulations > 0 else 0
        _require_ge(
            "search.dirichlet_thompson.max_depth",
            self.max_depth,
            minimum_depth,
        )


@dataclass
class SearchConfig:
    kind: SearchKind = SearchKind.gumbel
    policy: PolicySearchConfig = field(default_factory=PolicySearchConfig)
    gumbel: GumbelSearchConfig = field(default_factory=GumbelSearchConfig)
    dirichlet_thompson: DirichletThompsonSearchConfig = field(default_factory=DirichletThompsonSearchConfig)

    def active(self) -> PolicySearchConfig | GumbelSearchConfig | DirichletThompsonSearchConfig:
        return cast(
            PolicySearchConfig | GumbelSearchConfig | DirichletThompsonSearchConfig,
            getattr(self, str(self.kind)),
        )

    def value(self, name: str, default: Any) -> Any:
        return getattr(self.active(), name, default)


@dataclass
class ActionCommitmentConfig:
    kind: ActionCommitmentType = ActionCommitmentType.posterior_argmax
    # None reuses the updater selected by Dirichlet Thompson search.
    posterior_update: PosteriorUpdateKind | None = None
    posterior_sample_temperature: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.posterior_sample_temperature)
            or self.posterior_sample_temperature <= 0.0
        ):
            raise ValueError(
                "action_commitment.posterior_sample_temperature must be "
                "finite and > 0; got "
                f"{self.posterior_sample_temperature}."
            )


@dataclass
class SelfplayConfig:
    batch_size: int = 1024
    max_num_steps: int = 256
    search: SearchConfig = field(default_factory=SearchConfig)
    action_commitment: ActionCommitmentConfig = field(
        default_factory=ActionCommitmentConfig
    )

    def __post_init__(self) -> None:
        _require_ge("selfplay.batch_size", self.batch_size, 1)


@dataclass
class QSupervisionConfig:
    action_set: QActionSet = (
        QActionSet.positive_search_evidence_or_solved
    )
    reduction: QPairReduction = (
        QPairReduction.mean_over_selected_state_action_pairs
    )


@dataclass
class TrainingLossConfig:
    policy_weight: float = 1.0
    value_dir_kl_weight: float = 0.0
    q_dir_kl_weight: float = 0.0
    value_outcome_weight: float = 0.0
    q_outcome_weight: float = 0.0
    q_supervision: QSupervisionConfig = field(
        default_factory=QSupervisionConfig
    )
    # ``full`` trains the coupled Dirichlet density; ``mean`` ignores evidence
    # mass.
    dirichlet_loss_mode: DirichletLossMode = DirichletLossMode.full
    loss_mask_mode: LossMaskMode = LossMaskMode.search
    terminal_edge_targets: bool = False
    terminal_parent_targets: bool = False
    policy_target_mode: PolicyTargetMode = PolicyTargetMode.search
    categorical_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        for name, value in (
            ("training.losses.policy_weight", self.policy_weight),
            ("training.losses.value_dir_kl_weight", self.value_dir_kl_weight),
            ("training.losses.q_dir_kl_weight", self.q_dir_kl_weight),
            ("training.losses.value_outcome_weight", self.value_outcome_weight),
            ("training.losses.q_outcome_weight", self.q_outcome_weight),
        ):
            _require_gt(name, value, -1.0)
        _require_range(
            "training.losses.categorical_epsilon",
            self.categorical_epsilon,
            lower=0.0,
            upper=1.0 / 3.0,
        )

    def active_dirichlet_weights(self) -> list[str]:
        weights = (
            ("value_dir_kl_weight", self.value_dir_kl_weight),
            ("q_dir_kl_weight", self.q_dir_kl_weight),
            ("value_outcome_weight", self.value_outcome_weight),
            ("q_outcome_weight", self.q_outcome_weight),
        )
        return [f"{name}={value}" for name, value in weights if value != 0.0]


@dataclass
class TrainingRegularizationConfig:
    dirichlet_concentration_clip: float | None = 8.0

    def __post_init__(self) -> None:
        if (
            self.dirichlet_concentration_clip is not None
            and not math.isfinite(self.dirichlet_concentration_clip)
        ):
            raise ValueError(
                "training.regularization.dirichlet_concentration_clip must "
                "be finite when set; got "
                f"{self.dirichlet_concentration_clip}."
            )
        _require_gt(
            "training.regularization.dirichlet_concentration_clip",
            self.dirichlet_concentration_clip,
            0.0,
        )


@dataclass
class TrainingConfig:
    batch_size: int = 4096
    max_updates_per_iter: int | None = None
    learning_rate: float = 0.001
    lr_decay_after_iters: int | None = None
    lr_decay_factor: float = 1.0
    grad_clip_norm: float | None = None
    losses: TrainingLossConfig = field(default_factory=TrainingLossConfig)
    regularization: TrainingRegularizationConfig = field(
        default_factory=TrainingRegularizationConfig
    )

    def __post_init__(self) -> None:
        _require_ge("training.batch_size", self.batch_size, 1)
        _require_ge("training.max_updates_per_iter", self.max_updates_per_iter, 1)
        _require_gt("training.learning_rate", self.learning_rate, 0.0)
        _require_ge("training.lr_decay_after_iters", self.lr_decay_after_iters, 1)
        _require_gt("training.lr_decay_factor", self.lr_decay_factor, 0.0)
        _require_gt("training.grad_clip_norm", self.grad_clip_norm, 0.0)


@dataclass
class EvalConfig:
    interval: int = 5
    batch_size: int = 16
    player_search: SearchConfig = field(default_factory=SearchConfig)
    baseline_search: SearchConfig = field(default_factory=SearchConfig)
    player_action_commitment: ActionCommitmentConfig = field(
        default_factory=ActionCommitmentConfig
    )
    baseline_action_commitment: ActionCommitmentConfig = field(
        default_factory=ActionCommitmentConfig
    )
    baseline: EvalBaseline = EvalBaseline.checkpoint
    baseline_id: str | None = None
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        _require_ge("eval.batch_size", self.batch_size, 1)


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "scacchi-az"


@dataclass
class LoggingConfig:
    interval: int = 1
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class CheckpointingConfig:
    max_to_keep: int | None = 3
    save_interval_steps: int = 50
    directory: str | None = None

    def __post_init__(self) -> None:
        _require_ge("checkpointing.max_to_keep", self.max_to_keep, 0)
        _require_ge("checkpointing.save_interval_steps", self.save_interval_steps, 1)


@dataclass
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)

    def __post_init__(self) -> None:
        dirichlet_networks = {Network.aznet_dirichlet, Network.boardlaw_dirichlet}
        active_weights = self.training.losses.active_dirichlet_weights()
        if self.model.network not in dirichlet_networks and active_weights:
            weights = ", ".join(active_weights)
            raise ValueError(
                "Dirichlet loss weights require a Dirichlet network; "
                f"got network={self.model.network!r} with {weights}. Set these "
                "weights to 0.0 or use model.network='aznet_dirichlet' or "
                "model.network='boardlaw_dirichlet'."
            )
        for name, search in (
            ("selfplay.search", self.selfplay.search),
            ("eval.player_search", self.eval.player_search),
        ):
            if search.kind == SearchKind.dirichlet_thompson:
                assert self.model.network in dirichlet_networks, (
                    f"{name}.kind={search.kind!r} requires "
                    "model.network='aznet_dirichlet' or "
                    "model.network='boardlaw_dirichlet'."
                )
        for name, search in (
            ("selfplay.search", self.selfplay.search),
            ("eval.player_search", self.eval.player_search),
            ("eval.baseline_search", self.eval.baseline_search),
        ):
            if search.kind != SearchKind.dirichlet_thompson:
                continue
            dirichlet = search.dirichlet_thompson
            numerical_update_selected = (
                dirichlet.posterior_update.kind == PosteriorUpdateKind.numerical
            )
            if numerical_update_selected and self.env.num_outcomes != 2:
                raise ValueError(
                    f"{name}.dirichlet_thompson numerical posterior update "
                    "requires env.num_outcomes=2; use monte_carlo for "
                    "non-binary outcome heads."
                )
        for name, search, commitment in (
            (
                "selfplay.action_commitment",
                self.selfplay.search,
                self.selfplay.action_commitment,
            ),
            (
                "eval.player_action_commitment",
                self.eval.player_search,
                self.eval.player_action_commitment,
            ),
            (
                "eval.baseline_action_commitment",
                self.eval.baseline_search,
                self.eval.baseline_action_commitment,
            ),
        ):
            if (
                commitment.posterior_update is not None
                and search.kind != SearchKind.dirichlet_thompson
            ):
                raise ValueError(
                    f"{name}.posterior_update requires "
                    "dirichlet_thompson search."
                )
            if search.kind != SearchKind.dirichlet_thompson:
                continue
            selected_update = (
                search.dirichlet_thompson.posterior_update.kind
                if commitment.posterior_update is None
                else commitment.posterior_update
            )
            if (
                selected_update == PosteriorUpdateKind.numerical
                and self.env.num_outcomes != 2
            ):
                raise ValueError(
                    f"{name} numerical posterior update requires "
                    "env.num_outcomes=2; use monte_carlo for non-binary "
                    "outcome heads."
                )
        losses = self.training.losses
        search_emits_categorical = (
            self.selfplay.search.kind == SearchKind.dirichlet_thompson
        )
        categorical_head_loss_active = (
            losses.value_dir_kl_weight > 0.0
            and (search_emits_categorical or losses.terminal_parent_targets)
        ) or (
            losses.q_dir_kl_weight > 0.0
            and (search_emits_categorical or losses.terminal_edge_targets)
        )
        if (
            categorical_head_loss_active
            and self.training.regularization.dirichlet_concentration_clip is None
        ):
            raise ValueError(
                "categorical Dirichlet-density NLL requires a finite "
                "training.regularization.dirichlet_concentration_clip; "
                "epsilon moves the target off the simplex boundary but does "
                "not bound the density optimum."
            )
        if self.eval.baseline == EvalBaseline.none and self.eval.interval != 0:
            raise ValueError("eval.baseline=none requires eval.interval=0.")


Config = TrainConfig


def _apply_config_aliases(cfg: DictConfig) -> DictConfig:
    aliased = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(aliased, DictConfig):
        return cfg

    def copy_node(value: Any) -> Any:
        return OmegaConf.create(OmegaConf.to_container(value, resolve=False))

    training = aliased.get("training")
    if isinstance(training, DictConfig):
        losses = training.get("losses")
        if isinstance(losses, DictConfig):
            old_fields = {
                "q_loss_weight_mode",
                "q_dir_kl_reduction",
            } & set(losses.keys())
            if old_fields and "q_supervision" in losses:
                fields = ", ".join(sorted(old_fields))
                raise ValueError(
                    "training.losses cannot mix q_supervision with deprecated "
                    f"Q-supervision fields: {fields}."
                )
            if old_fields:
                old_action_source = str(
                    losses.pop("q_loss_weight_mode", "policy")
                )
                old_reduction = str(
                    losses.pop("q_dir_kl_reduction", "weighted")
                )
                action_set = {
                    "evidence_mass": (
                        "positive_search_evidence_or_solved"
                    ),
                    "policy": (
                        "positive_posterior_policy_or_solved"
                    ),
                }.get(old_action_source)
                reduction = {
                    "masked_mean": (
                        "mean_over_selected_state_action_pairs"
                    ),
                    "weighted": (
                        "legacy_normalized_source_weighted_mean"
                    ),
                }.get(old_reduction)
                if action_set is None or reduction is None:
                    raise ValueError(
                        "cannot migrate deprecated Q-supervision fields "
                        "training.losses.q_loss_weight_mode="
                        f"{old_action_source!r}, "
                        "training.losses.q_dir_kl_reduction="
                        f"{old_reduction!r}."
                    )
                warnings.warn(
                    "training.losses.q_loss_weight_mode and "
                    "training.losses.q_dir_kl_reduction are deprecated; use "
                    "training.losses.q_supervision.action_set and "
                    "training.losses.q_supervision.reduction.",
                    FutureWarning,
                    stacklevel=3,
                )
                losses["q_supervision"] = {
                    "action_set": action_set,
                    "reduction": reduction,
                }

    selfplay = aliased.get("selfplay")
    if selfplay is None:
        aliased["selfplay"] = OmegaConf.create({})
        selfplay = aliased["selfplay"]
    eval_cfg = aliased.get("eval")
    if eval_cfg is None:
        aliased["eval"] = OmegaConf.create({})
        eval_cfg = aliased["eval"]

    search_alias = aliased.get("search")
    if search_alias is None and isinstance(selfplay, DictConfig) and "search" in selfplay:
        search_alias = selfplay["search"]
        aliased["search"] = copy_node(search_alias)
    if search_alias is None and isinstance(eval_cfg, DictConfig) and "player_search" in eval_cfg:
        search_alias = eval_cfg["player_search"]
        aliased["search"] = copy_node(search_alias)

    if search_alias is not None:
        if isinstance(selfplay, DictConfig) and "search" not in selfplay:
            selfplay["search"] = copy_node(search_alias)
        if isinstance(eval_cfg, DictConfig):
            if "player_search" not in eval_cfg:
                eval_cfg["player_search"] = copy_node(search_alias)
            if "baseline_search" not in eval_cfg:
                eval_cfg["baseline_search"] = copy_node(search_alias)
    elif isinstance(eval_cfg, DictConfig) and "player_search" in eval_cfg:
        if "baseline_search" not in eval_cfg:
            eval_cfg["baseline_search"] = copy_node(eval_cfg["player_search"])

    def migrate_action_commitment(
        owner: Any,
        *,
        commitment_name: str,
        legacy_kind_name: str,
        search_name: str,
    ) -> None:
        if not isinstance(owner, DictConfig):
            return
        search = owner.get(search_name)
        legacy_temperature = None
        legacy_posterior_update = None
        if isinstance(search, DictConfig):
            legacy_temperature = search.pop(
                "posterior_sample_temperature",
                None,
            )
            dirichlet = search.get("dirichlet_thompson")
            if isinstance(dirichlet, DictConfig):
                legacy_action_estimator = dirichlet.pop(
                    "root_action_estimator",
                    None,
                )
                # Replay root behavior now follows the search updater.
                dirichlet.pop("root_policy_target_estimator", None)
                if legacy_action_estimator is not None:
                    legacy_posterior_update = {
                        "winner_mc": "monte_carlo",
                        "prefix_cdf": "numerical",
                    }.get(str(legacy_action_estimator))
                    if legacy_posterior_update is None:
                        raise ValueError(
                            "cannot migrate deprecated "
                            "root_action_estimator="
                            f"{legacy_action_estimator!r}."
                        )
        legacy_kind = owner.pop(legacy_kind_name, None)
        if (
            legacy_kind is None
            and legacy_temperature is None
            and legacy_posterior_update is None
        ):
            return
        commitment = owner.get(commitment_name)
        if commitment is None:
            owner[commitment_name] = OmegaConf.create({})
            commitment = owner[commitment_name]
        if not isinstance(commitment, DictConfig):
            raise ValueError(f"{commitment_name} must be a mapping.")
        if legacy_kind is not None:
            if "kind" in commitment:
                raise ValueError(
                    f"cannot mix {commitment_name}.kind with deprecated "
                    f"{legacy_kind_name}."
                )
            commitment["kind"] = legacy_kind
        if legacy_temperature is not None:
            if "posterior_sample_temperature" in commitment:
                raise ValueError(
                    f"cannot mix {commitment_name}."
                    "posterior_sample_temperature with deprecated "
                    f"{search_name}.posterior_sample_temperature."
                )
            commitment["posterior_sample_temperature"] = legacy_temperature
        if legacy_posterior_update is not None:
            if "posterior_update" in commitment:
                raise ValueError(
                    f"cannot mix {commitment_name}.posterior_update with "
                    "deprecated root_action_estimator."
                )
            commitment["posterior_update"] = legacy_posterior_update

    migrate_action_commitment(
        selfplay,
        commitment_name="action_commitment",
        legacy_kind_name="action_commitment_type",
        search_name="search",
    )
    migrate_action_commitment(
        eval_cfg,
        commitment_name="player_action_commitment",
        legacy_kind_name="player_action_commitment_type",
        search_name="player_search",
    )
    migrate_action_commitment(
        eval_cfg,
        commitment_name="baseline_action_commitment",
        legacy_kind_name="baseline_action_commitment_type",
        search_name="baseline_search",
    )
    top_level_search = aliased.get("search")
    if isinstance(top_level_search, DictConfig):
        top_level_search.pop("posterior_sample_temperature", None)
        top_level_dirichlet = top_level_search.get("dirichlet_thompson")
        if isinstance(top_level_dirichlet, DictConfig):
            top_level_dirichlet.pop("root_action_estimator", None)
            top_level_dirichlet.pop(
                "root_policy_target_estimator",
                None,
            )

    def default_dirichlet_max_depth(search: Any) -> None:
        if not isinstance(search, DictConfig):
            return
        dirichlet_cfg = search.get("dirichlet_thompson")
        if not isinstance(dirichlet_cfg, DictConfig):
            return
        if "num_simulations" in dirichlet_cfg and (
            "max_depth" not in dirichlet_cfg or dirichlet_cfg.get("max_depth") is None
        ):
            dirichlet_cfg["max_depth"] = dirichlet_cfg["num_simulations"]

    default_dirichlet_max_depth(aliased.get("search"))
    if isinstance(selfplay, DictConfig):
        default_dirichlet_max_depth(selfplay.get("search"))
    if isinstance(eval_cfg, DictConfig):
        default_dirichlet_max_depth(eval_cfg.get("player_search"))
        default_dirichlet_max_depth(eval_cfg.get("baseline_search"))
    return aliased


def load_config(cfg: DictConfig) -> TrainConfig:
    cfg = _apply_config_aliases(cfg)
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg)
    return cast(TrainConfig, OmegaConf.to_object(merged))


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(
        OmegaConf.structured(config),
        resolve=True,
        enum_to_str=True,
    )
    return cast(dict[str, Any], container)
