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


class OptimizerType(StrEnum):
    adam = "adam"
    muon = "muon"
    psgd = "psgd"


class ActionCommitmentType(StrEnum):
    posterior_argmax = "posterior_argmax"
    posterior_sample = "posterior_sample"
    search_action = "search_action"


class SearchKind(StrEnum):
    policy = "policy"
    gumbel = "gumbel"
    dirichlet_thompson = "dirichlet_thompson"


class QLossWeightMode(StrEnum):
    policy = "policy"
    evidence_mass = "evidence_mass"


class QDirKLReduction(StrEnum):
    weighted = "weighted"
    masked_mean = "masked_mean"


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
class DirichletThompsonSearchConfig:
    num_simulations: int = 32
    max_depth: int | None = None
    # Prior mass in gamma = n / (kappa + n) for bottom-up cache repair.
    kappa: float = 4.0
    policy_samples: int = 32
    # Monte Carlo budget for pi_search inside each bottom-up node repair.
    # None keeps the public root-policy budget.  A smaller value remains an
    # unbiased Thompson estimate and avoids multiplying every path repair by
    # the full root readout budget.
    posterior_policy_samples: int | None = None
    policy_sample_chunk_size: int | None = 32
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
        if not math.isfinite(self.kappa):
            raise ValueError(
                "search.dirichlet_thompson.kappa must be finite; "
                f"got {self.kappa}."
            )
        _require_gt("search.dirichlet_thompson.kappa", self.kappa, 0.0)
        _require_ge("search.dirichlet_thompson.policy_samples", self.policy_samples, 0)
        _require_ge(
            "search.dirichlet_thompson.posterior_policy_samples",
            self.posterior_policy_samples,
            1,
        )
        _require_ge(
            "search.dirichlet_thompson.policy_sample_chunk_size",
            self.policy_sample_chunk_size,
            1,
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
class SelfplayConfig:
    batch_size: int = 1024
    max_num_steps: int = 256
    search: SearchConfig = field(default_factory=SearchConfig)
    action_commitment_type: ActionCommitmentType = ActionCommitmentType.posterior_argmax

    def __post_init__(self) -> None:
        _require_ge("selfplay.batch_size", self.batch_size, 1)


@dataclass
class TrainingLossConfig:
    policy_weight: float = 1.0
    value_dir_kl_weight: float = 0.0
    q_dir_kl_weight: float = 0.0
    value_outcome_weight: float = 0.0
    q_outcome_weight: float = 0.0
    q_loss_weight_mode: QLossWeightMode = QLossWeightMode.policy
    q_dir_kl_reduction: QDirKLReduction = QDirKLReduction.weighted
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
        if self.dirichlet_loss_mode == DirichletLossMode.mean:
            warnings.warn(
                "training.losses.dirichlet_loss_mode='mean' trains only "
                "Dirichlet posterior means; it does not train or penalize "
                "Dirichlet concentration (evidence mass).",
                UserWarning,
                stacklevel=2,
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
    optimizer: OptimizerType = OptimizerType.adam
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
    player_action_commitment_type: ActionCommitmentType = (
        ActionCommitmentType.posterior_argmax
    )
    baseline_action_commitment_type: ActionCommitmentType = (
        ActionCommitmentType.posterior_argmax
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
