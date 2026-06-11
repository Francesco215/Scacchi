from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

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
    dqaz = "dqaz"


class QLossWeightMode(StrEnum):
    policy = "policy"
    evidence_mass = "evidence_mass"


class QDirKLReduction(StrEnum):
    weighted = "weighted"
    masked_mean = "masked_mean"


class LossMaskMode(StrEnum):
    search = "search"
    value = "value"


class PolicyTargetMode(StrEnum):
    search = "search"
    winner_action = "winner_action"


class EvalBaseline(StrEnum):
    checkpoint = "checkpoint"
    pgx = "pgx"
    none = "none"


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
    rezero_kernel_init: RezeroKernelInit = RezeroKernelInit.variance_scaling



@dataclass
class SearchConstantsConfig:
    kappa_leaf: float = 1.0
    kappa_terminal: float = 8.0
    categorical_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        _require_gt("search.constants.kappa_leaf", self.kappa_leaf, 0.0)
        _require_gt("search.constants.kappa_terminal", self.kappa_terminal, 0.0)
        _require_range(
            "search.constants.categorical_epsilon",
            self.categorical_epsilon,
            lower=0.0,
            upper=0.5,
        )


@dataclass
class PolicySearchConfig:
    temperature: float = 1.0

    def __post_init__(self) -> None:
        _require_gt("search.policy.temperature", self.temperature, 0.0)


@dataclass
class GumbelSearchConfig:
    num_simulations: int = 32

    # Used when Gumbel search runs against a Dirichlet network and produces
    # posterior targets. Ignored by scalar policy/value models.
    policy_sample_chunk_size: int | None = 32
    constants: SearchConstantsConfig = field(default_factory=SearchConstantsConfig)
    gumbel_scale: float = 1.0
    completed_q_value_scale: float = 0.1
    completed_q_rescale_values: bool = True

    def __post_init__(self) -> None:
        _require_ge("search.gumbel.num_simulations", self.num_simulations, 1)
        _require_ge(
            "search.gumbel.policy_sample_chunk_size",
            self.policy_sample_chunk_size,
            1,
        )
        _require_gt("search.gumbel.gumbel_scale", self.gumbel_scale, 0.0)
        _require_gt(
            "search.gumbel.completed_q_value_scale",
            self.completed_q_value_scale,
            0.0,
        )


@dataclass
class DirichletThompsonSearchConfig:
    num_simulations: int = 32
    num_blocks: int = 1
    policy_samples: int = 32
    policy_sample_chunk_size: int | None = 32
    constants: SearchConstantsConfig = field(default_factory=SearchConstantsConfig)

    def __post_init__(self) -> None:
        _require_ge(
            "search.dirichlet_thompson.num_simulations",
            self.num_simulations,
            0,
        )
        _require_ge("search.dirichlet_thompson.num_blocks", self.num_blocks, 1)
        _require_ge("search.dirichlet_thompson.policy_samples", self.policy_samples, 0)
        _require_ge(
            "search.dirichlet_thompson.policy_sample_chunk_size",
            self.policy_sample_chunk_size,
            1,
        )


@dataclass
class DQAZSearchConfig:
    num_simulations: int = 32
    policy_samples: int = 32
    inflight_limit: int = 1
    state_posterior_kappa_n: float = 9.0
    eval_batch_size: int | None = None
    pad_to_eval_batch: bool = False
    jax_backup: bool = True
    debug: bool = False
    epsilon_terminal: float = 1e-3
    constants: SearchConstantsConfig = field(default_factory=SearchConstantsConfig)

    def __post_init__(self) -> None:
        _require_ge("search.dqaz.num_simulations", self.num_simulations, 1)
        _require_ge("search.dqaz.policy_samples", self.policy_samples, 1)
        _require_ge("search.dqaz.inflight_limit", self.inflight_limit, 1)
        _require_gt(
            "search.dqaz.state_posterior_kappa_n",
            self.state_posterior_kappa_n,
            0.0,
        )
        _require_ge("search.dqaz.eval_batch_size", self.eval_batch_size, 1)
        _require_range(
            "search.dqaz.epsilon_terminal",
            self.epsilon_terminal,
            lower=0.0,
            upper=0.5,
        )


@dataclass
class SearchConfig:
    kind: SearchKind = SearchKind.gumbel
    policy: PolicySearchConfig = field(default_factory=PolicySearchConfig)
    gumbel: GumbelSearchConfig = field(default_factory=GumbelSearchConfig)
    dirichlet_thompson: DirichletThompsonSearchConfig = field(
        default_factory=DirichletThompsonSearchConfig
    )
    dqaz: DQAZSearchConfig = field(default_factory=DQAZSearchConfig)

    def active(self) -> (
        PolicySearchConfig
        | GumbelSearchConfig
        | DirichletThompsonSearchConfig
        | DQAZSearchConfig
    ):
        return cast(
            PolicySearchConfig
            | GumbelSearchConfig
            | DirichletThompsonSearchConfig
            | DQAZSearchConfig,
            getattr(self, str(self.kind)),
        )

    def active_constants(self) -> SearchConstantsConfig:
        active = self.active()
        return getattr(active, "constants", SearchConstantsConfig())


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
    loss_mask_mode: LossMaskMode = LossMaskMode.search
    terminal_edge_targets: bool = False
    terminal_parent_targets: bool = False
    policy_target_mode: PolicyTargetMode = PolicyTargetMode.search

    def __post_init__(self) -> None:
        for name, value in (
            ("training.losses.policy_weight", self.policy_weight),
            ("training.losses.value_dir_kl_weight", self.value_dir_kl_weight),
            ("training.losses.q_dir_kl_weight", self.q_dir_kl_weight),
            ("training.losses.value_outcome_weight", self.value_outcome_weight),
            ("training.losses.q_outcome_weight", self.q_outcome_weight),
        ):
            _require_gt(name, value, -1.0)

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
        default_search = SearchConfig()
        if self.search == default_search:
            self.search = self.selfplay.search
        else:
            if self.selfplay.search == default_search:
                self.selfplay.search = self.search
            if self.eval.player_search == default_search:
                self.eval.player_search = self.search
            if self.eval.baseline_search == default_search:
                self.eval.baseline_search = self.search

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
            if search.kind in {SearchKind.dirichlet_thompson, SearchKind.dqaz}:
                assert self.model.network in dirichlet_networks, (
                    f"{name}.kind={search.kind!r} requires "
                    "model.network='aznet_dirichlet' or "
                    "model.network='boardlaw_dirichlet'."
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
        selfplay = OmegaConf.create({})
        aliased["selfplay"] = selfplay
    eval_cfg = aliased.get("eval")
    if eval_cfg is None:
        eval_cfg = OmegaConf.create({})
        aliased["eval"] = eval_cfg

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
