from dataclasses import replace

from flax import nnx
import jax

from .distributed import (
    DISABLED_BATCH_PARALLEL,
    BatchParallel,
    assert_batch_axis_sharded,
)
from .play import play_eval
from .play_search import (
    make_search_player,
)
from .types import EvalBaseline, SearchConfig, SearchKind


def baseline_search_config(config) -> SearchConfig:
    search_config = config.eval.baseline_search
    if config.eval.baseline != EvalBaseline.pgx:
        return search_config
    if search_config.kind == SearchKind.gumbel:
        return search_config

    # PGX baselines expose scalar policy/value heads, not Dirichlet heads.
    num_simulations = max(1, int(search_config.active().num_simulations))
    return replace(
        search_config,
        kind=SearchKind.gumbel,
        gumbel=replace(search_config.gumbel, num_simulations=num_simulations),
    )


def make_mcts_evaluate(
    env,
    config,
    baseline_model,
    parallel: BatchParallel | None = None,
):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    eval_batch_size = int(config.eval.batch_size)
    player_search_config = config.eval.player_search
    player_action_commitment_type = config.eval.player_action_commitment_type
    opponent_search_config = baseline_search_config(config)
    opponent_action_commitment_type = config.eval.baseline_action_commitment_type

    def search_player(model, search_config, action_commitment_type):
        return make_search_player(
            env,
            model,
            search_config,
            action_commitment_type,
            q_loss_weight_mode=str(config.training.losses.q_loss_weight_mode),
        )

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: nnx.Module):
        """MCTS evaluation: model search vs pretrained opponent."""
        metrics = play_eval(
            env,
            search_player(model, player_search_config, player_action_commitment_type),
            search_player(
                baseline_model,
                opponent_search_config,
                opponent_action_commitment_type,
            ),
            rng_key,
            batch_size=eval_batch_size,
            parallel=parallel,
        )
        return assert_batch_axis_sharded(
            metrics.returns,
            parallel,
            batch_axis=0,
            label="eval returns",
        )

    return evaluate
