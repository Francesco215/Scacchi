"""A pool of earlier positions that training revisits (experimental).

Every round self-play supplies ``n`` new boards. Each gradient step takes a
fixed fraction of its batch from those fresh boards and the rest from a pool
of ``size_multiple * n`` earlier boards. Afterwards the fresh boards join the
pool and ``n`` rows leave it.

Every row carries what was measured the last time it was in a batch:

- ``bits``: its share of the search-target loss minus the entropy of its
  targets, i.e. the bits of the search's conclusions the network has not
  absorbed;
- ``gain``: how many bits it lost between its last two visits (its ``bits``
  until it has been revisited).

A score is one of those (``log_bits`` for draws in proportion to a power of the
bits; ``useful`` also halves the bits every ``half_life_updates`` of target
age), or the row's youth (minus its age in rounds). Rows
leave with probability proportional to ``exp(-score / temperature)``, drawn
without replacement (Gumbel top-k), and are drawn into a batch uniformly or
with probability proportional to ``exp(score / sample_temperature)``.

With ``research_lanes > 0`` targets are recomputed instead of only recycled:
every self-play search call also searches that many pool positions with the
current network (same call, not stepped), and each such row gets the new
target, its bits against it, and age zero. Comparing the old and the new target
of a row measures, for free, how much an old target is still worth.

Binned beliefs on a single device only; the pool is not checkpointed.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from . import binned
from .dirichlet_mctx.native_targets import TARGET_BINNED
from .dirichlet_mctx.native_targets import TARGET_CATEGORICAL
from .loss import (
    Sample,
    TrainMetrics,
    _categorical_entropy_from_probs,
    _compute_dirichlet_losses,
    _dirichlet_mean_categorical_nll,
    _mask_or,
    _native_binned_loss,
    _native_target_fields,
    _outcome_index,
)

_LN2 = math.log(2.0)


class RowLoss(NamedTuple):
    """Per-row pieces of the search-target loss, in nats except ``bits``."""

    bits: jax.Array
    valid: jax.Array
    policy_ce: jax.Array
    policy_entropy: jax.Array
    policy_mask: jax.Array
    value_nll: jax.Array
    value_mask: jax.Array
    q_nll_sum: jax.Array
    q_pairs: jax.Array
    outcome_nll: jax.Array
    outcome_mask: jax.Array


class Pool(NamedTuple):
    samples: Sample
    bits: jax.Array
    gain: jax.Array
    valid: jax.Array
    born: jax.Array  # round the row's target was computed (insertion or last re-search)
    visits: jax.Array  # times drawn from the pool into a batch
    extra: Any = None  # what a re-search needs: the position and its transition


def _binned_target_entropy(target, kind, weight):
    """Entropy of a binned search target; a proven outcome has none."""
    safe = jnp.maximum(target, jnp.finfo(target.dtype).tiny)
    entropy = -jnp.sum(jnp.where(target > 0, target * jnp.log(safe), 0.0), axis=-1)
    return jnp.where(jnp.asarray(kind) == TARGET_BINNED, entropy * weight, 0.0)


def row_losses(output: binned.BinnedOutput, data: Sample, config) -> RowLoss:
    """Each row's share of the batch's search-target loss, and its excess bits.

    Mirrors the masks and reductions of ``_compute_dirichlet_losses``: the mean
    of ``bits * ln 2`` over the batch's valid rows is the weighted policy, V and
    Q search loss minus the entropy of the targets. Outcome terms are left out:
    a single game result has no target entropy to subtract.
    """
    losses = config.training.losses
    fields = _native_target_fields(data, data.beta_Q_target, data.beta_V_target)
    policy_mask = _mask_or(data.policy_loss_mask, data.value_mask)
    value_mask = _mask_or(data.value_loss_mask, data.value_mask)
    search_mask = _mask_or(data.search_loss_mask, policy_mask)

    policy_ce = optax.softmax_cross_entropy(output.logits, data.policy_tgt, where=data.policy_mask)
    policy_entropy = _categorical_entropy_from_probs(data.policy_tgt, data.policy_mask)

    value_nll = _native_binned_loss(
        data.beta_V_target, output.value_probs, fields["v_target_kind"], fields["v_target_outcome"], fields["v_target_weight"]
    )
    value_entropy = _binned_target_entropy(data.beta_V_target, fields["v_target_kind"], fields["v_target_weight"])
    value_mask = value_mask & jnp.isfinite(value_nll)

    q_nll = _native_binned_loss(
        data.beta_Q_target, output.action_probs, fields["q_target_kind"], fields["q_target_outcome"], fields["q_target_weight"]
    )
    q_entropy = _binned_target_entropy(data.beta_Q_target, fields["q_target_kind"], fields["q_target_weight"])
    q_row_mask = value_mask if losses.loss_mask_mode == "value" else search_mask
    pair_mask = data.q_supervised_pair_mask & data.policy_mask & q_row_mask[..., None] & jnp.isfinite(q_nll)
    q_nll_sum = jnp.sum(jnp.where(pair_mask, q_nll, 0.0), axis=-1)
    q_kl_sum = jnp.sum(jnp.where(pair_mask, q_nll - q_entropy, 0.0), axis=-1)
    q_pairs = jnp.sum(pair_mask, axis=-1).astype(q_nll.dtype)

    valid = policy_mask | value_mask | q_row_mask
    rows = jnp.maximum(jnp.sum(valid), 1)
    # The batch loss averages Q over pairs; a row's share is its pair sum over
    # the batch's mean pair count.
    pairs_per_row = jnp.maximum(jnp.sum(q_pairs) / rows, 1e-6)
    nats = (
        losses.policy_weight * jnp.where(policy_mask, policy_ce - policy_entropy, 0.0)
        + losses.value_dir_kl_weight * jnp.where(value_mask, value_nll - value_entropy, 0.0)
        + losses.q_dir_kl_weight * q_kl_sum / pairs_per_row
    )
    bits = jnp.where(valid, jnp.nan_to_num(nats) / _LN2, 0.0).astype(jnp.float32)
    win = binned.mean(output.value_probs, output.support)
    outcome_nll = _dirichlet_mean_categorical_nll(
        jnp.stack((1.0 - win, win), axis=-1), _outcome_index(data.value_tgt, config.env.resolved_num_outcomes())
    )
    outcome_mask = _mask_or(data.outcome_mask, data.value_mask) & (jnp.asarray(fields["v_target_kind"]) != TARGET_CATEGORICAL)
    return RowLoss(
        bits=bits,
        valid=valid,
        policy_ce=policy_ce,
        policy_entropy=policy_entropy,
        policy_mask=policy_mask,
        value_nll=jnp.nan_to_num(value_nll),
        value_mask=value_mask,
        q_nll_sum=q_nll_sum,
        q_pairs=q_pairs,
        outcome_nll=outcome_nll,
        outcome_mask=outcome_mask,
    )


def _mean(value, mask):
    mask = mask.astype(value.dtype)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1)


def _subset_metrics(rows: RowLoss, prefix: str) -> dict[str, jax.Array]:
    """The usual train/* losses, restricted to a subset of the batch."""
    return {
        f"{prefix}/policy_kl_hat": _mean(rows.policy_ce - rows.policy_entropy, rows.policy_mask),
        f"{prefix}/policy_target_entropy": _mean(rows.policy_entropy, rows.policy_mask),
        f"{prefix}/value_binned_loss": _mean(rows.value_nll, rows.value_mask),
        f"{prefix}/q_binned_loss": jnp.sum(rows.q_nll_sum) / jnp.maximum(jnp.sum(rows.q_pairs), 1),
        f"{prefix}/value_outcome_loss": _mean(rows.outcome_nll, rows.outcome_mask),
        f"{prefix}/bits": _mean(rows.bits, rows.valid),
    }


def train_with_row_losses(model: Any, optimizer: nnx.Optimizer, data: Sample, config) -> tuple[TrainMetrics, RowLoss]:
    """``loss.train`` that also returns each row's loss before the update."""

    def loss_fn(model: Any):
        output = model(data.obs, train=True)
        if not isinstance(output, binned.BinnedOutput):
            raise NotImplementedError("training.pool supports binned beliefs only.")
        total, metrics = _compute_dirichlet_losses(
            output.logits, output.value_probs, output.action_probs, data, config, bin_support=output.support
        )
        rows = jax.tree_util.tree_map(jax.lax.stop_gradient, row_losses(output, data, config))
        return total, (metrics, rows)

    (_, (metrics, rows)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return metrics, rows


SCORES = ("bits", "log_bits", "useful", "gain", "youth")
# Target age, in gradient updates, of the buckets of replay/bits_by_age.
AGE_BUCKETS = ((0, 200), (200, 800), (800, 3200), (3200, None))


def empty_pool(fresh: Sample, size: int, extra: Any = None) -> Pool:
    blank = lambda tree: jax.tree_util.tree_map(lambda x: jnp.zeros((size,) + x.shape[1:], x.dtype), tree)
    zeros = jnp.zeros((size,), jnp.float32)
    return Pool(
        blank(fresh), zeros, zeros, jnp.zeros((size,), jnp.bool_), jnp.zeros((size,), jnp.int32), jnp.zeros((size,), jnp.int32), blank(extra)
    )


def row_score(kind: str, bits: jax.Array, gain: jax.Array, born: jax.Array, now: jax.Array, half_life: float = 1.0) -> jax.Array:
    """What a row is worth keeping for; higher stays longer and is drawn more."""
    if kind == "bits":
        return bits
    if kind == "log_bits":  # exp(score / T) is then bits ** (1 / T): T=1 draws in proportion to bits
        return jnp.log(jnp.maximum(bits, 1e-6))
    if kind == "useful":  # log of bits discounted by the worth of a target of that age (half_life in rounds)
        return jnp.log(jnp.maximum(bits, 1e-6)) - _LN2 * (now - born).astype(jnp.float32) / half_life
    if kind == "gain":
        return gain
    if kind == "youth":
        return -(now - born).astype(jnp.float32)
    raise ValueError(f"unknown pool score {kind!r}; expected one of {SCORES}")


def sample_valid(key: jax.Array, valid: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Uniform indices over the valid rows; arbitrary if there are none."""
    cumulative = jnp.cumsum(valid.astype(jnp.int32))
    rank = jax.random.randint(key, shape, 0, jnp.maximum(cumulative[-1], 1))
    return jnp.minimum(jnp.searchsorted(cumulative, rank + 1), valid.shape[0] - 1)


def sample_by_score(key: jax.Array, score: jax.Array, valid: jax.Array, shape: tuple[int, ...], temperature: float) -> jax.Array:
    """Indices with probability proportional to exp(score/T) over the valid rows."""
    logits = jnp.where(valid, score / temperature, -jnp.inf)
    weights = jnp.where(valid, jnp.exp(logits - jnp.max(logits)), 0.0)
    cumulative = jnp.cumsum(weights)
    draws = jax.random.uniform(key, shape) * cumulative[-1]
    return jnp.minimum(jnp.searchsorted(cumulative, draws, side="right"), valid.shape[0] - 1)


def eviction_mask(key: jax.Array, score: jax.Array, valid: jax.Array, is_fresh: jax.Array, count: int, temperature: float) -> jax.Array:
    """Rows to drop: ``count`` draws without replacement from softmax(-score/T).

    ``temperature=0`` drops the lowest-scoring rows deterministically.
    Rows that carry no loss go first, fresh ones before empty pool slots.
    """
    if temperature == 0.0:
        keys = -score
    else:
        keys = -score / temperature + jax.random.gumbel(key, score.shape, score.dtype)
    keys = jnp.where(valid, keys, jnp.where(is_fresh, jnp.inf, 1e30))
    _, evicted = jax.lax.top_k(keys, count)
    return jnp.zeros(score.shape, jnp.bool_).at[evicted].set(True)


def insert_and_evict(
    pool: Pool,
    fresh: Sample,
    fresh_bits: jax.Array,
    fresh_valid: jax.Array,
    key: jax.Array,
    now: jax.Array,
    temperature: float,
    evict_by: str = "bits",
    half_life: float = 1.0,
    fresh_extra: Any = None,
):
    """Add the fresh rows to the pool, then evict as many rows as were added."""
    size, count = pool.bits.shape[0], fresh_bits.shape[0]
    fresh_born = jnp.full((count,), now, jnp.int32)
    # A row that has not been revisited may still give up all its bits.
    score = row_score(
        evict_by,
        jnp.concatenate((pool.bits, fresh_bits)),
        jnp.concatenate((pool.gain, fresh_bits)),
        jnp.concatenate((pool.born, fresh_born)),
        now,
        half_life,
    )
    evicted = eviction_mask(key, score, jnp.concatenate((pool.valid, fresh_valid)), jnp.arange(size + count) >= size, count, temperature)
    pool_evicted, fresh_kept = evicted[:size], ~evicted[size:]
    moved = jnp.sum(fresh_kept)  # == jnp.sum(pool_evicted)
    slots = jnp.argsort(~pool_evicted, stable=True)[:count]
    source = jnp.argsort(~fresh_kept, stable=True)[:count]
    slots = jnp.where(jnp.arange(count) < moved, slots, size)  # out of range: dropped

    def write(old, new):
        return old.at[slots].set(new[source], mode="drop")

    new_pool = Pool(
        jax.tree_util.tree_map(write, pool.samples, fresh),
        write(pool.bits, fresh_bits),
        write(pool.gain, fresh_bits),
        write(pool.valid, fresh_valid),
        write(pool.born, fresh_born),
        write(pool.visits, jnp.zeros((count,), jnp.int32)),
        jax.tree_util.tree_map(write, pool.extra, fresh_extra),
    )
    age = (now - new_pool.born).astype(jnp.float32)
    lost = pool_evicted & pool.valid
    stats = {
        "pool/valid_fraction": jnp.mean(new_pool.valid.astype(jnp.float32)),
        "pool/bits": _mean(new_pool.bits, new_pool.valid),
        "pool/gain": _mean(new_pool.gain, new_pool.valid & (new_pool.visits > 0)),
        "pool/visits": _mean(new_pool.visits.astype(jnp.float32), new_pool.valid),
        "pool/age_iters": _mean(age, new_pool.valid),
        "pool/max_age_iters": jnp.max(jnp.where(new_pool.valid, age, 0.0)),
        "pool/evicted_rows": jnp.sum(lost),
        "pool/evicted_bits": _mean(pool.bits, lost),
        "pool/evicted_visits": _mean(pool.visits.astype(jnp.float32), lost),
        "pool/evicted_age_iters": _mean((now - pool.born).astype(jnp.float32), lost),
        "pool/fresh_kept_fraction": _mean(fresh_kept.astype(jnp.float32), fresh_valid),
    }
    return new_pool, stats


def target_pulls(output: binned.BinnedOutput, data: Sample, config) -> jax.Array:
    """Per row, the direction ``p - t`` its targets pull the network's outputs.

    This is the gradient of the row's share of the search loss with respect to
    the logits of each head (policy, V and every supervised Q), flattened. The
    inner product of two rows' pulls on the same position is how much a step on
    one lowers the loss against the other, to first order.
    """
    losses = config.training.losses
    fields = _native_target_fields(data, data.beta_Q_target, data.beta_V_target)
    policy_mask = _mask_or(data.policy_loss_mask, data.value_mask)
    value_mask = _mask_or(data.value_loss_mask, data.value_mask)
    search_mask = _mask_or(data.search_loss_mask, policy_mask)
    q_row_mask = value_mask if losses.loss_mask_mode == "value" else search_mask

    def belief(target, kind, outcome, weight):
        kind = jnp.asarray(kind)
        exact = jax.nn.one_hot(jnp.asarray(outcome, jnp.int32) * (target.shape[-1] - 1), target.shape[-1], dtype=target.dtype)
        valid = (kind == TARGET_BINNED) | (kind == TARGET_CATEGORICAL)
        return jnp.where((kind == TARGET_CATEGORICAL)[..., None], exact, target), jnp.where(valid, weight, 0.0)

    logits = jnp.where(data.policy_mask, output.logits.astype(jnp.float32), -jnp.inf)
    policy = (jax.nn.softmax(logits, axis=-1) - data.policy_tgt) * policy_mask[..., None] * losses.policy_weight
    policy = jnp.where(data.policy_mask, policy, 0.0)
    value_target, value_weight = belief(data.beta_V_target, fields["v_target_kind"], fields["v_target_outcome"], fields["v_target_weight"])
    value = (output.value_probs - value_target) * (value_weight * value_mask)[..., None] * losses.value_dir_kl_weight
    q_target, q_weight = belief(data.beta_Q_target, fields["q_target_kind"], fields["q_target_outcome"], fields["q_target_weight"])
    pair_mask = data.q_supervised_pair_mask & data.policy_mask & q_row_mask[..., None]
    q_weight = q_weight * pair_mask
    # The same normalisation as row_losses: pairs per valid row of the batch.
    pairs = jnp.maximum(jnp.sum(pair_mask) / jnp.maximum(jnp.sum(policy_mask | value_mask | q_row_mask), 1), 1e-6)
    q = (output.action_probs - q_target) * q_weight[..., None] * (losses.q_dir_kl_weight / pairs)
    rows = policy.shape[0]
    return jnp.concatenate((policy.reshape(rows, -1), value.reshape(rows, -1), q.reshape(rows, -1)), axis=-1).astype(jnp.float32)


RESEARCH_BY = ("bits", "uniform", "age")


def select_for_research(
    key: jax.Array, pool: Pool, now: jax.Array, count: int, by: str, temperature: float = 0.0
) -> tuple[jax.Array, jax.Array]:
    """``count`` pool rows to search again, and which of them are real picks.

    When the pool holds fewer valid rows than ``count`` the remaining lanes
    repeat the first pick, so that every lane searches a legal position; those
    repeats are marked False and their results are discarded.
    """
    if by == "bits":
        # temperature > 0: draws without replacement with probability
        # proportional to exp(bits / temperature); 0 takes the rows with the most bits.
        score = pool.bits
        if temperature > 0.0:
            score = score / temperature + jax.random.gumbel(key, pool.bits.shape)
    elif by == "age":
        score = (now - pool.born).astype(jnp.float32)
    elif by == "uniform":
        score = jax.random.gumbel(key, pool.bits.shape)
    else:
        raise ValueError(f"unknown research_by {by!r}; expected one of {RESEARCH_BY}")
    _, indices = jax.lax.top_k(jnp.where(pool.valid, score, -jnp.inf), count)
    live = pool.valid[indices]
    return jnp.where(live, indices, indices[0]), live


def apply_research(
    model, pool: Pool, indices: jax.Array, live: jax.Array, posterior, now: jax.Array, compute_input_for_lossfn, config, updates_per_round: int
):
    """Give the re-searched rows their new targets and compare them with the old.

    ``posterior`` holds the search targets of ``pool.extra['state'][indices]``.
    The row keeps what came from its game (outcome, masks, the played move's
    terminal result) and takes everything search produced.
    """
    from .play import TrainingSamples

    stored = jax.tree_util.tree_map(lambda x: x[indices], pool.samples)
    extra = jax.tree_util.tree_map(lambda x: x[indices], pool.extra)
    timed = lambda tree: jax.tree_util.tree_map(lambda x: x[:, None], tree)
    source = TrainingSamples(
        obs=stored.obs[:, None],
        reward=extra["reward"][:, None],
        terminated=extra["terminated"][:, None],
        discount=jnp.where(extra["terminated"], 0.0, -1.0)[:, None],
        posterior=timed(posterior),
        played_action=stored.played_action[:, None],
        legal_action_mask=stored.policy_mask[:, None],
    )
    renewed = jax.tree_util.tree_map(lambda x: x[:, 0], compute_input_for_lossfn(source))
    kept = {"value_tgt": stored.value_tgt, "value_mask": stored.value_mask, "outcome_mask": stored.outcome_mask}
    mode = config.training.losses.loss_mask_mode
    if mode == "value":  # these masks follow the game, not the search
        kept.update(policy_loss_mask=stored.policy_loss_mask, value_loss_mask=stored.value_loss_mask)
    elif mode != "search":
        raise NotImplementedError("training.pool.research_lanes supports loss_mask_mode value or search.")
    renewed = renewed._replace(**kept)

    output = model(stored.obs, train=False)
    old_rows, new_rows = row_losses(output, stored, config), row_losses(output, renewed, config)
    old_pull, new_pull = target_pulls(output, stored, config), target_pulls(output, renewed, config)
    agree = jnp.sum(old_pull * new_pull, axis=-1)
    norm = jnp.sum(new_pull * new_pull, axis=-1)
    seen = live.astype(jnp.float32)
    age = (now - pool.born[indices]) * updates_per_round
    ratios = {
        # <p - t_old, p - t_new> / |p - t_new|^2: worth of the old target times
        # the signal share of a fresh one.
        "research/agreement": (jnp.sum(agree * seen), jnp.sum(norm * seen)),
        "research/wrong_way": (jnp.sum((agree < 0) * seen), jnp.sum(seen)),
        "research/bits_before": (jnp.sum(old_rows.bits * seen), jnp.sum(seen)),
        "research/bits_after": (jnp.sum(new_rows.bits * seen), jnp.sum(seen)),
        "research/age_updates": (jnp.sum(age * seen), jnp.sum(seen)),
        "research/rows": (jnp.sum(seen), jnp.ones(())),
    }
    for low, high in AGE_BUCKETS:
        inside = seen * (age >= low) * (1.0 if high is None else age < high)
        name = f"{low}_{'up' if high is None else high}"
        ratios[f"research/agreement_by_age/{name}"] = (jnp.sum(agree * inside), jnp.sum(norm * inside))
        ratios[f"research/wrong_way_by_age/{name}"] = (jnp.sum((agree < 0) * inside), jnp.sum(inside))

    def write(old, new):
        mask = live.reshape((-1,) + (1,) * (new.ndim - 1))
        return old.at[indices].set(jnp.where(mask, new.astype(old.dtype), old[indices]))

    pool = pool._replace(
        samples=jax.tree_util.tree_map(write, pool.samples, renewed),
        bits=write(pool.bits, new_rows.bits),
        gain=write(pool.gain, new_rows.bits),
        born=write(pool.born, jnp.full(indices.shape, now, jnp.int32)),
    )
    return pool, ratios


def make_pool_training_iteration(selfplay, compute_input_for_lossfn, with_data_stats, config, env=None):
    """One iteration: self-play, then rounds of mixed gradient steps and turnover.

    Self-play costs about the same for 64 games as for 512, so one iteration
    plays ``selfplay.batch_size`` games and feeds them to the pool in
    ``rounds_per_selfplay`` equal slices, one per round; ``n`` in the module
    docstring is the boards of one slice.

    With ``fill_games > 0`` the run starts with a fill phase: every iteration
    plays that many games, trains on them alone (no pool rows in the batch, no
    re-search) and stores them, until the pool has no empty slot. The pool
    then starts full of positions the network has just been trained on.
    """
    settings = config.training.pool
    batch_size = config.training.batch_size
    steps = int(config.selfplay.max_num_steps)
    for kind in (settings.evict_by, settings.sample_by):
        if kind not in SCORES + ("uniform",):
            raise ValueError(f"unknown pool score {kind!r}; expected one of {SCORES} or 'uniform'")
    if settings.research_lanes and (env is None or settings.research_by not in RESEARCH_BY):
        raise ValueError(f"training.pool.research_lanes needs the env and research_by in {RESEARCH_BY}")

    def shape_of(games: int, fresh_fraction: float, rounds: int) -> tuple[int, int, int]:
        rows = games * steps
        fresh_per_batch = round(batch_size * fresh_fraction)
        if rows % rounds or fresh_per_batch < 1 or (rows // rounds) % fresh_per_batch:
            raise ValueError(
                f"selfplay rows per iteration ({rows}) must split into {rounds} rounds, each a multiple of the "
                f"fresh rows per batch ({fresh_per_batch})."
            )
        return rows // rounds, fresh_per_batch, (rows // rounds) // fresh_per_batch

    main_games = int(config.selfplay.batch_size)
    main_rows_per_round, _, num_updates = shape_of(main_games, settings.fresh_fraction, settings.rounds_per_selfplay)
    pool_size = settings.size_multiple * main_rows_per_round
    # 'uniform' eviction is the infinite-temperature limit of any score.
    evict_by = "bits" if settings.evict_by == "uniform" else settings.evict_by
    temperature = 1e9 if settings.evict_by == "uniform" else float(settings.temperature)
    sample_temperature = float(settings.sample_temperature)
    half_life = settings.half_life_updates / num_updates  # in rounds
    keep_payload = bool(settings.research_lanes) or settings.start_from_pool
    if settings.start_from_pool and env is None:
        raise ValueError("training.pool.start_from_pool needs the env.")
    if settings.research_lanes * steps > pool_size:
        raise ValueError(f"research_lanes * steps ({settings.research_lanes * steps}) exceeds the pool ({pool_size} rows).")

    state: dict[str, Any] = {"pool": None, "round": 0, "extras": {}, "searches": 0, "frames": 0, "filling": bool(settings.fill_games)}

    def build_phase(games: int, fresh_fraction: float, rounds: int, lanes: int):
        """The iteration for one batch composition; phases share the pool."""
        rows_per_round, fresh_per_batch, phase_updates = shape_of(games, fresh_fraction, rounds)
        if phase_updates != num_updates:  # ages are counted in rounds of num_updates gradient steps
            raise ValueError(f"every phase needs {num_updates} updates per round; this one has {phase_updates}.")
        rows_per_iteration = games * steps
        pool_per_batch = batch_size - fresh_per_batch
        if env is not None:
            from .play import make_research_selfplay

            play = make_research_selfplay(env, config, games)
        elif games == main_games and not lanes:
            play = lambda model, key, extra_states, initial_states=None: (selfplay(model, key), None)
        else:
            raise ValueError("this pool configuration needs the env to build its self-play.")

        @jax.jit
        def shuffled_rows(data, key):
            samples = compute_input_for_lossfn(data)
            permutation = jax.random.permutation(key, rows_per_iteration)
            shuffle = lambda tree: jax.tree_util.tree_map(lambda x: x.reshape((rows_per_iteration,) + x.shape[2:])[permutation], tree)
            extra = {"state": data.state, "reward": data.reward, "terminated": data.terminated} if keep_payload else None
            return shuffle(samples), shuffle(extra)

        @jax.jit
        def research_states(pool: Pool, key, now):
            indices, live = select_for_research(key, pool, now, lanes * steps, settings.research_by, float(settings.research_temperature))
            states = jax.tree_util.tree_map(lambda x: x[indices].reshape((steps, lanes) + x.shape[1:]), pool.extra["state"])
            return indices, live, states

        @jax.jit
        def start_states(pool: Pool, key, now):
            """A pool position per lane, drawn by bits like the re-searched rows; the opening where the pool runs short."""
            pick_key, init_key = jax.random.split(key)
            indices, live = select_for_research(pick_key, pool, now, games, "bits", float(settings.research_temperature))
            opening = jax.vmap(env.init)(jax.random.split(init_key, games))
            choose = lambda stored, fresh: jnp.where(live.reshape((games,) + (1,) * (fresh.ndim - 1)), stored[indices], fresh)
            return jax.tree_util.tree_map(choose, pool.extra["state"], opening)

        @nnx.jit
        def research_update(model, pool: Pool, indices, live, posterior, now):
            posterior = jax.tree_util.tree_map(lambda x: x.reshape((steps * lanes,) + x.shape[2:]), posterior)
            return apply_research(model, pool, indices, live, posterior, now, compute_input_for_lossfn, config, num_updates)

        @nnx.jit
        def train_round(model, optimizer, rows: Sample, start, pool: Pool, now, key):
            fresh = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, rows_per_round), rows)
            fresh_batches = jax.tree_util.tree_map(lambda x: x.reshape((num_updates, fresh_per_batch) + x.shape[1:]), fresh)
            shape = (num_updates, pool_per_batch)
            if settings.sample_by == "uniform" or pool_per_batch == 0:
                pool_indices = sample_valid(key, pool.valid, shape)
            else:
                score = row_score(settings.sample_by, pool.bits, pool.gain, pool.born, now, half_life)
                pool_indices = sample_by_score(key, score, pool.valid, shape, sample_temperature)
            age_updates = (now - pool.born) * num_updates

            @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
            def scan_step(carry, fresh_batch, indices):
                model, optimizer, pool_bits, pool_gain, pool_visits = carry
                replay = jax.tree_util.tree_map(lambda x: x[indices], pool.samples)
                batch = jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b), axis=0), fresh_batch, replay)
                metrics, losses = train_with_row_losses(model, optimizer, batch, config)
                fresh_losses = jax.tree_util.tree_map(lambda x: x[:fresh_per_batch], losses)
                replay_losses = jax.tree_util.tree_map(lambda x: x[fresh_per_batch:], losses)
                means = {**_subset_metrics(fresh_losses, "fresh"), **_subset_metrics(replay_losses, "replay")}

                # (sum, count) pairs, reduced over the whole iteration.
                seen = replay_losses.valid.astype(jnp.float32)
                change = replay_losses.bits - pool_bits[indices]
                age = age_updates[indices]
                ratios = {
                    "replay/bits_change_since_last_visit": (jnp.sum(change * seen), jnp.sum(seen)),
                    "replay/bits_rising_fraction": (jnp.sum((change > 0) * seen), jnp.sum(seen)),
                    "replay/age_updates": (jnp.sum(age * seen), jnp.sum(seen)),
                    "replay/visits": (jnp.sum(pool_visits[indices] * seen), jnp.sum(seen)),
                }
                for low, high in AGE_BUCKETS:
                    inside = seen * (age >= low) * (1.0 if high is None else age < high)
                    name = f"{low}_{'up' if high is None else high}"
                    ratios[f"replay/bits_by_age/{name}"] = (jnp.sum(replay_losses.bits * inside), jnp.sum(inside))
                    ratios[f"replay/rising_by_age/{name}"] = (jnp.sum((change > 0) * inside), jnp.sum(inside))

                pool_gain = pool_gain.at[indices].set(-change)
                pool_bits = pool_bits.at[indices].set(replay_losses.bits)
                pool_visits = pool_visits.at[indices].add(1)
                return (model, optimizer, pool_bits, pool_gain, pool_visits), (metrics, means, ratios, fresh_losses.bits, fresh_losses.valid)

            (_, _, pool_bits, pool_gain, pool_visits), (metrics, means, ratios, fresh_bits, fresh_valid) = scan_step(
                (model, optimizer, pool.bits, pool.gain, pool.visits), fresh_batches, pool_indices
            )
            means = {name: jnp.mean(value) for name, value in means.items()}
            ratios = {name: (jnp.sum(total), jnp.sum(count)) for name, (total, count) in ratios.items()}
            return metrics, means, ratios, (pool_bits, pool_gain, pool_visits), fresh, fresh_bits.reshape(-1), fresh_valid.reshape(-1)

        def _turnover(pool, fresh, bits, valid, key, now, payload, start):
            fresh_extra = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, rows_per_round), payload)
            return insert_and_evict(pool, fresh, bits, valid, key, now, temperature, evict_by, half_life, fresh_extra)

        turnover = jax.jit(_turnover, donate_argnums=(0,))

        def iteration(model, optimizer, rng_key):
            selfplay_key, permutation_key, round_key, research_key, start_key = jax.random.split(rng_key, 5)
            research_ratios = {}
            starts = None
            if settings.start_from_pool and state["pool"] is not None and not state["filling"]:
                starts = start_states(state["pool"], start_key, jnp.asarray(state["round"], jnp.int32))
            if not lanes:
                data, _ = play(model, selfplay_key, None, starts)
            elif state["pool"] is None:  # nothing to search again yet: the extra lanes get opening positions
                opening = jax.vmap(env.init)(jax.random.split(research_key, lanes))
                filler = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (steps,) + x.shape), opening)
                data, _ = play(model, selfplay_key, filler)
            else:
                now = jnp.asarray(state["round"], jnp.int32)
                indices, live, states = research_states(state["pool"], research_key, now)
                data, posterior = play(model, selfplay_key, states, starts)
                state["pool"], research_ratios = research_update(model, state["pool"], indices, live, posterior, now)
            state["frames"] = rows_per_iteration
            state["searches"] += rows_per_iteration + lanes * steps
            rows, payload = shuffled_rows(data, permutation_key)
            if state["pool"] is None:
                state["pool"] = empty_pool(rows, pool_size, payload)
            metrics, means, ratios = [], [], [research_ratios]
            for index, key in enumerate(jax.random.split(round_key, rounds)):
                train_key, evict_key = jax.random.split(key)
                pool, now = state["pool"], jnp.asarray(state["round"], jnp.int32)
                start = jnp.asarray(index * rows_per_round, jnp.int32)
                round_metrics, round_means, round_ratios, (bits, gain, visits), fresh, fresh_bits, fresh_valid = train_round(
                    model, optimizer, rows, start, pool, now, train_key
                )
                state["pool"], turnover_stats = turnover(
                    pool._replace(bits=bits, gain=gain, visits=visits), fresh, fresh_bits, fresh_valid, evict_key, now, payload, start
                )
                state["round"] += 1
                metrics.append(round_metrics)
                means.append({**round_means, **turnover_stats})
                ratios.append(round_ratios)
            extras = {name: float(sum(m[name] for m in means)) / rounds for name in means[0]}
            for name in {name for r in ratios for name in r}:
                total = float(sum(r[name][0] for r in ratios if name in r))
                count = float(sum(r[name][1] for r in ratios if name in r))
                if count > 0:
                    extras[name] = total / count
            extras["pool/age_updates"] = extras["pool/age_iters"] * num_updates
            extras["pool/filling"] = float(state["filling"])
            extras["train/updates"] = state["round"] * num_updates
            extras["train/gradient_rows"] = state["round"] * num_updates * batch_size
            extras["train/searches"] = state["searches"]
            state["extras"] = extras
            metrics = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *metrics)
            return with_data_stats(metrics, data), float(turnover_stats["pool/valid_fraction"])

        return iteration

    main_iteration = build_phase(main_games, settings.fresh_fraction, settings.rounds_per_selfplay, int(settings.research_lanes))
    fill_iteration = None
    if settings.fill_games:
        fill_rounds, remainder = divmod(int(settings.fill_games) * steps, num_updates * batch_size)
        if remainder or not fill_rounds:
            raise ValueError(
                f"fill_games * steps ({int(settings.fill_games) * steps}) must be a multiple of {num_updates * batch_size} "
                "(one round of gradient steps on fresh rows only)."
            )
        fill_iteration = build_phase(int(settings.fill_games), 1.0, fill_rounds, 0)

    def training_iteration(model, optimizer, rng_key):
        if state["filling"]:
            metrics, valid_fraction = fill_iteration(model, optimizer, rng_key)
            state["filling"] = valid_fraction < 1.0
            return metrics
        return main_iteration(model, optimizer, rng_key)[0]

    training_iteration.extras = lambda: state["extras"]
    training_iteration.frames = lambda: state["frames"]
    return training_iteration
