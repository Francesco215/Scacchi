import math

import jax
import jax.numpy as jnp
import numpy as np

from scacchi import binned
from scacchi.dirichlet_mctx.native_targets import TARGET_BINNED, TARGET_CATEGORICAL
from scacchi.loss import Sample, _compute_dirichlet_losses
from scacchi.pool import (
    empty_pool,
    eviction_mask,
    insert_and_evict,
    row_losses,
    sample_by_score,
    sample_valid,
    select_for_research,
    target_pulls,
)
from hydra import compose, initialize_config_module

from scacchi.types import load_config


def _random_batch(key, rows=24, actions=5, bins=9):
    keys = jax.random.split(key, 12)
    probs = lambda k, shape: jax.nn.softmax(jax.random.normal(k, shape), axis=-1)
    value_mask = jax.random.bernoulli(keys[0], 0.7, (rows,))
    policy_mask = jax.random.bernoulli(keys[1], 0.8, (rows, actions)).at[:, 0].set(True)
    policy_tgt = jnp.where(policy_mask, probs(keys[2], (rows, actions)), 0.0)
    policy_tgt = policy_tgt / policy_tgt.sum(-1, keepdims=True)
    q_kind = jnp.where(jax.random.bernoulli(keys[3], 0.3, (rows, actions)), TARGET_CATEGORICAL, TARGET_BINNED)
    v_kind = jnp.where(jax.random.bernoulli(keys[4], 0.3, (rows,)), TARGET_CATEGORICAL, TARGET_BINNED)
    data = Sample(
        obs=jnp.zeros((rows, 1)),
        policy_tgt=policy_tgt,
        value_tgt=jnp.where(jax.random.bernoulli(keys[5], 0.5, (rows,)), 1.0, -1.0),
        played_action=jnp.zeros((rows,), jnp.int32),
        policy_mask=policy_mask,
        value_mask=value_mask,
        beta_Q_target=probs(keys[6], (rows, actions, bins)),
        beta_V_target=probs(keys[7], (rows, bins)),
        q_supervised_pair_mask=jax.random.bernoulli(keys[8], 0.6, (rows, actions)),
        q_pair_weight=jnp.ones((rows, actions)),
        q_target_kind=q_kind,
        q_target_weight=jnp.ones((rows, actions)),
        q_target_outcome=jax.random.bernoulli(keys[9], 0.5, (rows, actions)).astype(jnp.int32),
        q_target_distance=jnp.zeros((rows, actions), jnp.int32),
        v_target_kind=v_kind,
        v_target_weight=jnp.ones((rows,)),
        v_target_outcome=jax.random.bernoulli(keys[10], 0.5, (rows,)).astype(jnp.int32),
        v_target_distance=jnp.zeros((rows,), jnp.int32),
    )
    subkeys = jax.random.split(keys[11], 3)
    output = binned.BinnedOutput(
        logits=jax.random.normal(subkeys[0], (rows, actions)),
        value_probs=probs(subkeys[1], (rows, bins)),
        action_probs=probs(subkeys[2], (rows, actions, bins)),
        support=jnp.linspace(0.0, 1.0, bins),
    )
    return data, output


def test_row_bits_average_to_the_batch_search_loss_minus_target_entropy():
    with initialize_config_module("scacchi.configs", version_base=None):
        config = load_config(compose("hex6", overrides=["+algorithm=binned", "training.losses.value_dir_kl_weight=0.7", "training.losses.q_dir_kl_weight=1.3", "training.losses.value_outcome_weight=0", "training.losses.q_outcome_weight=0"]))
    data, output = _random_batch(jax.random.PRNGKey(0))
    total, metrics = _compute_dirichlet_losses(output.logits, output.value_probs, output.action_probs, data, config, bin_support=output.support)
    rows = row_losses(output, data, config)
    count = rows.valid.sum()
    np.testing.assert_array_equal(rows.valid, data.value_mask)
    # The NLL pieces reduce to the batch loss exactly.
    nll = (
        jnp.sum(jnp.where(rows.policy_mask, rows.policy_ce, 0.0)) / count
        + 0.7 * jnp.sum(jnp.where(rows.value_mask, rows.value_nll, 0.0)) / count
        + 1.3 * jnp.sum(rows.q_nll_sum) / jnp.sum(rows.q_pairs)
    )
    np.testing.assert_allclose(nll, total, rtol=1e-5)
    np.testing.assert_allclose(jnp.sum(rows.q_nll_sum) / jnp.sum(rows.q_pairs), metrics.q_binned_loss, rtol=1e-5)
    assert float(jnp.min(rows.bits)) >= -1e-5  # a KL
    assert float(jnp.max(jnp.where(rows.valid, 0.0, jnp.abs(rows.bits)))) == 0.0

    # Predicting the targets exactly leaves no bits.
    exact = lambda target, kind, outcome: jnp.where(
        (kind == TARGET_CATEGORICAL)[..., None], jax.nn.one_hot(outcome * (target.shape[-1] - 1), target.shape[-1]), target
    )
    perfect = output._replace(
        logits=jnp.where(data.policy_mask, jnp.log(jnp.maximum(data.policy_tgt, 1e-30)), -1e9),
        value_probs=exact(data.beta_V_target, data.v_target_kind, data.v_target_outcome),
        action_probs=exact(data.beta_Q_target, data.q_target_kind, data.q_target_outcome),
    )
    np.testing.assert_allclose(row_losses(perfect, data, config).bits, 0.0, atol=1e-4)


def test_single_eviction_follows_softmax_of_negative_bits():
    bits = jnp.asarray([0.0, 1.0, 2.0, 4.0])
    valid = jnp.ones(4, jnp.bool_)
    fresh = jnp.zeros(4, jnp.bool_)
    temperature = 1.5
    masks = jax.vmap(lambda key: eviction_mask(key, bits, valid, fresh, 1, temperature))(jax.random.split(jax.random.PRNGKey(1), 40000))
    np.testing.assert_allclose(masks.mean(0), jax.nn.softmax(-bits / temperature), atol=0.01)


def test_two_evictions_are_draws_without_replacement():
    bits = jnp.asarray([0.0, 0.5, 3.0])
    weights = np.exp(-np.asarray(bits))
    # P(row i survives) = sum over ordered pairs (a, b) of the other two rows.
    survive = []
    for i in range(3):
        a, b = [j for j in range(3) if j != i]
        total = weights.sum()
        survive.append(weights[a] / total * weights[b] / (total - weights[a]) + weights[b] / total * weights[a] / (total - weights[b]))
    masks = jax.vmap(lambda key: eviction_mask(key, bits, jnp.ones(3, jnp.bool_), jnp.zeros(3, jnp.bool_), 2, 1.0))(jax.random.split(jax.random.PRNGKey(2), 40000))
    np.testing.assert_allclose(1.0 - masks.mean(0), survive, atol=0.01)


def test_rows_without_loss_leave_first_and_the_pool_fills():
    size, count = 12, 4
    template = Sample(*(jnp.zeros((1,)) for _ in range(10)))._replace(obs=jnp.zeros((1, 2)))
    pool = empty_pool(template, size)
    for iteration in range(6):
        fresh = jax.tree_util.tree_map(lambda x: jnp.full((count,) + x.shape[1:], iteration + 1.0, x.dtype), template)
        valid = jnp.asarray([True, True, True, False])
        pool, stats = insert_and_evict(pool, fresh, jnp.full((count,), 5.0), valid, jax.random.PRNGKey(iteration), jnp.asarray(iteration), 1.0)
        assert int(pool.valid.sum()) == min(3 * (iteration + 1), size)
        np.testing.assert_array_equal(pool.samples.obs[:, 0] > 0, pool.valid)
        np.testing.assert_array_equal(pool.born[pool.valid] + 1, pool.samples.obs[pool.valid, 0].astype(jnp.int32))
    assert float(stats["pool/valid_fraction"]) == 1.0


def test_high_loss_rows_outlive_low_loss_rows():
    size, count = 64, 16
    template = Sample(*(jnp.zeros((1,)) for _ in range(10)))
    pool = empty_pool(template, size)
    key = jax.random.PRNGKey(3)
    for iteration in range(40):
        key, bits_key, evict_key = jax.random.split(key, 3)
        bits = jnp.where(jax.random.bernoulli(bits_key, 0.5, (count,)), 6.0, 0.0)
        fresh = jax.tree_util.tree_map(lambda x: jnp.zeros((count,) + x.shape[1:], x.dtype), template)
        pool, _ = insert_and_evict(pool, fresh, bits, jnp.ones(count, jnp.bool_), evict_key, jnp.asarray(iteration), 1.0)
    assert float(jnp.mean(pool.bits == 6.0)) > 0.85


def test_sample_valid_is_uniform_over_valid_rows():
    valid = jnp.asarray([False, True, False, True, True, False])
    indices = sample_valid(jax.random.PRNGKey(4), valid, (30000,))
    counts = np.bincount(np.asarray(indices), minlength=6) / 30000
    np.testing.assert_allclose(counts, [0, 1 / 3, 0, 1 / 3, 1 / 3, 0], atol=0.01)
    assert math.isfinite(float(sample_valid(jax.random.PRNGKey(5), jnp.zeros(6, jnp.bool_), (4,)).sum()))


def test_zero_temperature_evicts_the_lowest_bits():
    bits = jnp.asarray([3.0, 0.5, 2.0, 0.1, 1.0])
    mask = eviction_mask(jax.random.PRNGKey(0), bits, jnp.ones(5, jnp.bool_), jnp.zeros(5, jnp.bool_), 2, 0.0)
    np.testing.assert_array_equal(mask, [False, True, False, True, False])


def test_youth_eviction_at_zero_temperature_is_first_in_first_out():
    size, count = 12, 4
    template = Sample(*(jnp.zeros((1,)) for _ in range(10)))
    pool = empty_pool(template, size)
    for now in range(8):
        fresh = jax.tree_util.tree_map(lambda x: jnp.zeros((count,) + x.shape[1:], x.dtype), template)
        bits = jax.random.uniform(jax.random.PRNGKey(now), (count,)) * 5
        pool, stats = insert_and_evict(pool, fresh, bits, jnp.ones(count, jnp.bool_), jax.random.PRNGKey(now), jnp.asarray(now), 0.0, "youth")
    np.testing.assert_array_equal(np.sort(np.asarray(pool.born)), np.repeat([5, 6, 7], 4))
    assert float(stats["pool/evicted_age_iters"]) == 3.0


def test_gain_eviction_keeps_unvisited_rows_on_their_bits():
    template = Sample(*(jnp.zeros((1,)) for _ in range(10)))
    pool = empty_pool(template, 4)
    fresh = jax.tree_util.tree_map(lambda x: jnp.zeros((4,) + x.shape[1:], x.dtype), template)
    ones = jnp.ones(4, jnp.bool_)
    pool, _ = insert_and_evict(pool, fresh, jnp.asarray([1.0, 2.0, 3.0, 4.0]), ones, jax.random.PRNGKey(0), jnp.asarray(0), 0.0, "gain")
    np.testing.assert_array_equal(pool.gain, pool.bits)
    # After a visit the rows with 3 and 4 bits stopped improving; they leave before the others.
    order = np.argsort(np.asarray(pool.bits))
    gain = np.zeros(4, np.float32)
    gain[order] = [0.5, 0.6, 0.0, -0.1]
    pool = pool._replace(gain=jnp.asarray(gain), visits=jnp.ones(4, jnp.int32))
    pool, _ = insert_and_evict(pool, jax.tree_util.tree_map(lambda x: x[:2], fresh), jnp.asarray([0.2, 0.3]), ones[:2], jax.random.PRNGKey(1), jnp.asarray(1), 0.0, "gain")
    np.testing.assert_allclose(np.sort(np.asarray(pool.bits)), [0.2, 0.3, 1.0, 2.0])


def test_sample_by_score_follows_softmax_over_valid_rows():
    score = jnp.asarray([0.0, 1.0, 5.0, 2.0])
    valid = jnp.asarray([True, True, False, True])
    indices = sample_by_score(jax.random.PRNGKey(6), score, valid, (60000,), 0.7)
    counts = np.bincount(np.asarray(indices), minlength=4) / 60000
    expected = np.where(np.asarray(valid), np.exp(np.asarray(score) / 0.7), 0.0)
    np.testing.assert_allclose(counts, expected / expected.sum(), atol=0.01)


def test_log_bits_score_draws_in_proportion_to_bits():
    from scacchi.pool import row_score

    bits = jnp.asarray([0.5, 1.0, 2.5])
    score = row_score("log_bits", bits, bits, jnp.zeros(3, jnp.int32), jnp.asarray(0))
    indices = sample_by_score(jax.random.PRNGKey(7), score, jnp.ones(3, jnp.bool_), (60000,), 1.0)
    np.testing.assert_allclose(np.bincount(np.asarray(indices), minlength=3) / 60000, np.asarray(bits) / 4.0, atol=0.01)


def test_target_pulls_are_the_gradient_of_the_row_bits_with_respect_to_the_logits():
    with initialize_config_module("scacchi.configs", version_base=None):
        config = load_config(compose("hex6", overrides=["+algorithm=binned", "training.losses.value_dir_kl_weight=0.7", "training.losses.q_dir_kl_weight=1.3"]))
    data, output = _random_batch(jax.random.PRNGKey(8))
    logits = (output.logits, jnp.log(output.value_probs), jnp.log(output.action_probs))

    def total_nats(logits):
        policy, value, q = logits
        predicted = output._replace(logits=policy, value_probs=jax.nn.softmax(value, axis=-1), action_probs=jax.nn.softmax(q, axis=-1))
        return jnp.sum(row_losses(predicted, data, config).bits) * math.log(2.0)

    gradient = jax.grad(total_nats)(logits)
    rows = data.obs.shape[0]
    expected = jnp.concatenate([g.reshape(rows, -1) for g in gradient], axis=-1)
    np.testing.assert_allclose(target_pulls(output, data, config), expected, atol=2e-5)


def test_research_selection_takes_distinct_valid_rows_with_the_most_bits():
    template = Sample(*(jnp.zeros((1,)) for _ in range(10)))
    pool = empty_pool(template, 6)._replace(
        bits=jnp.asarray([0.2, 3.0, 1.0, 9.0, 2.0, 0.5]), valid=jnp.asarray([True, True, True, False, True, True])
    )
    indices, live = select_for_research(jax.random.PRNGKey(0), pool, jnp.asarray(0), 3, "bits")
    np.testing.assert_array_equal(np.sort(np.asarray(indices)), [1, 2, 4])
    assert bool(live.all())
    uniform, live = select_for_research(jax.random.PRNGKey(1), pool, jnp.asarray(0), 5, "uniform")
    np.testing.assert_array_equal(np.sort(np.asarray(uniform)), [0, 1, 2, 4, 5])
    # More lanes than valid rows: the surplus repeats a valid row and is marked dead.
    padded, live = select_for_research(jax.random.PRNGKey(2), pool, jnp.asarray(0), 6, "bits")
    assert int(live.sum()) == 5 and bool(pool.valid[padded].all())
    assert int(padded[~live][0]) == int(padded[0])


def test_turnover_carries_the_payload_of_each_row():
    template = Sample(*(jnp.zeros((1,)) for _ in range(10)))
    payload = {"state": jnp.zeros((1, 3)), "reward": jnp.zeros((1,))}
    pool = empty_pool(template, 8, payload)
    for now in range(4):
        fresh = jax.tree_util.tree_map(lambda x: jnp.full((4,) + x.shape[1:], now + 1.0, x.dtype), template)
        extra = {"state": jnp.full((4, 3), now + 1.0), "reward": jnp.full((4,), now + 1.0)}
        pool, _ = insert_and_evict(pool, fresh, jnp.ones(4), jnp.ones(4, jnp.bool_), jax.random.PRNGKey(now), jnp.asarray(now), 0.0, "youth", 1.0, extra)
    np.testing.assert_array_equal(pool.extra["reward"], pool.samples.obs)
    np.testing.assert_array_equal(pool.extra["state"][:, 0], pool.born + 1)


def test_research_selection_with_temperature_draws_by_softmax_of_bits():
    template = Sample(*(jnp.zeros((1,)) for _ in range(10)))
    bits = jnp.asarray([0.0, 1.0, 2.0, 4.0])
    pool = empty_pool(template, 4)._replace(bits=bits, valid=jnp.ones(4, jnp.bool_))
    picks = jax.vmap(lambda key: select_for_research(key, pool, jnp.asarray(0), 1, "bits", 1.5)[0][0])(jax.random.split(jax.random.PRNGKey(3), 40000))
    np.testing.assert_allclose(np.bincount(np.asarray(picks), minlength=4) / 40000, jax.nn.softmax(bits / 1.5), atol=0.01)
