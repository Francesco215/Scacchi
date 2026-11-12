import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from hydra import compose, initialize_config_module

from scacchi.curiosity import curious_action, next_bits_nll, next_bits_targets
from scacchi.envs import make_env
from scacchi.network import build_model
from scacchi.types import load_config


def test_next_bits_target_is_the_following_position_of_the_same_game():
    bits = jnp.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    valid = jnp.asarray([[True, True, True, False], [True, True, True, True]])
    terminated = jnp.asarray([[False, True, False, False], [False, False, False, True]])
    target, mask = next_bits_targets(bits, valid, terminated)
    # Lane 0: move 1 ends a game, row 3 is not trained on. Lane 1: the last row has no successor.
    np.testing.assert_array_equal(mask, [[True, False, False, False], [True, True, True, False]])
    np.testing.assert_array_equal(target, [[2.0, 0.0, 0.0, 0.0], [6.0, 7.0, 8.0, 0.0]])


def test_gaussian_nll_learns_the_mean_and_the_expected_error_and_stays_finite_at_small_sigma():
    target = jnp.asarray([0.5, 1.5, 4.0, 9.0])
    played = jnp.asarray([1, 1, 1, 0])
    mask = jnp.asarray([True, True, True, False])

    def loss(params):
        mu, log_var = params
        return next_bits_nll(jnp.stack((jnp.zeros(4), jnp.full(4, mu)), -1), jnp.stack((jnp.zeros(4), jnp.full(4, log_var)), -1), played, target, mask)["nll"]

    best_mu = jnp.mean(target[:3])
    best_log_var = jnp.log(jnp.mean((target[:3] - best_mu) ** 2))
    gradient = jax.grad(loss)((best_mu, best_log_var))
    assert max(abs(float(g)) for g in gradient) < 1e-5
    # A target of zero bits with a confident prediction: finite loss and gradient.
    zero = lambda params: next_bits_nll(jnp.full((1, 1), params[0]), jnp.full((1, 1), params[1]), jnp.zeros(1, jnp.int32), jnp.zeros(1), jnp.ones(1, bool))["nll"]
    value, gradient = jax.value_and_grad(zero)((jnp.asarray(0.0), jnp.asarray(-20.0)))
    assert np.isfinite(float(value)) and all(np.isfinite(float(g)) for g in gradient)
    stats = next_bits_nll(jnp.stack((jnp.ones(4), target), -1), jnp.zeros((4, 2)), played, target, mask)
    np.testing.assert_allclose(stats["correlation"], 1.0, atol=1e-5)
    np.testing.assert_allclose(stats["realised"], 2.0, atol=1e-6)


def test_curious_action_tilts_the_search_policy_and_never_plays_an_unsupported_move():
    policy = jnp.asarray([0.5, 0.3, 0.2, 0.0])
    legal = jnp.asarray([True, True, True, True])
    next_bits = jnp.asarray([0.0, 1.0, 2.0, 50.0])
    keys = jax.random.split(jax.random.PRNGKey(0), 40000)
    actions = jax.vmap(lambda key: curious_action(key, policy, legal, next_bits, 2.0))(keys)
    expected = np.asarray(policy) * np.exp(np.asarray(next_bits) / 2.0)
    expected[3] = 0.0
    np.testing.assert_allclose(np.bincount(np.asarray(actions), minlength=4) / 40000, expected / expected.sum(), atol=0.01)


def test_curiosity_head_predicts_per_move_and_leaves_the_features_alone():
    with initialize_config_module("scacchi.configs", version_base=None):
        config = load_config(compose("hex4", overrides=["+algorithm=binned", "model.curiosity_head=true", "model.num_channels=16", "model.num_layers=1"]))
    env = make_env(config.env.id, config.env.board_size)
    model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (3,) + tuple(env.observation_shape))
    output = model(obs, train=False)
    assert output.next_bits.shape == output.next_bits_log_variance.shape == (3, env.num_actions)

    def head_loss(model):
        output = model(obs, train=True)
        return jnp.sum(output.next_bits) + jnp.sum(output.next_bits_log_variance)

    grads = nnx.grad(head_loss)(model)
    flat = dict(nnx.to_flat_state(grads)) if hasattr(nnx, "to_flat_state") else None
    leaves = {"/".join(str(p) for p in path): np.abs(np.asarray(value[...] if hasattr(value, "__getitem__") else value)).sum() for path, value in (flat.items() if flat else [])}
    assert leaves, "no gradient leaves found"
    trunk = [total for name, total in leaves.items() if not name.startswith("curiosity")]
    head = [total for name, total in leaves.items() if name.startswith("curiosity")]
    assert max(trunk) == 0.0 and max(head) > 0.0


def test_curiosity_gradient_does_not_shrink_the_belief_update():
    import optax

    from scacchi.pipeline import build_optimizer

    with initialize_config_module("scacchi.configs", version_base=None):
        config = load_config(compose("hex4", overrides=["+algorithm=binned", "model.curiosity_head=true", "model.num_channels=16", "model.num_layers=1"]))
    env = make_env(config.env.id, config.env.board_size)
    model = build_model(config, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    clip = build_optimizer(model, config).tx  # the chain; its first transform is the clip
    grads = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e-3), params)
    huge = jax.tree_util.tree_map_with_path(
        lambda path, x: jnp.full_like(x, 1e6) if str(getattr(path[0], "key", "")).startswith("curiosity") else x, grads
    )
    updates_small, _ = clip.update(grads, clip.init(params), params)
    updates_huge, _ = clip.update(huge, clip.init(params), params)
    np.testing.assert_allclose(np.asarray(updates_small["policy_linear"]["bias"][...]), np.asarray(updates_huge["policy_linear"]["bias"][...]))
