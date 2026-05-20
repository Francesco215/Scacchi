from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset

from .dirichlet_q_search import (
    NO_PARENT,
    NodeEmbedding,
    flip_outcome,
    outcome_mean,
    outcome_utility,
    dirichlet_q_policy,
    posterior_best_action,
    posterior_best_policy_target,
    posterior_sample_action,
    posterior_targets,
    q_evidence_sum_from_tree,
    root_action_value_priors_from_tree,
    terminal_outcome_from_reward,
)
from .network import policy_value_from_output


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    action_weights: chex.Array
    played_action: jax.Array
    legal_action_mask: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_evidence_mass: jax.Array
    discount: jax.Array


def make_recurrent_fn(env, predict_fn):
    def recurrent_fn(
        _,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        env_state: pgx.State,
    ):
        del rng_key

        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        logits, value = policy_value_from_output(predict_fn(env_state.observation))
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            env_state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            env_state,
        )

    return recurrent_fn


def make_dirichlet_recurrent_fn(env, predict_fn, config):
    def recurrent_fn(
        _,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        embedding: NodeEmbedding,
    ):
        del rng_key

        current_player = embedding.state.current_player
        env_state = jax.vmap(env.step)(embedding.state, action)
        logits, alpha_v, alpha_q = predict_fn(env_state.observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            env_state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        nonterminal_outcome = outcome_mean(alpha_v)
        terminal_parent_outcome = terminal_outcome_from_reward(
            reward,
            alpha_v.shape[-1],
        )
        terminal_child_outcome = flip_outcome(terminal_parent_outcome)
        outcome_dist = jnp.where(
            env_state.terminated[..., None],
            terminal_child_outcome,
            nonterminal_outcome,
        )
        evidence_weight = jnp.where(
            env_state.terminated,
            jnp.asarray(config.c_terminal, dtype=outcome_dist.dtype),
            jnp.asarray(config.c_leaf, dtype=outcome_dist.dtype),
        )
        root_action = jnp.where(
            embedding.root_action == NO_PARENT,
            action,
            embedding.root_action,
        )
        depth_parity = 1 - embedding.depth_parity

        value = outcome_utility(outcome_dist)
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        next_embedding = NodeEmbedding(
            state=env_state,
            outcome_dist=outcome_dist,
            alpha_V_prior=alpha_v,
            evidence_weight=evidence_weight,
            root_action=root_action,
            depth_parity=depth_parity,
            alpha_Q_prior=alpha_q,
        )
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            next_embedding,
        )

    return recurrent_fn


def _empty_posterior_targets(
    policy_target: jax.Array,
    num_outcomes: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size, num_actions = policy_target.shape
    beta_q = jnp.zeros(
        (batch_size, num_actions, num_outcomes),
        dtype=policy_target.dtype,
    )
    beta_v = jnp.zeros((batch_size, num_outcomes), dtype=policy_target.dtype)
    q_evidence_mass = jnp.zeros((batch_size, num_actions), dtype=policy_target.dtype)
    return beta_q, beta_v, q_evidence_mass


def make_selfplay(env, config):
    @nnx.jit
    def selfplay(model: nnx.Module, rng_key: jax.Array) -> SelfplayOutput:
        predict_fn = lambda obs: model(obs, train=False)
        recurrent_fn = make_recurrent_fn(env, predict_fn)
        dirichlet_recurrent_fn = make_dirichlet_recurrent_fn(env, predict_fn, config)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step_fn(
            env_state: pgx.State,
            key: jax.Array,
        ) -> tuple[pgx.State, SelfplayOutput]:
            search_key, posterior_key, action_key, reset_key = jax.random.split(key, 4)
            observation = env_state.observation
            legal_action_mask = env_state.legal_action_mask
            model_output = predict_fn(observation)

            if len(model_output) == 2:
                logits, value = policy_value_from_output(model_output)
                root = mctx.RootFnOutput(
                    prior_logits=logits,
                    value=value,
                    embedding=env_state,
                )
                policy_output = mctx.gumbel_muzero_policy(
                    params=(),
                    rng_key=search_key,
                    root=root,
                    recurrent_fn=recurrent_fn,
                    num_simulations=config.num_simulations,
                    invalid_actions=~env_state.legal_action_mask,
                    qtransform=mctx.qtransform_completed_by_mix_value,
                    gumbel_scale=1.0,
                )
                policy_target = policy_output.action_weights
                played_action = policy_output.action
                num_outcomes = config.num_outcomes
                if num_outcomes is None:
                    num_outcomes = 2 if config.env_id == "hex" else 3
                beta_Q_target, beta_V_target, q_evidence_mass = (
                    _empty_posterior_targets(policy_target, num_outcomes)
                )
            else:
                logits, alpha_v, alpha_q = model_output
                root_outcome = outcome_mean(alpha_v)
                value = outcome_utility(root_outcome)
                root_embedding = NodeEmbedding(
                    state=env_state,
                    outcome_dist=root_outcome,
                    alpha_V_prior=alpha_v,
                    evidence_weight=jnp.zeros_like(value),
                    root_action=jnp.full_like(env_state.current_player, NO_PARENT),
                    depth_parity=jnp.zeros_like(env_state.current_player),
                    alpha_Q_prior=alpha_q,
                )
                root = mctx.RootFnOutput(
                    prior_logits=logits,
                    value=value,
                    embedding=root_embedding,
                )
                action_value_prior = alpha_q
                if config.search_policy == "dirichlet_thompson":
                    policy_output = dirichlet_q_policy(
                        params=(),
                        rng_key=search_key,
                        root=root,
                        recurrent_fn=dirichlet_recurrent_fn,
                        action_value_prior=action_value_prior,
                        num_simulations=config.num_simulations,
                        invalid_actions=~env_state.legal_action_mask,
                        num_search_blocks=getattr(config, "num_search_blocks", 1),
                    )
                    q_evidence_sum = policy_output.q_evidence_sum
                    action_alpha_post = policy_output.alpha_search
                    action_value_target_prior = action_alpha_post - q_evidence_sum
                else:
                    policy_output = mctx.gumbel_muzero_policy(
                        params=(),
                        rng_key=search_key,
                        root=root,
                        recurrent_fn=dirichlet_recurrent_fn,
                        num_simulations=config.num_simulations,
                        invalid_actions=~env_state.legal_action_mask,
                        qtransform=mctx.qtransform_completed_by_mix_value,
                        gumbel_scale=1.0,
                    )
                    q_evidence_sum = q_evidence_sum_from_tree(policy_output.search_tree)
                    action_value_target_prior = root_action_value_priors_from_tree(
                        policy_output.search_tree,
                        action_value_prior,
                    )
                    action_alpha_post = action_value_target_prior + q_evidence_sum
                policy_target = posterior_best_policy_target(
                    posterior_key,
                    action_alpha_post,
                    legal_action_mask,
                    config.policy_mc_samples,
                )
                beta_Q_target, beta_V_target = (
                    posterior_targets(
                        alpha_v,
                        action_value_target_prior,
                        q_evidence_sum,
                        policy_target,
                    )
                )
                q_evidence_mass = jnp.sum(q_evidence_sum, axis=-1)
                if config.selfplay_action_source in ("posterior_best", "posterior_argmax"):
                    posterior_action = posterior_best_action(
                        policy_target,
                        legal_action_mask,
                    )
                    played_action = posterior_action
                elif config.selfplay_action_source == "posterior_sample":
                    posterior_action = posterior_sample_action(
                        action_key,
                        policy_target,
                        legal_action_mask,
                    )
                    played_action = posterior_action
                else:
                    played_action = policy_output.action

            actor = env_state.current_player
            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            env_state = jax.vmap(auto_reset(env.step, env.init))(
                env_state,
                played_action,
                reset_keys,
            )
            discount = -jnp.ones_like(value)
            discount = jnp.where(env_state.terminated, 0.0, discount)
            return env_state, SelfplayOutput(
                obs=observation,
                action_weights=policy_target,
                played_action=played_action,
                legal_action_mask=legal_action_mask,
                beta_Q_target=beta_Q_target,
                beta_V_target=beta_V_target,
                q_evidence_mass=q_evidence_mass,
                reward=env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor],
                terminated=env_state.terminated,
                discount=discount,
            )

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        step_keys = jax.random.split(rng_key, config.max_num_steps)
        _, data = step_fn(env_state, step_keys)
        return data

    return selfplay
