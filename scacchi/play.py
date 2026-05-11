from typing import NamedTuple

import chex
from flax import nnx
import jax
import mctx
import pgx
from pgx.experimental import auto_reset

from .dirichlet_q_search import (
    dirichlet_q_policy,
    make_dirichlet_recurrent_fn,
    make_root_embedding,
    posterior_targets,
    utility,
    wdl_mean,
)
from .network import AZNet


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    policy_target: chex.Array
    beta_V_target: jax.Array  # [B, 3] posterior Dirichlet target for the value head
    value_target_mask: jax.Array  # [B] True when search produced value evidence
    beta_Q_target: jax.Array  # [B, A, 3] posterior Dirichlet target for the Q head
    q_target_mask: jax.Array  # [B, A] True for actions with search evidence


def make_selfplay(env, config):
    """Build a self-play rollout function for the configured environment."""

    def selfplay(model: AZNet, rng_key: jax.Array) -> SelfplayOutput:
        """Generate one batched self-play trajectory with the current model."""
        recurrent_fn = make_dirichlet_recurrent_fn(
            env,
            lambda obs: model(obs, train=False),
            config.c_terminal,
            config.c_leaf,
        )

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step_fn(
            env_state: pgx.State,
            key: jax.Array,
        ) -> tuple[pgx.State, SelfplayOutput]:
            """Run one environment/search step and emit its training sample."""
            search_key, reset_key = jax.random.split(key, 2)
            observation = env_state.observation
            logits, alpha_V, alpha_Q = model(observation, train=False)
            alpha_V_mean = wdl_mean(alpha_V)
            value = utility(alpha_V_mean)
            root = mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=make_root_embedding(
                    env_state,
                    alpha_V_mean,
                    alpha_Q,
                    config.c_leaf,
                ),
            )

            invalid_actions = ~env_state.legal_action_mask
            search = dirichlet_q_policy(
                params=(),
                rng_key=search_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                alpha_Q_prior=alpha_Q,
                invalid_actions=invalid_actions,
                policy_mc_samples=config.policy_mc_samples,
                interior_selector=getattr(config, "train_interior_selector", "policy_prior"),
                num_search_blocks=getattr(config, "num_search_blocks", 1),
                action_selection_mode="sample",
            )
            policy_target = search.action_weights
            beta_V_target, value_target_mask, beta_Q_target, q_target_mask = posterior_targets(
                alpha_V,
                alpha_Q,
                search.q_evidence_sum,
                policy_target,
            )

            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            env_state = jax.vmap(auto_reset(env.step, env.init))(
                env_state,
                search.action,
                reset_keys,
            )
            return env_state, SelfplayOutput(
                obs=observation,
                policy_target=policy_target,
                beta_V_target=beta_V_target,
                value_target_mask=value_target_mask,
                beta_Q_target=beta_Q_target,
                q_target_mask=q_target_mask,
            )

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        step_keys = jax.random.split(rng_key, config.max_num_steps)
        _, data = step_fn(env_state, step_keys)
        return data

    return selfplay
