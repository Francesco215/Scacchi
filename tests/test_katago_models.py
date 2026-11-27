"""Tests for 2026 KataGo binned architectures (ConvNet & QK-Norm 2D-RoPE Transformer)."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from scacchi import binned
from scacchi.envs import make_env
from scacchi.loss import make_compute_input_for_lossfn, train, train_binned_fast
from scacchi.network import (
    AZBinnedNet,
    AZInklingBinnedNet,
    AZKataConvBinnedNet,
    AZKataTransformerBinnedNet,
    build_model,
)
from scacchi.pipeline import _muon_weight_dimension_numbers, build_optimizer
from scacchi.play import _selfplay_step_frame
from scacchi.play_search import make_search_player
from scacchi.types import (
    ActionCommitmentConfig,
    ActionCommitmentType,
    BinConfig,
    BinnedThompsonSearchConfig,
    Config,
    EnvConfig,
    EvalConfig,
    ModelConfig,
    Network,
    SearchConfig,
    SearchKind,
    SelfplayConfig,
    TrainingConfig,
    TrainingLossConfig,
)


def _param_count(model: nnx.Module) -> int:
    params = nnx.state(model, nnx.Param)
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _make_hex_obs_pair(size: int = 8, batch: int = 8):
    """Build a real Black-to-move Hex board and its exact transposed White-to-move twin."""
    env = make_env("hex", size)
    rng = jax.random.PRNGKey(42)
    st = jax.vmap(env.init)(jax.random.split(rng, batch))
    for ply in range(6):
        logits = jnp.where(st.legal_action_mask, 0.0, -1e9)
        act = jax.random.categorical(jax.random.PRNGKey(10 + ply), logits).astype(jnp.int32)
        st = jax.vmap(env.step)(st, act)
    obs_black = st.observation.astype(jnp.float32)  # step 6 -> color == 0
    assert bool(jnp.all(obs_black[:, 0, 0, 2] == 0.0))
    obs_white = jnp.concatenate(
        [
            jnp.swapaxes(obs_black[..., :2], 1, 2),
            jnp.ones_like(obs_black[..., 2:3]),
            jnp.zeros_like(obs_black[..., 3:4]),
        ],
        axis=-1,
    )
    return obs_black, obs_white


def test_kata_models_param_parity_and_d2_color_equivariance():
    env = make_env("hex", 8)
    bins = BinConfig(count=9)
    base = AZBinnedNet(
        env.num_actions, env.observation_shape, num_channels=128, num_blocks=6, dtype=jnp.bfloat16, bins=bins, rngs=nnx.Rngs(0)
    )
    kconv = AZKataConvBinnedNet(
        env.num_actions, env.observation_shape, num_channels=128, num_blocks=6, dtype=jnp.bfloat16, bins=bins, rngs=nnx.Rngs(1)
    )
    ktf_nbt = AZKataTransformerBinnedNet(
        env.num_actions,
        env.observation_shape,
        num_channels=256,
        num_blocks=4,
        mid_channels=128,
        ffn_channels=320,
        num_heads=4,
        inner_blocks=2,
        qk_norm=True,
        dtype=jnp.bfloat16,
        bins=bins,
        rngs=nnx.Rngs(2),
    )
    ktf_flat = AZKataTransformerBinnedNet(
        env.num_actions,
        env.observation_shape,
        num_channels=192,
        num_blocks=4,
        mid_channels=192,
        ffn_channels=512,
        num_heads=6,
        inner_blocks=1,
        qk_norm=True,
        infer_chunk_size=4,
        dtype=jnp.bfloat16,
        bins=bins,
        rngs=nnx.Rngs(3),
    )
    inkling = AZInklingBinnedNet(
        env.num_actions,
        env.observation_shape,
        num_channels=192,
        num_blocks=6,
        layers_per_block=2,
        passes=1,
        ffn_channels=256,
        num_heads=6,
        kv_heads=2,
        infer_chunk_size=4,
        dtype=jnp.bfloat16,
        bins=bins,
        rngs=nnx.Rngs(4),
    )

    p_base = _param_count(base)
    for m in (kconv, ktf_nbt, ktf_flat, inkling):
        assert abs(_param_count(m) - p_base) / p_base < 0.15

    obs_b, obs_w = _make_hex_obs_pair(8, batch=8)
    perm = jnp.asarray([c * 8 + r for r in range(8) for c in range(8)] + [64], dtype=jnp.int32)

    for model in (kconv, ktf_nbt, ktf_flat, inkling):
        model.policy_q_head.policy_conv.kernel[...] = jax.random.normal(
            jax.random.PRNGKey(7), model.policy_q_head.policy_conv.kernel[...].shape, dtype=jnp.float32
        ) * 0.1
        model.policy_q_head.q_conv.kernel[...] = jax.random.normal(
            jax.random.PRNGKey(8), model.policy_q_head.q_conv.kernel[...].shape, dtype=jnp.float32
        ) * 0.1
        model.value_head.out.kernel[...] = jax.random.normal(
            jax.random.PRNGKey(9), model.value_head.out.kernel[...].shape, dtype=jnp.float32
        ) * 0.1

        out_b = model(obs_b, train=False)
        out_w = model(obs_w, train=False)
        assert isinstance(out_b, binned.BinnedOutput)
        assert np.allclose(np.asarray(out_b.value_probs), np.asarray(out_w.value_probs), atol=1e-5)
        assert np.allclose(np.asarray(out_b.logits[:, perm]), np.asarray(out_w.logits), atol=1e-4)
        assert np.allclose(np.asarray(out_b.action_probs[:, perm, :]), np.asarray(out_w.action_probs), atol=1e-5)

    # Verify L2-chunked inference matches unchunked inference under nnx.jit (within bf16 GEMM tiling tolerance).
    @nnx.jit
    def fwd(m, x):
        return m(x, train=False)

    out_chunked = fwd(ktf_flat, obs_b)
    ktf_flat.infer_chunk_size = 0
    out_full = fwd(ktf_flat, obs_b)
    assert np.allclose(np.asarray(out_chunked.logits), np.asarray(out_full.logits), atol=1.5e-2)
    assert np.allclose(np.asarray(out_chunked.value_probs), np.asarray(out_full.value_probs), atol=1e-2)
    assert np.allclose(np.asarray(out_chunked.action_probs), np.asarray(out_full.action_probs), atol=1e-2)


def test_muon_dimension_numbers_skip_rope_and_norm_scales():
    env = make_env("hex", 8)
    for model in (
        AZKataConvBinnedNet(
            env.num_actions, env.observation_shape, num_channels=128, num_blocks=2, bins=BinConfig(count=9), rngs=nnx.Rngs(0)
        ),
        AZKataTransformerBinnedNet(
            env.num_actions,
            env.observation_shape,
            num_channels=192,
            num_blocks=2,
            mid_channels=192,
            ffn_channels=512,
            num_heads=6,
            inner_blocks=1,
            qk_norm=True,
            bins=BinConfig(count=9),
            rngs=nnx.Rngs(0),
        ),
        AZInklingBinnedNet(
            env.num_actions,
            env.observation_shape,
            num_channels=192,
            layers_per_block=1,
            passes=2,
            ffn_channels=384,
            num_heads=6,
            kv_heads=2,
            gate_rank=32,
            bins=BinConfig(count=9),
            rngs=nnx.Rngs(0),
        ),
    ):
        params = nnx.state(model, nnx.Param)
        dims = _muon_weight_dimension_numbers(params)
        leaves = jax.tree_util.tree_leaves_with_path(
            dims,
            is_leaf=lambda x: x is None or isinstance(x, optax.contrib.MuonDimensionNumbers),
        )
        muon_names = []
        for path, val in leaves:
            name = ".".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)
            if val is not None:
                assert "rope_freqs" not in name
                assert "scale" not in name
                assert "bias" not in name
                assert "dw_weight" not in name
                assert "sink_k" not in name
                assert name.startswith("blocks.")
                muon_names.append(name)
        assert len(muon_names) > 0


def test_kata_models_train_step_odd_and_even_boards():
    # Cover odd board (Hex 5 -> XLA SDPA fallback) and even board (Hex 8 -> cuDNN FlashAttention),
    # flat + QK-Norm Transformer, dense Inkling-2 Mini trunk, and Muon weight decay.
    for board_size, net_kind, ch, layers, heads, inner in (
        (5, Network.aznet_kata_conv, 64, 2, 4, 2),
        (5, Network.aznet_kata_transformer, 128, 2, 4, 1),
        (8, Network.aznet_kata_transformer, 192, 2, 6, 1),
        (5, Network.aznet_inkling, 128, 1, 4, 1),
        (8, Network.aznet_inkling, 192, 1, 6, 1),
    ):
        env = make_env("hex", board_size)
        search_cfg = SearchConfig(
            kind=SearchKind.binned_thompson,
            binned_thompson=BinnedThompsonSearchConfig(num_simulations=4, search_kernel="xla"),
        )
        cfg = Config(
            env=EnvConfig(id="hex", board_size=board_size, num_outcomes=2),
            model=ModelConfig(
                network=net_kind,
                num_channels=ch,
                num_layers=layers,
                transformer_mid_channels=ch if inner == 1 else ch // 2,
                transformer_ffn_channels=ch * 2,
                transformer_heads=heads,
                transformer_inner_blocks=inner,
                transformer_qk_norm=True,
                transformer_infer_chunk_size=4,
                compute_dtype="bfloat16",
                bins=BinConfig(count=9),
            ),
            selfplay=SelfplayConfig(
                batch_size=8,
                max_num_steps=2,
                search=search_cfg,
                action_commitment=ActionCommitmentConfig(kind=ActionCommitmentType.posterior_sample),
            ),
            eval=EvalConfig(player_search=search_cfg),
            training=TrainingConfig(
                batch_size=8,
                grad_clip_norm=1.0,
                muon_weight_decay=0.02,
                losses=TrainingLossConfig(
                    policy_weight=1.0,
                    value_dir_kl_weight=1.0,
                    q_dir_kl_weight=1.0,
                ),
            ),
        )
        model = build_model(cfg, num_actions=env.num_actions, observation_shape=env.observation_shape, rngs=nnx.Rngs(0))
        opt = build_optimizer(model, cfg)
        player = make_search_player(
            env, model, cfg.selfplay.search, cfg.selfplay.action_commitment, q_supervision_config=cfg.training.losses.q_supervision
        )
        st = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 8))
        _, frame = _selfplay_step_frame(env, player, st, jax.random.PRNGKey(1), batch_size=8)
        data = jax.tree.map(lambda x: x[:, None] if isinstance(x, jax.Array) else x, frame)
        sample = jax.tree.map(lambda x: x[:, 0] if isinstance(x, jax.Array) else x, make_compute_input_for_lossfn(cfg)(data))
        losses_fast = train_binned_fast(model, opt, sample, cfg)
        assert all(bool(jnp.isfinite(x)) for x in losses_fast)
        metrics = train(model, opt, sample, cfg)
        assert bool(jnp.isfinite(metrics.policy_loss))
        assert bool(jnp.isfinite(metrics.value_binned_loss))
        assert bool(jnp.isfinite(metrics.q_binned_loss))
