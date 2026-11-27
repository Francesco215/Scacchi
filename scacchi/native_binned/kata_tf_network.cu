// XLA FFI entry points for the SMEM-resident INT8 + FP8-FlashAttention KataTransformer megakernel.

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "cache_key.cuh"
#include "kata_tf_board_int8.cuh"

namespace ffi = xla::ffi;

namespace {

using scacchi_kata_tf::KataTfParams;
using scacchi_kata_tf::launch_kata_tf_board_int8;
using scacchi_net::CacheKey;
using scacchi_net::CacheWrite;

template <typename T>
const T* cptr(const ffi::AnyBuffer& b) {
  return reinterpret_cast<const T*>(b.untyped_data());
}
template <typename T>
T* mptr(ffi::AnyBuffer& b) {
  return reinterpret_cast<T*>(b.untyped_data());
}

ffi::Error run_kata_tf_common(
    cudaStream_t stream,
    const ffi::AnyBuffer& obs,
    const int32_t* d_boards,
    const ffi::AnyBuffer& stem_w,
    const ffi::AnyBuffer& stem_b,
    const ffi::AnyBuffer& attn_norm_sb,
    const ffi::AnyBuffer& ffn_norm_sb,
    const ffi::AnyBuffer& qk_norm_sb,
    const ffi::AnyBuffer& rope_freqs,
    const ffi::AnyBuffer& w_qkv,
    const ffi::AnyBuffer& s_qkv,
    const ffi::AnyBuffer& w_out,
    const ffi::AnyBuffer& s_out,
    const ffi::AnyBuffer& w_gu,
    const ffi::AnyBuffer& s_gu,
    const ffi::AnyBuffer& w_down,
    const ffi::AnyBuffer& s_down,
    const ffi::AnyBuffer& qkv_fp8_scales,
    const ffi::AnyBuffer& final_norm_sb,
    const ffi::AnyBuffer& conv1pgv,
    const ffi::AnyBuffer& head_norm_sb,
    const ffi::AnyBuffer& linear_g_w,
    const ffi::AnyBuffer& pol_w,
    const ffi::AnyBuffer& pol_b,
    const ffi::AnyBuffer& q_w,
    const ffi::AnyBuffer& q_b,
    const ffi::AnyBuffer& pass_pol_w,
    const ffi::AnyBuffer& pass_pol_b,
    const ffi::AnyBuffer& pass_q_w,
    const ffi::AnyBuffer& pass_q_b,
    const ffi::AnyBuffer& val_l1_w,
    const ffi::AnyBuffer& val_l1_b,
    const ffi::AnyBuffer& val_out_w,
    const ffi::AnyBuffer& val_out_b,
    float* out_logits,
    float* out_vprobs,
    float* out_qprobs,
    int A,
    int K,
    const CacheWrite& cw,
    ffi::Result<ffi::AnyBuffer> sc_x,
    ffi::Result<ffi::AnyBuffer> sc_tmp,
    ffi::Result<ffi::AnyBuffer> sc_act_s8,
    ffi::Result<ffi::AnyBuffer> sc_act_sc) {
  const auto od = obs.dimensions();
  const int max_boards = static_cast<int>(od[0]);
  const int side = static_cast<int>(od[1]);
  const int cells = side * side;
  const int L = static_cast<int>(w_out.dimensions()[0]);
  const int FFN = static_cast<int>(w_down.dimensions()[2]);
  if (max_boards <= 0) return ffi::Error::Success();
  if (cells > 121) {
    return ffi::Error::InvalidArgument("kata_tf_board_int8 supports Hex boards up to 11x11 (cells <= 121)");
  }

  KataTfParams p{};
  p.obs = cptr<uint8_t>(obs);
  p.d_boards = d_boards;
  p.max_boards = max_boards;
  p.side = side;
  p.cells = cells;
  p.A = A;
  p.K = K;
  p.L = L;
  p.FFN = FFN;
  p.stem_w = cptr<__nv_bfloat16>(stem_w);
  p.stem_b = cptr<float>(stem_b);
  p.attn_norm_sb = cptr<float>(attn_norm_sb);
  p.ffn_norm_sb = cptr<float>(ffn_norm_sb);
  p.qk_norm_sb = cptr<float>(qk_norm_sb);
  p.rope_freqs = cptr<float>(rope_freqs);
  p.w_qkv = cptr<int8_t>(w_qkv);
  p.s_qkv = cptr<float>(s_qkv);
  p.w_out = cptr<int8_t>(w_out);
  p.s_out = cptr<float>(s_out);
  p.w_gu = cptr<int8_t>(w_gu);
  p.s_gu = cptr<float>(s_gu);
  p.w_down = cptr<int8_t>(w_down);
  p.s_down = cptr<float>(s_down);
  p.qkv_fp8_scales = cptr<float>(qkv_fp8_scales);

  p.heads.final_norm_sb = cptr<float>(final_norm_sb);
  p.heads.conv1pgv = cptr<__nv_bfloat16>(conv1pgv);
  p.heads.head_norm_sb = cptr<float>(head_norm_sb);
  p.heads.linear_g_w = cptr<float>(linear_g_w);
  p.heads.pol_w = cptr<float>(pol_w);
  p.heads.pol_b = cptr<float>(pol_b);
  p.heads.q_w = cptr<float>(q_w);
  p.heads.q_b = cptr<float>(q_b);
  p.heads.pass_pol_w = cptr<float>(pass_pol_w);
  p.heads.pass_pol_b = cptr<float>(pass_pol_b);
  p.heads.pass_q_w = cptr<float>(pass_q_w);
  p.heads.pass_q_b = cptr<float>(pass_q_b);
  p.heads.val_l1_w = cptr<float>(val_l1_w);
  p.heads.val_l1_b = cptr<float>(val_l1_b);
  p.heads.val_out_w = cptr<float>(val_out_w);
  p.heads.val_out_b = cptr<float>(val_out_b);

  p.logits = out_logits;
  p.vprobs = out_vprobs;
  p.qprobs = out_qprobs;
  p.cw = cw;

  p.cta_x = mptr<__nv_bfloat16>(*sc_x);
  p.cta_tmp = mptr<__nv_bfloat16>(*sc_tmp);
  p.cta_act_s8 = mptr<int8_t>(*sc_act_s8);
  p.cta_act_sc = mptr<float>(*sc_act_sc);

  const cudaError_t err = launch_kata_tf_board_int8(p, stream);
  if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
  return ffi::Error::Success();
}

ffi::Error kata_tf_network_entry(
    cudaStream_t stream,
    ffi::AnyBuffer obs,
    ffi::AnyBuffer stem_w, ffi::AnyBuffer stem_b,
    ffi::AnyBuffer attn_norm_sb, ffi::AnyBuffer ffn_norm_sb,
    ffi::AnyBuffer qk_norm_sb, ffi::AnyBuffer rope_freqs,
    ffi::AnyBuffer w_qkv, ffi::AnyBuffer s_qkv,
    ffi::AnyBuffer w_out, ffi::AnyBuffer s_out,
    ffi::AnyBuffer w_gu, ffi::AnyBuffer s_gu,
    ffi::AnyBuffer w_down, ffi::AnyBuffer s_down,
    ffi::AnyBuffer qkv_fp8_scales,
    ffi::AnyBuffer final_norm_sb, ffi::AnyBuffer conv1pgv, ffi::AnyBuffer head_norm_sb,
    ffi::AnyBuffer linear_g_w, ffi::AnyBuffer pol_w, ffi::AnyBuffer pol_b,
    ffi::AnyBuffer q_w, ffi::AnyBuffer q_b,
    ffi::AnyBuffer pass_pol_w, ffi::AnyBuffer pass_pol_b,
    ffi::AnyBuffer pass_q_w, ffi::AnyBuffer pass_q_b,
    ffi::AnyBuffer val_l1_w, ffi::AnyBuffer val_l1_b,
    ffi::AnyBuffer val_out_w, ffi::AnyBuffer val_out_b,
    ffi::Result<ffi::AnyBuffer> out_logits,
    ffi::Result<ffi::AnyBuffer> out_vprobs,
    ffi::Result<ffi::AnyBuffer> out_qprobs,
    ffi::Result<ffi::AnyBuffer> sc_x,
    ffi::Result<ffi::AnyBuffer> sc_tmp,
    ffi::Result<ffi::AnyBuffer> sc_act_s8,
    ffi::Result<ffi::AnyBuffer> sc_act_sc) {
  const int A = static_cast<int>(out_logits->dimensions()[1]);
  const int K = static_cast<int>(out_vprobs->dimensions()[1]);
  CacheWrite cw{};
  return run_kata_tf_common(
      stream, obs, nullptr,
      stem_w, stem_b, attn_norm_sb, ffn_norm_sb, qk_norm_sb, rope_freqs,
      w_qkv, s_qkv, w_out, s_out, w_gu, s_gu, w_down, s_down, qkv_fp8_scales,
      final_norm_sb, conv1pgv, head_norm_sb, linear_g_w, pol_w, pol_b, q_w, q_b,
      pass_pol_w, pass_pol_b, pass_q_w, pass_q_b, val_l1_w, val_l1_b, val_out_w, val_out_b,
      mptr<float>(*out_logits), mptr<float>(*out_vprobs), mptr<float>(*out_qprobs),
      A, K, cw, sc_x, sc_tmp, sc_act_s8, sc_act_sc);
}

ffi::Error kata_tf_compact_entry(
    cudaStream_t stream,
    ffi::AnyBuffer obs,
    ffi::AnyBuffer d_boards,
    ffi::AnyBuffer miss_keys,
    ffi::AnyBuffer stem_w, ffi::AnyBuffer stem_b,
    ffi::AnyBuffer attn_norm_sb, ffi::AnyBuffer ffn_norm_sb,
    ffi::AnyBuffer qk_norm_sb, ffi::AnyBuffer rope_freqs,
    ffi::AnyBuffer w_qkv, ffi::AnyBuffer s_qkv,
    ffi::AnyBuffer w_out, ffi::AnyBuffer s_out,
    ffi::AnyBuffer w_gu, ffi::AnyBuffer s_gu,
    ffi::AnyBuffer w_down, ffi::AnyBuffer s_down,
    ffi::AnyBuffer qkv_fp8_scales,
    ffi::AnyBuffer final_norm_sb, ffi::AnyBuffer conv1pgv, ffi::AnyBuffer head_norm_sb,
    ffi::AnyBuffer linear_g_w, ffi::AnyBuffer pol_w, ffi::AnyBuffer pol_b,
    ffi::AnyBuffer q_w, ffi::AnyBuffer q_b,
    ffi::AnyBuffer pass_pol_w, ffi::AnyBuffer pass_pol_b,
    ffi::AnyBuffer pass_q_w, ffi::AnyBuffer pass_q_b,
    ffi::AnyBuffer val_l1_w, ffi::AnyBuffer val_l1_b,
    ffi::AnyBuffer val_out_w, ffi::AnyBuffer val_out_b,
    ffi::AnyBuffer cache_keys_in,
    ffi::AnyBuffer cache_locks_in,
    ffi::AnyBuffer cache_v_in,
    ffi::AnyBuffer cache_q_in,
    ffi::AnyBuffer cache_wave_in,
    ffi::Result<ffi::AnyBuffer> out_vprobs,
    ffi::Result<ffi::AnyBuffer> out_qprobs,
    ffi::Result<ffi::AnyBuffer> cache_keys_out,
    ffi::Result<ffi::AnyBuffer> cache_locks_out,
    ffi::Result<ffi::AnyBuffer> cache_v_out,
    ffi::Result<ffi::AnyBuffer> cache_q_out,
    ffi::Result<ffi::AnyBuffer> sc_x,
    ffi::Result<ffi::AnyBuffer> sc_tmp,
    ffi::Result<ffi::AnyBuffer> sc_act_s8,
    ffi::Result<ffi::AnyBuffer> sc_act_sc,
    int64_t use_cache) {
  (void)cache_keys_in; (void)cache_locks_in; (void)cache_v_in; (void)cache_q_in;
  const int A = static_cast<int>(out_qprobs->dimensions()[1]);
  const int K = static_cast<int>(out_vprobs->dimensions()[1]);
  CacheWrite cw{};
  if (use_cache) {
    cw.miss_keys = cptr<CacheKey>(miss_keys);
    cw.table_keys = mptr<CacheKey>(*cache_keys_out);
    cw.table_locks = mptr<uint32_t>(*cache_locks_out);
    cw.table_v = mptr<__nv_bfloat16>(*cache_v_out);
    cw.table_q = mptr<__nv_bfloat16>(*cache_q_out);
    cw.cache_wave = cptr<uint32_t>(cache_wave_in);
  }
  return run_kata_tf_common(
      stream, obs, cptr<int32_t>(d_boards),
      stem_w, stem_b, attn_norm_sb, ffn_norm_sb, qk_norm_sb, rope_freqs,
      w_qkv, s_qkv, w_out, s_out, w_gu, s_gu, w_down, s_down, qkv_fp8_scales,
      final_norm_sb, conv1pgv, head_norm_sb, linear_g_w, pol_w, pol_b, q_w, q_b,
      pass_pol_w, pass_pol_b, pass_q_w, pass_q_b, val_l1_w, val_l1_b, val_out_w, val_out_b,
      nullptr, mptr<float>(*out_vprobs), mptr<float>(*out_qprobs),
      A, K, cw, sc_x, sc_tmp, sc_act_s8, sc_act_sc);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiKataTfNetwork, kata_tf_network_entry,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()  // obs
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // stem_w, stem_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // attn_norm_sb, ffn_norm_sb
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // qk_norm_sb, rope_freqs
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // w_qkv, s_qkv
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // w_out, s_out
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // w_gu, s_gu
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // w_down, s_down
        .Arg<ffi::AnyBuffer>()                        // qkv_fp8_scales
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // final_norm_sb, conv1pgv, head_norm_sb
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // linear_g_w, pol_w, pol_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // q_w, q_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // pass_pol_w, pass_pol_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // pass_q_w, pass_q_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // val_l1_w, val_l1_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // val_out_w, val_out_b
        .Ret<ffi::AnyBuffer>()  // out_logits
        .Ret<ffi::AnyBuffer>()  // out_vprobs
        .Ret<ffi::AnyBuffer>()  // out_qprobs
        .Ret<ffi::AnyBuffer>()  // sc_x
        .Ret<ffi::AnyBuffer>()  // sc_tmp
        .Ret<ffi::AnyBuffer>()  // sc_act_s8
        .Ret<ffi::AnyBuffer>()  // sc_act_sc
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiKataTfNetworkCompact, kata_tf_compact_entry,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()  // obs
        .Arg<ffi::AnyBuffer>()  // d_boards
        .Arg<ffi::AnyBuffer>()  // miss_keys
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // stem_w, stem_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // attn_norm_sb, ffn_norm_sb
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // qk_norm_sb, rope_freqs
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // w_qkv, s_qkv
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // w_out, s_out
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // w_gu, s_gu
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // w_down, s_down
        .Arg<ffi::AnyBuffer>()                        // qkv_fp8_scales
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // final_norm_sb, conv1pgv, head_norm_sb
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // linear_g_w, pol_w, pol_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // q_w, q_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // pass_pol_w, pass_pol_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // pass_q_w, pass_q_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // val_l1_w, val_l1_b
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()  // val_out_w, val_out_b
        .Arg<ffi::AnyBuffer>()  // cache_keys_in
        .Arg<ffi::AnyBuffer>()  // cache_locks_in
        .Arg<ffi::AnyBuffer>()  // cache_v_in
        .Arg<ffi::AnyBuffer>()  // cache_q_in
        .Arg<ffi::AnyBuffer>()  // cache_wave_in
        .Ret<ffi::AnyBuffer>()  // out_vprobs
        .Ret<ffi::AnyBuffer>()  // out_qprobs
        .Ret<ffi::AnyBuffer>()  // cache_keys_out
        .Ret<ffi::AnyBuffer>()  // cache_locks_out
        .Ret<ffi::AnyBuffer>()  // cache_v_out
        .Ret<ffi::AnyBuffer>()  // cache_q_out
        .Ret<ffi::AnyBuffer>()  // sc_x
        .Ret<ffi::AnyBuffer>()  // sc_tmp
        .Ret<ffi::AnyBuffer>()  // sc_act_s8
        .Ret<ffi::AnyBuffer>()  // sc_act_sc
        .Attr<int64_t>("use_cache")
);
