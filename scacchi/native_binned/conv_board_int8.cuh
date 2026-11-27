// Single-kernel board-resident INT8 forward pass for AZBinnedNet (128 channels,
// 6 ResNet-v2 blocks, policy + value + Q heads) on Hex boards (sides 3..11).
//
// Each persistent CTA owns G whole boards (1 board on Hex 6..11, up to 7 on
// smaller boards) and keeps both the bf16 residual stream X and the u8
// quantized activations Zq in shared memory from the 3x3 stem through all 12
// residual convolutions and the three output heads:
//   - zero intermediate global memory reads or writes (xs, zs, z2s unused);
//   - zero per-layer global input cp.async loads;
//   - 1 kernel launch per forward call instead of 14;
//   - exact int32 tensor-core sums (mma.sync m16n8k32 u8 x s8), matching the
//     multi-kernel int8 path bit for bit;
//   - optional device-side dynamic board count (*d_boards) and GPU position
//     cache writeback, compatible with CUDA Graph / XLA command-buffer capture.

#pragma once

#include "cache_key.cuh"
#include "conv_int8.cuh"

namespace scacchi_net {

constexpr int B_THREADS = 256;
constexpr int B_XPITCH = CH + 8;     // 136 bf16 = 272 bytes (16-byte aligned, bank-conflict free)
constexpr int B_WSTAGE = CH * 64;    // 8192 bytes: all 128 out channels x 64 in channels (half a tap)
constexpr int B_HEAD_CH = 5;

struct NetHeads {
  const __nv_bfloat16* conv_w;  // [5][128]
  const float* conv_v;          // [4][5]
  const __nv_bfloat16* pol_w;   // [2C][A]
  const float* pol_b;           // [A]
  const __nv_bfloat16* val_w;   // [C][128]
  const float* val_b;           // [128]
  const float* vout_w;          // [128][K]
  const float* vout_b;          // [K]
  const __nv_bfloat16* q_w;     // [2C][128]
  const float* q_b;             // [128]
  const float* qout_w;          // [128][A K]
  const float* qout_b;          // [A K]
};

struct CacheWrite {
  const CacheKey* miss_keys;   // [max_boards]: key of compacted slot b, meta bit 31 == 1 if valid
  CacheKey* table_keys;        // [CACHE_SLOTS]
  uint32_t* table_locks;       // [CACHE_SLOTS]: 0 empty/unlocked, 1 locked, 2 ready
  float* table_v;              // [CACHE_SLOTS][K]
  float* table_q;              // [CACHE_SLOTS][A * K]
};

template <int FRAGS>
__host__ __device__ inline I8Layout board_layout(int side, int boards) {
  const int e = side - 2;
  I8Layout l;
  l.size[0] = boards * e * e;
  int padded = (l.size[0] + 15) / 16;
  for (int k = 1; k < 5; ++k) {
    l.size[k] = boards * (e + 1);
    padded += (l.size[k] + 15) / 16;
  }
  const bool pad = padded <= FRAGS;
  int at = 0;
  for (int k = 0; k < 5; ++k) {
    l.start[k] = at;
    at += pad ? (l.size[k] + 15) / 16 * 16 : l.size[k];
  }
  return l;
}

template <int MW>
__host__ __device__ constexpr int board_net_smem() {
  constexpr int ROWS = 32 * MW;
  const int x_bytes = (ROWS + 1) * B_XPITCH * 2;
  const int z_bytes = (ROWS + 1) * CH;
  const int w_bytes = 2 * B_WSTAGE;
  const int t_bytes = ROWS * 2;
  return x_bytes + z_bytes + w_bytes + t_bytes;
}

// Stem row helper: vec row indices matching network.cu.
__host__ __device__ constexpr int b_bn1_row(int i) { return 1 + 8 * i; }
__host__ __device__ constexpr int b_b1_row(int i) { return 4 + 8 * i; }
__host__ __device__ constexpr int b_bn2_row(int i) { return 5 + 8 * i; }
__host__ __device__ constexpr int b_b2_row(int i) { return 8 + 8 * i; }
constexpr int B_FINAL_BN = 49;

// MW = 2 (64 rows max, 2 CTAs/SM, ~42.5 KB smem: Hex 3..8)
// MW = 4 (128 rows max, 1 CTA/SM, ~68.2 KB smem: Hex 9..11)
template <int MW, int G, int LAUNCH_BOUNDS_BLOCKS>
__global__ void __launch_bounds__(B_THREADS, LAUNCH_BOUNDS_BLOCKS) board_net_int8_kernel(
    const uint8_t* __restrict__ obs,
    const int32_t* __restrict__ d_boards,
    int max_boards,
    int side,
    int cells,
    int A,
    int K,
    const __nv_bfloat16* __restrict__ stem_w,
    const int8_t* __restrict__ trunk_w,
    const float* __restrict__ wscale,
    const float* __restrict__ qinv,
    const float* __restrict__ vec,
    NetHeads h,
    float* __restrict__ logits,
    float* __restrict__ vprobs,
    float* __restrict__ qprobs,
    CacheWrite cw) {
  const int boards = d_boards ? *d_boards : max_boards;
  if (boards <= 0) return;

  constexpr int FRAGS = 2 * MW;
  constexpr int ROWS = 16 * FRAGS;  // 64 or 128

  extern __shared__ __align__(128) char smem[];
  __nv_bfloat16* X_smem = reinterpret_cast<__nv_bfloat16*>(smem);
  char* z_ptr = smem + (ROWS + 1) * B_XPITCH * 2;  // 16-byte aligned (272 * (ROWS + 1))
  char* w_ptr = z_ptr + (ROWS + 1) * CH;
  uint16_t* table = reinterpret_cast<uint16_t*>(w_ptr + 2 * B_WSTAGE);
  const uint32_t z_u32 = smem_u32(z_ptr);
  const uint32_t w_u32 = smem_u32(w_ptr);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2;  // 0..1
  const int wn = warp & 3;   // 0..3
  const int AK = A * K;

  int table_boards = 0;
  int ent[MW], taps[MW], er[MW][2];

  for (int b0 = blockIdx.x * G; b0 < boards; b0 += gridDim.x * G) {
    const int nb = min(G, boards - b0);
    const int rows = nb * cells;
    const int cap = rows;  // zero-padding row index in Zq_smem

    // ---- 1. Stem: 3x3 conv 4 -> 128 + bias (X_smem) + bn1_0 + ReLU + u8 quant (Zq_smem)
    // Stage stem weights [36][128] float (18 KB) temporarily in z_ptr..w_ptr (24+ KB free).
    float* ws = reinterpret_cast<float*>(z_ptr);
    for (int i = tid; i < CH * 36; i += B_THREADS) {
      const int n = i / 36, k = i - n * 36;
      ws[k * CH + n] = __bfloat162float(stem_w[i]);
    }
    __syncthreads();

    constexpr int MAX_STEM_IT = (ROWS + 15) / 16;  // 4 for MW=2, 8 for MW=4
    uint2 q_reg[MAX_STEM_IT];
    const int n0 = (tid & 15) * 8;
#pragma unroll
    for (int it = 0; it < MAX_STEM_IT; ++it) {
      const int r = it * 16 + (tid >> 4);
      q_reg[it] = make_uint2(0u, 0u);
      if (r < rows) {
        const int brd = r / cells, cell = r - brd * cells;
        const int y = cell / side, x = cell - y * side;
        float acc_s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int t = 0; t < 9; ++t) {
          const int yy = y + t / 3 - 1, xx = x + t % 3 - 1;
          if (yy < 0 || yy >= side || xx < 0 || xx >= side) continue;
          const uint32_t o = *reinterpret_cast<const uint32_t*>(
              obs + (size_t)((b0 + brd) * cells + yy * side + xx) * 4);
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            if (!((o >> (8 * c)) & 0xffu)) continue;
            const float4* wr = reinterpret_cast<const float4*>(ws + (t * 4 + c) * CH + n0);
            const float4 w0 = wr[0], w1 = wr[1];
            acc_s[0] += w0.x; acc_s[1] += w0.y; acc_s[2] += w0.z; acc_s[3] += w0.w;
            acc_s[4] += w1.x; acc_s[5] += w1.y; acc_s[6] += w1.z; acc_s[7] += w1.w;
          }
        }
        __align__(16) __nv_bfloat16 xo[8];
        uint32_t q0 = 0u, q1 = 0u;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int n = n0 + j;
          const float v = i8_bf(i8_bf(acc_s[j]) + vec[n]);
          xo[j] = __float2bfloat16_rn(v);
          const float zf = fmaxf(i8_bf(fmaf(v - vec[b_bn1_row(0) * CH + n],
                                            vec[(b_bn1_row(0) + 1) * CH + n],
                                            vec[(b_bn1_row(0) + 2) * CH + n])), 0.f);
          const float zr = i8_bf(zf);
          const uint32_t byte = i8_quant(zr, qinv[n]);
          if (j < 4) q0 |= byte << (8 * j);
          else q1 |= byte << (8 * (j - 4));
        }
        *reinterpret_cast<uint4*>(X_smem + r * B_XPITCH + n0) = *reinterpret_cast<const uint4*>(xo);
        q_reg[it] = make_uint2(q0, q1);
      }
    }
    __syncthreads();  // done reading ws; z_ptr now becomes Zq_smem

    // Write stem's quantized Zq in swizzled 16-byte chunks + zero row `cap`.
    // Note: n0 = (tid & 15) * 8; two adjacent threads (even, odd) form one 16-byte chunk `n0 >> 4`!
    // Write 8 bytes (`uint2`) directly at swizzled chunk `((n0 >> 4) ^ (r & 7)) << 4` + `(n0 & 8)`.
#pragma unroll
    for (int it = 0; it < MAX_STEM_IT; ++it) {
      const int r = it * 16 + (tid >> 4);
      if (r < rows) {
        const int chunk = n0 >> 4;
        const int off = ((chunk ^ (r & 7)) << 4) + (n0 & 8);
        *reinterpret_cast<uint2*>(z_ptr + r * CH + off) = q_reg[it];
      }
    }
    if (tid < CH / 4) {
      reinterpret_cast<uint32_t*>(z_ptr + cap * CH)[tid] = 0u;
    }

    if (nb != table_boards) {
      const I8Layout lay = board_layout<FRAGS>(side, nb);
      if (tid < ROWS) {
        const int mine = i8_row(lay, side, tid);
        table[tid] = (uint16_t)(mine < 0 ? rows : mine);
      }
      __syncthreads();
#pragma unroll
      for (int mt = 0; mt < MW; ++mt) {
        const int f = wm * MW + mt;
        const int row = table[16 * f + (lane & 15)];
        const int cell = row % cells;
        ent[mt] = row < rows ? row | ((cell / side) << 12) | ((cell % side) << 16) : (15 << 12);
        taps[mt] = i8_taps(lay, f);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          er[mt][half] = table[16 * f + (lane >> 2) + half * 8];
        }
      }
      table_boards = nb;
    } else {
      __syncthreads();
    }

    // ---- 2. Twelve residual 3x3 INT8 convolutions, board-resident in smem ------
    for (int layer = 0; layer < 12; ++layer) {
      const int8_t* wl = trunk_w + (size_t)layer * (CH * KDIM);
      auto load_wstage = [&](int st) {
        const uint32_t dst = w_u32 + (st & 1) * B_WSTAGE;
        const int tap = st >> 1, khalf = st & 1;
        const int k0 = tap * CH + khalf * 64;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int idx = tid + B_THREADS * i;
          const int n = idx >> 2, chunk = idx & 3;
          cp_async16(dst + swz(n, chunk), wl + (size_t)n * KDIM + k0 + chunk * 16, true);
        }
      };

      load_wstage(0);
      cp_async_commit();

      int acc[MW][4][4];
#pragma unroll
      for (int mt = 0; mt < MW; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int k = 0; k < 4; ++k) acc[mt][nt][k] = 0;

      for (int st = 0; st < 18; ++st) {
        cp_async_wait<0>();
        __syncthreads();
        if (st + 1 < 18) {
          load_wstage(st + 1);
          cp_async_commit();
        }
        const int tap = st >> 1, khalf = st & 1;
        const int dy = tap / 3 - 1, dx = tap % 3 - 1;
        uint32_t arow[MW];
        int asw[MW];
        bool on[MW];
        bool any = false;
#pragma unroll
        for (int mt = 0; mt < MW; ++mt) {
          on[mt] = (taps[mt] >> tap) & 1;
          any |= on[mt];
          const int y = ((ent[mt] >> 12) & 15) + dy, x = ((ent[mt] >> 16) & 15) + dx;
          const bool ok = y >= 0 && y < side && x >= 0 && x < side;
          const int slot = ok ? (ent[mt] & 0xFFF) + dy * side + dx : cap;
          arow[mt] = z_u32 + slot * CH;
          asw[mt] = slot & 7;
        }
        if (!any) continue;
        const uint32_t wst = w_u32 + (st & 1) * B_WSTAGE;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t b_frag[2][4];
#pragma unroll
          for (int np = 0; np < 2; ++np) {
            const int n = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
            const int wchunk = ks * 2 + ((lane >> 3) & 1);
            ldmatrix_x4(b_frag[np], wst + swz(n, wchunk));
          }
          const int ks_full = khalf * 2 + ks;
#pragma unroll
          for (int mt = 0; mt < MW; ++mt) {
            if (!on[mt]) continue;
            uint32_t a_frag[4];
            ldmatrix_x4(a_frag, arow[mt] + (((ks_full * 2 + (lane >> 4)) ^ asw[mt]) << 4));
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
              mma_u8s8(acc[mt][nt], a_frag, b_frag[nt >> 1][(nt & 1) * 2], b_frag[nt >> 1][(nt & 1) * 2 + 1]);
            }
          }
        }
      }
      __syncthreads();  // all warps finished reading Zq_smem for this layer

      // Fused in-register + smem epilogue: scale, bias, [+ residual], BN, ReLU, quantize.
      const bool is_conv2 = (layer & 1) == 1;
      const int blk = layer >> 1;
      const int bias_r = is_conv2 ? b_b2_row(blk) : b_b1_row(blk);
      const int bn_r = is_conv2 ? (blk + 1 < 6 ? b_bn1_row(blk + 1) : B_FINAL_BN) : b_bn2_row(blk);
      const float* ws_l = wscale + layer * CH;
      const float* qi_next = (layer < 11) ? (qinv + (layer + 1) * CH) : nullptr;

#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = wn * 32 + nt * 8 + (lane & 3) * 2;
        const float2 sc = *reinterpret_cast<const float2*>(ws_l + col);
        const float2 bi = *reinterpret_cast<const float2*>(vec + bias_r * CH + col);
        const float2 mn = *reinterpret_cast<const float2*>(vec + bn_r * CH + col);
        const float2 ml = *reinterpret_cast<const float2*>(vec + (bn_r + 1) * CH + col);
        const float2 bt = *reinterpret_cast<const float2*>(vec + (bn_r + 2) * CH + col);
        const float2 qi = qi_next ? *reinterpret_cast<const float2*>(qi_next + col) : make_float2(0.f, 0.f);
#pragma unroll
        for (int mt = 0; mt < MW; ++mt) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int r = er[mt][half];
            if (r >= rows) continue;
            float v0 = i8_bf((float)acc[mt][nt][half * 2] * sc.x);
            float v1 = i8_bf((float)acc[mt][nt][half * 2 + 1] * sc.y);
            v0 = i8_bf(v0 + bi.x);
            v1 = i8_bf(v1 + bi.y);
            if (is_conv2) {
              const __nv_bfloat162 res = *reinterpret_cast<const __nv_bfloat162*>(X_smem + r * B_XPITCH + col);
              v0 = i8_bf(v0 + __bfloat162float(res.x));
              v1 = i8_bf(v1 + __bfloat162float(res.y));
              if (layer < 11) {
                *reinterpret_cast<__nv_bfloat162*>(X_smem + r * B_XPITCH + col) = __floats2bfloat162_rn(v0, v1);
              }
            }
            const float z0 = fmaxf(i8_bf(fmaf(v0 - mn.x, ml.x, bt.x)), 0.f);
            const float z1 = fmaxf(i8_bf(fmaf(v1 - mn.y, ml.y, bt.y)), 0.f);
            if (layer < 11) {
              const int zoff = (((col >> 4) ^ (r & 7)) << 4) + (col & 15);
              *reinterpret_cast<uint16_t*>(z_ptr + r * CH + zoff) =
                  (uint16_t)(i8_quant(z0, qi.x) | (i8_quant(z1, qi.y) << 8));
            } else {
              *reinterpret_cast<__nv_bfloat162*>(X_smem + r * B_XPITCH + col) = __floats2bfloat162_rn(z0, z1);
            }
          }
        }
      }
      __syncthreads();
    }

    // ---- 3. Policy, Value and Q heads directly from X_smem --------------------
    // z_ptr..w_ptr (24.7 KB for MW=2, 32.9 KB for MW=4) is now free float scratch.
    float* sh = reinterpret_cast<float*>(z_ptr);
    float* feat = sh;                            // [G][cells][5]
    float* vh = feat + G * cells * B_HEAD_CH;    // [G][128]
    float* qh = vh + G * CH;                     // [G][128]
    float* ql = qh + G * CH;                     // [max(G * A * K, 640)]

    float* cws = ql;  // [128][5] before Q logits use this space
    for (int i = tid; i < B_HEAD_CH * CH; i += B_THREADS) {
      const int o = i / CH, c = i - o * CH;
      cws[c * B_HEAD_CH + o] = __bfloat162float(h.conv_w[i]);
    }
    __syncthreads();

    for (int rr = tid; rr < rows; rr += B_THREADS) {
      const uint4* zr = reinterpret_cast<const uint4*>(X_smem + rr * B_XPITCH);
      float part[B_HEAD_CH] = {0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int q = 0; q < CH / 8; ++q) {
        const uint4 raw = zr[q];
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float lo = __low2float(p2[e]), hi = __high2float(p2[e]);
          const float* w0 = cws + (q * 8 + 2 * e) * B_HEAD_CH;
#pragma unroll
          for (int o = 0; o < B_HEAD_CH; ++o) part[o] = fmaf(hi, w0[B_HEAD_CH + o], fmaf(lo, w0[o], part[o]));
        }
      }
#pragma unroll
      for (int o = 0; o < B_HEAD_CH; ++o) {
        const float v = i8_bf(i8_bf(part[o]) + h.conv_v[o]);
        feat[rr * B_HEAD_CH + o] = fmaxf(
            i8_bf(fmaf(v - h.conv_v[B_HEAD_CH + o], h.conv_v[2 * B_HEAD_CH + o], h.conv_v[3 * B_HEAD_CH + o])),
            0.f);
      }
    }
    __syncthreads();

    if (logits) {
      for (int a = tid; a < A; a += B_THREADS) {
        float acc_p[G];
#pragma unroll
        for (int bb = 0; bb < G; ++bb) acc_p[bb] = 0.f;
        for (int i = 0; i < 2 * cells; ++i) {
          const float wv = __bfloat162float(h.pol_w[(size_t)i * A + a]);
          const int c = i >> 1, ch = i & 1;
#pragma unroll
          for (int bb = 0; bb < G; ++bb) {
            if (bb < nb) acc_p[bb] = fmaf(feat[(bb * cells + c) * B_HEAD_CH + ch], wv, acc_p[bb]);
          }
        }
        const float bias = h.pol_b[a];
        for (int bb = 0; bb < nb; ++bb) logits[(size_t)(b0 + bb) * A + a] = i8_bf(i8_bf(acc_p[bb]) + bias);
      }
    }
    {
      const int j = tid & (CH - 1);
      const bool is_q = tid >= CH;
      float acc_h[G];
#pragma unroll
      for (int bb = 0; bb < G; ++bb) acc_h[bb] = 0.f;
      if (!is_q) {
        for (int c = 0; c < cells; ++c) {
          const float wv = __bfloat162float(h.val_w[(size_t)c * CH + j]);
#pragma unroll
          for (int bb = 0; bb < G; ++bb) {
            if (bb < nb) acc_h[bb] = fmaf(feat[(bb * cells + c) * B_HEAD_CH + 2], wv, acc_h[bb]);
          }
        }
      } else {
        for (int i = 0; i < 2 * cells; ++i) {
          const float wv = __bfloat162float(h.q_w[(size_t)i * CH + j]);
          const int c = i >> 1, ch = 3 + (i & 1);
#pragma unroll
          for (int bb = 0; bb < G; ++bb) {
            if (bb < nb) acc_h[bb] = fmaf(feat[(bb * cells + c) * B_HEAD_CH + ch], wv, acc_h[bb]);
          }
        }
      }
      const float bias = is_q ? h.q_b[j] : h.val_b[j];
      float* dst = is_q ? qh : vh;
#pragma unroll
      for (int bb = 0; bb < G; ++bb) {
        if (bb < nb) dst[bb * CH + j] = fmaxf(i8_bf(i8_bf(acc_h[bb]) + bias), 0.f);
      }
    }
    __syncthreads();

    float* vl = ql;  // [G][K]
    for (int it = tid; it < nb * K; it += B_THREADS) {
      const int bb = it / K, k = it - bb * K;
      float sacc = 0.f;
      for (int j = 0; j < CH; ++j) sacc = fmaf(vh[bb * CH + j], h.vout_w[j * K + k], sacc);
      vl[bb * K + k] = sacc + h.vout_b[k];
    }
    __syncthreads();
    if (tid < nb) {
      const int bb = tid;
      float mx = -INFINITY;
      for (int k = 0; k < K; ++k) mx = fmaxf(mx, vl[bb * K + k]);
      float tot = 0.f;
      for (int k = 0; k < K; ++k) tot += expf(vl[bb * K + k] - mx);
      for (int k = 0; k < K; ++k) vprobs[(size_t)(b0 + bb) * K + k] = expf(vl[bb * K + k] - mx) / tot;
    }
    __syncthreads();

    for (int j = tid; j < AK; j += B_THREADS) {
      float acc_q[G];
#pragma unroll
      for (int bb = 0; bb < G; ++bb) acc_q[bb] = 0.f;
      for (int i = 0; i < CH; ++i) {
        const float wv = h.qout_w[(size_t)i * AK + j];
#pragma unroll
        for (int bb = 0; bb < G; ++bb) {
          if (bb < nb) acc_q[bb] = fmaf(qh[bb * CH + i], wv, acc_q[bb]);
        }
      }
      const float bias = h.qout_b[j];
#pragma unroll
      for (int bb = 0; bb < G; ++bb) {
        if (bb < nb) ql[bb * AK + j] = acc_q[bb] + bias;
      }
    }
    __syncthreads();

    for (int it = tid; it < nb * A; it += B_THREADS) {
      const int bb = it / A, a = it - bb * A;
      const float* l = ql + bb * AK + a * K;
      float mx = -INFINITY;
      for (int k = 0; k < K; ++k) mx = fmaxf(mx, l[k]);
      float tot = 0.f;
      for (int k = 0; k < K; ++k) tot += expf(l[k] - mx);
      float* out = qprobs + ((size_t)(b0 + bb) * A + a) * K;
      for (int k = 0; k < K; ++k) out[k] = expf(l[k] - mx) / tot;
    }
    __syncthreads();

    // Optional lock-free writeback to the GPU position cache (write-only phase:
    // no reader executes concurrently with this kernel).
    if (cw.miss_keys) {
      for (int bb = 0; bb < nb; ++bb) {
        const CacheKey mk = cw.miss_keys[b0 + bb];
        if (!(mk.meta & 0x80000000u)) continue;
        const uint32_t slot = hash_cache_key(mk);
        __shared__ int won_slot;
        if (tid == 0) {
          won_slot = (atomicCAS(cw.table_locks + slot, 0u, 1u) == 0u) ? (int)slot : -1;
          if (won_slot >= 0) cw.table_keys[slot] = mk;
        }
        __syncthreads();
        if (won_slot >= 0) {
          const float* vp = vprobs + (size_t)(b0 + bb) * K;
          const float* qp = qprobs + (size_t)(b0 + bb) * AK;
          for (int k = tid; k < K; k += B_THREADS) cw.table_v[(size_t)slot * K + k] = vp[k];
          for (int i = tid; i < AK; i += B_THREADS) cw.table_q[(size_t)slot * AK + i] = qp[i];
          __syncthreads();
          if (tid == 0) {
            __threadfence();
            cw.table_locks[slot] = 2u;
          }
        }
        __syncthreads();
      }
    }
  }
}

}  // namespace scacchi_net
