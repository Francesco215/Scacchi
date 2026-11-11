// 3x3 "SAME" convolution of 128 channels on Hex boards with 8-bit integer
// inputs: unsigned activations (after ReLU) times signed weights, summed
// exactly in int32 on the tensor cores (mma.sync m16n8k32 u8 x s8). On an RTX
// 5090 this instruction runs twice as often as its FP8 or bf16 counterparts
// (about 700 against 350 per microsecond and SM), and the sums do not depend
// on the order of the additions.
//
//   out[r, n] = wscale[n] * sum over taps t and input channels c of
//               in[neighbour(r, t), c] * W[n, t * 128 + c]
//
// with in = round(z / step[c]) in 0..255 and W = round(w * step[c] / wscale[n])
// in -127..127 (network.py chooses the steps). The epilogue then applies
// bias, BatchNorm, ReLU and the residual in bf16 exactly as conv_mma.cuh does
// and rounds the next convolution's input to 8 bits with its own steps.
//
// How the work is laid out:
//
//   - A unit of work is a few whole boards x 64 of the 128 output channels.
//     The CTA copies the unit's input rows into shared memory once and keeps
//     them for all 9 taps. ldmatrix takes one address per row, so a tap's A
//     fragments are gathered straight from those rows; a neighbour outside
//     the board points at a row of zeros. (conv_mma.cuh copies every input
//     row 9 times, once per tap.)
//   - The gather lets the rows of a fragment be any positions, so they are
//     sorted by border class: interior cells, then the four edges, each with
//     one corner. A fragment whose 16 rows all lack a tap (edge fragments lack
//     3 of 9) skips that tap's mma: the products it drops are all with the
//     zero padding.
//   - Only the weights stream through cp.async: one tap of the unit's 64
//     channels (8 KiB) per stage, two stages.
//   - CTAs are persistent: gridDim.x of them (two per SM) each take units
//     blockIdx.x, blockIdx.x + gridDim.x, ... .
//   - The epilogue stages its rows through shared memory so every global load
//     and store is a full 16-byte vector.
//
// Warp (wm, wn) owns fragments wm, wm + 4, wm + 8, wm + 12 (16 rows each) and
// channels wn * 32 .. + 31 of the unit: up to 4 x 4 mma per 32 input channels,
// 64 accumulators a thread.

#pragma once

#include "conv_mma.cuh"

namespace scacchi_net {

constexpr int I8_NH = 64;                  // output channels per unit
constexpr int I8_WSTAGE = I8_NH * CH;      // one tap of their weights, bytes
constexpr int I8_XPITCH = 2 * I8_NH + 16;  // scratch row of 64 bf16; + 16 so 8 rows hit 8 bank groups
constexpr int I8_ZPITCH = I8_NH + 16;      // scratch row of 64 bytes
constexpr int I8_THREADS = 256;
constexpr int I8_FRAGS = 16;               // 16-row fragments per unit: 4 warp rows x 4
constexpr int I8_ROWS = 16 * I8_FRAGS;     // rows of the fragments
constexpr int I8_MIN_SIDE = 3;             // an interior, edges and corners
constexpr int I8_SMEM_PAIR = 50176;        // most shared memory per CTA with two CTAs on an SM

constexpr int I8_CONSTS = 6;  // per-channel epilogue constants: bias, wscale, bn mean, mul, beta, qinv

// Shared memory for units of up to `cap` rows: the rows, a row of zeros, two
// weight stages, the table of rows by rank (16 bits each) and the epilogue
// constants of the CTA's 64 channels. Two CTAs fit an SM up to 50176 bytes each.
__host__ __device__ constexpr int i8_smem(int cap) {
  return (cap + 1) * CH + 2 * I8_WSTAGE + I8_ROWS * 2 + I8_CONSTS * I8_NH * 4;
}

// The rows of a unit in fragment order: interior cells, then top + top-left
// corner, left + bottom-left, bottom + bottom-right, right + top-right. Each
// group starts on a fragment boundary when all of them fit in I8_FRAGS that way.
struct I8Layout {
  int start[5];  // first rank of the group
  int size[5];   // its rows
};

__host__ __device__ inline I8Layout i8_layout(int side, int boards) {
  const int e = side - 2;
  I8Layout l;
  l.size[0] = boards * e * e;
  int padded = (l.size[0] + 15) / 16;
  for (int k = 1; k < 5; ++k) {
    l.size[k] = boards * (e + 1);
    padded += (l.size[k] + 15) / 16;
  }
  const bool pad = padded <= I8_FRAGS;
  int at = 0;
  for (int k = 0; k < 5; ++k) {
    l.start[k] = at;
    at += pad ? (l.size[k] + 15) / 16 * 16 : l.size[k];
  }
  return l;
}

// Row (board * cells + cell, within the unit) at a rank, or -1 for none.
__host__ __device__ inline int i8_row(const I8Layout& l, int side, int rank) {
  const int e = side - 2;
  int k = 0;
  for (int j = 1; j < 5; ++j)
    if (rank >= l.start[j]) k = j;
  const int i = rank - l.start[k];
  if (i >= l.size[k]) return -1;
  int board, y, x;
  if (k == 0) {
    board = i / (e * e);
    const int j = i - board * e * e;
    y = 1 + j / e;
    x = 1 + j % e;
  } else {
    const int edge = l.size[k] / (e + 1) * e;  // the edge cells of every board, then the corners
    int j;
    if (i < edge) {
      board = i / e;
      j = 1 + i % e;
    } else {
      board = i - edge;
      j = (k == 1 || k == 4) ? 0 : side - 1;
    }
    y = k == 1 ? 0 : k == 3 ? side - 1 : j;
    x = k == 2 ? 0 : k == 4 ? side - 1 : j;
  }
  return (board * side + y) * side + x;
}

// Taps (bit (dy + 1) * 3 + dx + 1) that some row of fragment f has on the board.
__host__ __device__ inline int i8_taps(const I8Layout& l, int f) {
  // interior: all; top: dy >= 0; left: dx >= 0; bottom: dy <= 0; right: dx <= 0
  const unsigned long long group_taps = 0x1FFull | (0x1F8ull << 9) | (0x1B6ull << 18) | (0x03Full << 27) | (0x0DBull << 36);
  int mask = 0;
  for (int k = 0; k < 5; ++k)
    if (l.start[k] < 16 * f + 16 && l.start[k] + l.size[k] > 16 * f) mask |= (int)((group_taps >> (9 * k)) & 0x1FF);
  return mask;
}

__device__ __forceinline__ float i8_bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ void mma_u8s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// z >= 0 on the next convolution's 8-bit grid: round(z * qinv), at most 255.
__device__ __forceinline__ uint32_t i8_quant(float z, float qinv) { return min(__float2uint_rn(z * qinv), 255u); }

struct I8Plan {
  int boards_per_unit;  // G
  int grid;             // persistent CTAs
  int smem;
};

// mma work of a unit of `boards` boards, in fragment-taps (9 for a full fragment).
inline int i8_unit_taps(int side, int boards) {
  const I8Layout l = i8_layout(side, boards);
  int taps = 0;
  for (int f = 0; f < I8_FRAGS; ++f) taps += __builtin_popcount(i8_taps(l, f));
  return taps;
}

// Boards per unit, among the sizes that leave room for two CTAs on an SM: the
// least time per row, taking a unit's time as its mma (in fragment-taps) plus a
// fixed part for its inputs, 9 weight stages and barriers. The fixed part,
// about 25 fragment-taps on an RTX 5090, is why large units win even where a
// smaller one would skip a little more padding.
constexpr int I8_UNIT_FIXED = 25;

inline bool i8_supports(int side) { return side >= I8_MIN_SIDE && i8_smem(side * side) <= I8_SMEM_PAIR; }

inline I8Plan i8_plan(int boards, int side, int sms) {
  const int cells = side * side;
  auto cost = [&](int g) { return I8_UNIT_FIXED + i8_unit_taps(side, g); };
  int best = 1;
  for (int g = 2; g * cells <= I8_ROWS && i8_smem(g * cells) <= I8_SMEM_PAIR; ++g)
    if ((long long)cost(g) * best <= (long long)cost(best) * g) best = g;
  if (best > boards) best = boards;
  const int units = 2 * ((boards + best - 1) / best);
  return {best, units < 2 * sms ? units : 2 * sms, i8_smem(best * cells)};
}

// G boards per unit. in: [rows, 128] u8; w: [128, 1152] s8. The epilogue's
// wscale is the weight step per output channel and out_zq the u8 output with
// inverse steps qinv[128]; without out_zq the output is bf16 (out_z).
__global__ void __launch_bounds__(I8_THREADS, 2) conv3x3_int8_kernel(
    const uint8_t* __restrict__ in, const int8_t* __restrict__ w, ConvShape sh, Epilogue ep,
    const float* __restrict__ qinv, int G) {
  extern __shared__ __align__(128) char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 1, wn = warp & 1;
  const int S = sh.side, cells = sh.cells;
  const int boards = sh.rows / cells;
  const int cap = G * cells;  // slot `cap` is the row of zeros
  const uint32_t z_u32 = smem_u32(smem);
  const uint32_t w_u32 = z_u32 + (cap + 1) * CH;
  const int units = 2 * ((boards + G - 1) / G);
  uint16_t* table = reinterpret_cast<uint16_t*>(smem + (cap + 1) * CH + 2 * I8_WSTAGE);  // [I8_ROWS]: row by rank
  int table_boards = 0;
  // Epilogue constants of channels n0 .. n0 + 63 as [kind][wn][lane & 3][nt][2]: a thread's pair of
  // one kind and nt is one 8-byte load, and a warp reads 64 consecutive bytes. (With shared memory
  // this large the SM has almost no L1 cache left, so global loads of them would each wait on L2.)
  float* cst = reinterpret_cast<float*>(table + I8_ROWS);
  int cst_n0 = -1;
  // This thread's ldmatrix row in each of the warp's 4 fragments as row | y << 12 | x << 16
  // (y = 15 for no row: every tap is off the board), and the fragments' taps.
  int ent[4], taps[4];

  for (int u = blockIdx.x; u < units; u += gridDim.x) {
    const int b0 = (u >> 1) * G, nb = min(G, boards - b0);
    const int n0 = (u & 1) * I8_NH;
    if (n0 != cst_n0) {
      if (cst_n0 >= 0) __syncthreads();
      for (int i = tid; i < I8_CONSTS * I8_NH; i += I8_THREADS) {
        const int kind = i >> 6, e = i & 63;
        const int col = (e >> 5) * 32 + ((e >> 1) & 3) * 8 + ((e >> 3) & 3) * 2 + (e & 1);
        const float* src = kind == 0 ? ep.bias : kind == 1 ? ep.wscale : kind == 2 ? ep.bn_mean : kind == 3 ? ep.bn_mul
                         : kind == 4 ? ep.bn_beta : qinv;
        cst[i] = (kind < 5 || ep.out_zq) ? src[n0 + col] : 0.f;
      }
      __syncthreads();
      cst_n0 = n0;
    }
    {
      const int rows = nb * cells;
      const size_t g0 = (size_t)b0 * cells * CH + n0;  // the unit's first element in [rows, 128] arrays
      if (nb != table_boards) {
        // Rows by rank for units of nb boards: every thread fills one entry, then reads its own.
        const I8Layout lay = i8_layout(S, nb);
        if (table_boards) __syncthreads();
        const int mine = i8_row(lay, S, tid);
        table[tid] = (uint16_t)(mine < 0 ? rows : mine);  // no row: the spare scratch row
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          const int f = wm + 4 * mt;
          const int row = table[16 * f + (lane & 15)];
          const int cell = row % cells;
          ent[mt] = row < rows ? row | ((cell / S) << 12) | ((cell % S) << 16) : (15 << 12);
          taps[mt] = i8_taps(lay, f);
        }
        table_boards = nb;
      }

      // Inputs: the unit's rows in natural order, then the zero row. A row's 8
      // chunks of 16 bytes are XOR-swizzled with its slot, so the rows of an
      // ldmatrix (mostly consecutive slots, or S apart) spread over the banks.
      for (int i = tid; i < (rows + 1) * 8; i += I8_THREADS) {
        const int r = i >> 3, chunk = i & 7;
        const bool ok = r < rows;
        const int slot = ok ? r : cap;
        cp_async16(z_u32 + slot * CH + ((chunk ^ (slot & 7)) << 4),
                   in + ((size_t)b0 * cells + (ok ? r : 0)) * CH + chunk * 16, ok);
      }
      auto load_weights = [&](int tap) {
        const uint32_t dst = w_u32 + (tap & 1) * I8_WSTAGE;
        const int8_t* src = w + (size_t)n0 * KDIM + tap * CH;
        for (int i = tid; i < I8_NH * 8; i += I8_THREADS) {
          const int n = i >> 3, chunk = i & 7;
          cp_async16(dst + n * CH + ((chunk ^ (n & 7)) << 4), src + (size_t)n * KDIM + chunk * 16, true);
        }
      };
      load_weights(0);
      cp_async_commit();

      int acc[4][4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int k = 0; k < 4; ++k) acc[i][j][k] = 0;

      for (int tap = 0; tap < TAPS; ++tap) {
        cp_async_wait<0>();
        __syncthreads();  // this tap's weights are in; every warp is done with the other stage
        if (tap + 1 < TAPS) {
          load_weights(tap + 1);
          cp_async_commit();
        }
        const int dy = tap / 3 - 1, dx = tap % 3 - 1;
        uint32_t arow[4];
        int asw[4];
        bool on[4];
        bool any = false;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          on[mt] = (taps[mt] >> tap) & 1;
          any |= on[mt];
          const int y = ((ent[mt] >> 12) & 15) + dy, x = ((ent[mt] >> 16) & 15) + dx;
          const bool ok = y >= 0 && y < S && x >= 0 && x < S;
          const int slot = ok ? (ent[mt] & 0xFFF) + dy * S + dx : cap;
          arow[mt] = z_u32 + slot * CH;
          asw[mt] = slot & 7;
        }
        if (!any) continue;
        const uint32_t wst = w_u32 + (tap & 1) * I8_WSTAGE;
#pragma unroll
        for (int ks = 0; ks < CH / 32; ++ks) {
          uint32_t b[2][4];
#pragma unroll
          for (int np = 0; np < 2; ++np) {
            const int n = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
            ldmatrix_x4(b[np], wst + n * CH + (((ks * 2 + ((lane >> 3) & 1)) ^ (n & 7)) << 4));
          }
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) {
            if (!on[mt]) continue;
            uint32_t a[4];
            ldmatrix_x4(a, arow[mt] + (((ks * 2 + (lane >> 4)) ^ asw[mt]) << 4));
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) mma_u8s8(acc[mt][nt], a, b[nt >> 1][(nt & 1) * 2], b[nt >> 1][(nt & 1) * 2 + 1]);
          }
        }
      }
      __syncthreads();  // shared memory becomes the epilogue's scratch

      // Epilogue as in conv_mma.cuh: v = bf16(bf16(sum * wscale) + bias)
      // [+ residual in bf16], then z = relu(bf16(bn(v))). Scratch rows are the
      // unit's rows in natural order; a fragment row that is no row writes to
      // the spare row `rows`.
      int er[4][2];  // rows of this thread's accumulators: fragment row lane / 4, and + 8
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int half = 0; half < 2; ++half) er[mt][half] = table[16 * (wm + 4 * mt) + (lane >> 2) + half * 8];
      if (ep.residual) {
        for (int i = tid; i < rows * 8; i += I8_THREADS) {
          const int r = i >> 3, c = i & 7;
          cp_async16(z_u32 + r * I8_XPITCH + c * 16, ep.residual + g0 + (size_t)r * CH + c * 8, true);
        }
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
      }
      const bool use_x = ep.residual || ep.out_x;
      auto constant = [&](int kind, int nt) {
        return *reinterpret_cast<const float2*>(cst + kind * I8_NH + (((wn * 4 + (lane & 3)) * 4 + nt) << 1));
      };
      // z = relu(bf16(bn(v))) of one pair to the scratch, on the next layer's 8-bit grid or in bf16.
      auto put_z = [&](float v0, float v1, float2 mean, float2 mul, float2 beta, float2 q, int row, int col) {
        const float z0 = fmaxf(i8_bf(fmaf(v0 - mean.x, mul.x, beta.x)), 0.f);
        const float z1 = fmaxf(i8_bf(fmaf(v1 - mean.y, mul.y, beta.y)), 0.f);
        if (ep.out_zq)
          *reinterpret_cast<uint16_t*>(smem + row * I8_ZPITCH + col) = (uint16_t)(i8_quant(z0, q.x) | (i8_quant(z1, q.y) << 8));
        else
          *reinterpret_cast<__nv_bfloat162*>(smem + row * I8_XPITCH + col * 2) = __floats2bfloat162_rn(z0, z1);
      };
      // Pass 1: v. Without a residual stream z follows at once; with one, x goes
      // through the scratch first and v waits in acc (as float bits) for pass 2.
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = wn * 32 + nt * 8 + (lane & 3) * 2;
        const float2 bias = constant(0, nt), scale = constant(1, nt);
        float2 mean, mul, beta, q;
        if (!use_x) { mean = constant(2, nt); mul = constant(3, nt); beta = constant(4, nt); q = constant(5, nt); }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            float v0 = i8_bf((float)acc[mt][nt][half * 2] * scale.x);
            float v1 = i8_bf((float)acc[mt][nt][half * 2 + 1] * scale.y);
            v0 = i8_bf(v0 + bias.x);
            v1 = i8_bf(v1 + bias.y);
            if (!use_x) {
              put_z(v0, v1, mean, mul, beta, q, er[mt][half], col);
              continue;
            }
            __nv_bfloat162* xp = reinterpret_cast<__nv_bfloat162*>(smem + er[mt][half] * I8_XPITCH + col * 2);
            if (ep.residual) {
              const __nv_bfloat162 res = *xp;
              v0 = i8_bf(v0 + __bfloat162float(res.x));
              v1 = i8_bf(v1 + __bfloat162float(res.y));
            }
            acc[mt][nt][half * 2] = __float_as_int(v0);
            acc[mt][nt][half * 2 + 1] = __float_as_int(v1);
            if (ep.out_x) *xp = __floats2bfloat162_rn(v0, v1);
          }
        }
      }
      if (use_x) {
        if (ep.out_x) {
          __syncthreads();
          for (int i = tid; i < rows * 8; i += I8_THREADS) {
            const int r = i >> 3, c = i & 7;
            *reinterpret_cast<uint4*>(ep.out_x + g0 + (size_t)r * CH + c * 8) =
                *reinterpret_cast<const uint4*>(smem + r * I8_XPITCH + c * 16);
          }
          __syncthreads();
        }
        // Pass 2: z through the scratch.
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int col = wn * 32 + nt * 8 + (lane & 3) * 2;
          const float2 mean = constant(2, nt), mul = constant(3, nt), beta = constant(4, nt), q = constant(5, nt);
#pragma unroll
          for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int half = 0; half < 2; ++half)
              put_z(__int_as_float(acc[mt][nt][half * 2]), __int_as_float(acc[mt][nt][half * 2 + 1]), mean, mul, beta, q,
                    er[mt][half], col);
        }
      }
      __syncthreads();
      if (ep.out_zq) {
        for (int i = tid; i < rows * 4; i += I8_THREADS) {
          const int r = i >> 2, c = i & 3;
          *reinterpret_cast<uint4*>(ep.out_zq + g0 + (size_t)r * CH + c * 16) =
              *reinterpret_cast<const uint4*>(smem + r * I8_ZPITCH + c * 16);
        }
      } else {
        for (int i = tid; i < rows * 8; i += I8_THREADS) {
          const int r = i >> 3, c = i & 7;
          *reinterpret_cast<uint4*>(ep.out_z + g0 + (size_t)r * CH + c * 8) =
              *reinterpret_cast<const uint4*>(smem + r * I8_XPITCH + c * 16);
        }
      }
      __syncthreads();  // the next unit's loads reuse the scratch
    }
  }
}

}  // namespace scacchi_net
