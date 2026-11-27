// Shared GPU position-cache key, hash, and wave-tagged lock-free overwrite helpers
// for search.cu, conv_board_int8.cuh, network.cu, and inkling_network.cu.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace scacchi_net {

constexpr int CACHE_SLOTS = 524288;  // 512K slots (power of two; bf16 v/q -> ~640 MB on Hex 8)
constexpr uint32_t CACHE_MASK = CACHE_SLOTS - 1;

struct CacheKey {
  uint64_t my_lo, my_hi, op_lo, op_hi;
  uint32_t meta;  // bit 0: color != 0, bit 1: step == 1, bit 31: occupied
  uint32_t pad;
};

struct CacheWrite {
  const CacheKey* miss_keys;    // [max_boards]: key of compacted slot b, meta bit 31 == 1 if valid
  CacheKey* table_keys;         // [CACHE_SLOTS]
  uint32_t* table_locks;        // [CACHE_SLOTS]: 0 empty, 1 writing, >=2 valid (2 + wave_id)
  __nv_bfloat16* table_v;       // [CACHE_SLOTS][K]
  __nv_bfloat16* table_q;       // [CACHE_SLOTS][A * K]
  const uint32_t* cache_wave;   // [1] current wave id
};

__host__ __device__ __forceinline__ uint32_t hash_cache_key(const CacheKey& k) {
  uint64_t h = k.my_lo * 0x9E3779B97F4A7C15ull;
  h ^= k.op_lo * 0xBF58476D1CE4E5B9ull;
  h ^= k.my_hi * 0x94D049BB133111EBull;
  h ^= k.op_hi * 0xD6E8FEB86659FD93ull;
  h ^= (uint64_t)k.meta * 0x2545F4914F6CDD1Dull;
  h ^= h >> 32;
  h *= 0xD6E8FEB86659FD93ull;
  h ^= h >> 32;
  return (uint32_t)h & CACHE_MASK;
}

// Acquire slot for writing in wave `tag = 2 + *cache_wave`:
//   - Allows inserting into empty slots (old == 0) AND evicting stale slots (old >= 2 && old != tag).
//   - Refuses to overwrite slots currently being written (old == 1) OR slots pinned by a cache hit
//     or already written in this same wave (old == tag), guaranteeing 100% race-free reads in FINISH.
__device__ __forceinline__ bool cache_try_acquire(uint32_t* locks, uint32_t slot, uint32_t tag) {
  const uint32_t old = locks[slot];
  if (old == 1u || old == tag) return false;
  return atomicCAS(locks + slot, old, 1u) == old;
}

}  // namespace scacchi_net
