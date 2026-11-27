// Shared GPU position-cache key and hash for search.cu and conv_board_int8.cuh.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace scacchi_net {

constexpr int CACHE_SLOTS = 65536;  // power of two
constexpr uint32_t CACHE_MASK = CACHE_SLOTS - 1;

struct CacheKey {
  uint64_t my_lo, my_hi, op_lo, op_hi;
  uint32_t meta;  // bit 0: color != 0, bit 1: step == 1, bit 31: occupied
  uint32_t pad;
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

}  // namespace scacchi_net
