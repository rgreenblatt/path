#pragma once

#include "lib/cuda/utils.h"

#include <limits>
#include <concepts>

inline HOST_DEVICE uint32_t popcount(uint32_t v) {
  static_assert(sizeof(uint32_t) == sizeof(unsigned));
  static_assert(sizeof(uint32_t) == sizeof(int));
#ifdef __CUDA_ARCH__
  // this type is technically int (signed), but this works on all uint32_t (see
  // tests)
  return __popc(v);
#else
  return __builtin_popcount(v);
#endif
}

inline HOST_DEVICE uint64_t popcount(uint64_t v) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
  static_assert(sizeof(uint64_t) == sizeof(long long int));
#ifdef __CUDA_ARCH__
  // this type is technically long long int (signed), but this works on all
  // uint64_t (see tests)
  return __popcll(v);
#else
  return __builtin_popcountll(v);
#endif
}

inline HOST_DEVICE unsigned count_leading_zeros(uint32_t v) {
  static_assert(sizeof(uint32_t) == sizeof(unsigned));
#ifdef __CUDA_ARCH__
  return __clz(v);
#else
  return __builtin_clz(v);
#endif
}

inline HOST_DEVICE unsigned count_leading_zeros(uint64_t v) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
#ifdef __CUDA_ARCH__
  return __clzll(v);
#else
  return __builtin_clzll(v);
#endif
}

template <std::integral T>
static constexpr unsigned bits_per = sizeof(T) * CHAR_BIT;

template <std::integral T> constexpr inline T bit_mask(unsigned bit_idx) {
  return T(1) << bit_idx;
}

template <std::integral T> constexpr inline T up_to_mask(unsigned n) {
  // in binary, 1 (n + 1) times
  // 0: 1
  // 1: 11
  // ...

  // ternary statement required to avoid undefined behavior
  return n >= bits_per<T> - 1 ? std::numeric_limits<T>::max()
                              : bit_mask<T>(n + T(1)) - T(1);
}
