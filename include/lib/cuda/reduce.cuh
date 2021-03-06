#pragma once

#include "cub/util_ptx.cuh"
#include "lib/assert.h"
#include "lib/bit_utils.h"
#include "lib/cuda/utils.h"
#include "lib/reducible_bin_op.h"
#include "meta/trivial.h"

#include <algorithm>
#include <cstdint>

constexpr uint32_t full_mask = 0xffffffff;

template <Trivial T>
inline __device__ T shuffle_down(T input, unsigned member_mask, unsigned offset,
                                 unsigned width) {
  typedef typename cub::UnitWord<T>::ShuffleWord ShuffleWord;
  static_assert(sizeof(T) % sizeof(ShuffleWord) == 0);
  constexpr unsigned words = sizeof(T) / sizeof(ShuffleWord);

  T output;
  ShuffleWord *output_alias = reinterpret_cast<ShuffleWord *>(&output);
  const ShuffleWord *input_alias =
      reinterpret_cast<const ShuffleWord *>(&input);

#pragma unroll
  for (unsigned work = 0; work < words; ++work) {
    output_alias[work] = __shfl_down_sync(
        member_mask, (unsigned)input_alias[work], offset, width);
  }

  return output;
}

template <Trivial T, BinOp<T> F>
inline __device__ T warp_reduce(T val, const F &f,
                                unsigned sub_warp_size = warp_size) {
  debug_assert_assume(warp_size % sub_warp_size == 0);
  debug_assert_assume(power_of_2(sub_warp_size));
  // equivalent to above, the compiler isn't quite smart enough to realize...
  debug_assert_assume(sub_warp_size <= warp_size);

  for (unsigned offset = sub_warp_size / 2; offset > 0; offset /= 2) {
    val = f(shuffle_down(val, full_mask, offset, sub_warp_size), val);
  }

  return val;
}

template <Trivial T, BinOp<T> F>
inline __device__ T sub_block_reduce(T val, const F &f, unsigned thread_idx,
                                     unsigned block_size,
                                     unsigned sub_block_size) {
  debug_assert_assume(thread_idx < block_size);
  debug_assert_assume(block_size % sub_block_size == 0);
  debug_assert_assume(block_size % warp_size == 0);
  debug_assert_assume(sub_block_size <= block_size);
  debug_assert_assume(warp_size <= block_size);
  debug_assert_assume(block_size != 0);
  debug_assert_assume(sub_block_size != 0);
  debug_assert_assume(block_size <= max_num_warps_per_block * warp_size);

  // Each warp performs partial reduction
  val = warp_reduce(val, f, std::min(sub_block_size, warp_size));

  if (sub_block_size <= warp_size) {
    return val;
  }

  // Shared mem for partially reduced values
  static __shared__ T shared[max_num_warps_per_block];

  unsigned lane = thread_idx % warp_size;
  unsigned warp_idx = thread_idx / warp_size;

  debug_assert_assume(block_size / warp_size <= max_num_warps_per_block);
  debug_assert_assume(warp_idx < max_num_warps_per_block);

  if (lane == 0) {
    shared[warp_idx] = val; // Write reduced value to shared memory
  }

  // TODO: why language server error here?
  __syncthreads(); // Wait for all partial reductions

  unsigned n_warps_per_sub_group = sub_block_size / warp_size;
  unsigned sub_block_thread_idx = thread_idx % sub_block_size;
  unsigned sub_block_idx = thread_idx / sub_block_size;

  // read from shared memory only if that warp existed
  if (sub_block_thread_idx < n_warps_per_sub_group) {
    val = shared[lane + sub_block_idx * n_warps_per_sub_group];
  }

  // Final reduce within first warp of each subgroup
  // we only reduce the values of n_warps_per_sub_group
  // so other values in greater lanes in the warp don't matter...
  val = warp_reduce(val, f, n_warps_per_sub_group);

  return val;
}

// it's plausible the compile won't be able to optimize this function
// to be as efficient as possible because sub_block_reduce is more general :(
template <Trivial T, BinOp<T> F>
inline __device__ T block_reduce(const T &val, const F &f, unsigned thread_idx,
                                 unsigned block_size) {
  return sub_block_reduce(val, f, thread_idx, block_size, block_size);
}
