#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <cstdint>
#include <limits>

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

template <typename Block> class BitSetRef {
public:
  BitSetRef(const Span<Block> &data, unsigned num_bits)
      : data_(data), num_bits_(num_bits) {}

  static constexpr unsigned bits_per_block = sizeof(Block) * CHAR_BIT;

  // at some point it may be worthwhile to allow for assigning bits...
  // note potential issues with parallelism here though...
  HOST_DEVICE void set_block(unsigned block_idx, Block new_value) {
    data_[block_idx] = new_value;
  }

  HOST_DEVICE bool operator[](unsigned pos) const { return test(pos); }

  HOST_DEVICE bool test(unsigned block_idx, unsigned bit_idx) const {
    return (data_[block_idx] & bit_mask(bit_idx)) != 0;
  }

  HOST_DEVICE bool test(unsigned pos) const {
    return test(block_index(pos), bit_index(pos));
  }

  HOST_DEVICE unsigned count(unsigned block_idx) const {
    return popcount(data_[block_idx]);
  }

  HOST_DEVICE unsigned masked_count(unsigned block_idx, Block mask) const {
    return popcount(data_[block_idx] & mask);
  }

  HOST_DEVICE unsigned num_bits_set_inclusive_up_to(unsigned block_idx,
                                                    unsigned bit_idx) {
    return masked_count(block_idx, up_to_mask(bit_idx));
  }

  HOST_DEVICE unsigned num_bits_set_inclusive_up_to(unsigned pos) {
    return num_bits_set_inclusive_up_to(block_index(pos), bit_index(pos));
  }

  HOST_DEVICE Block find_mask_same(unsigned block_idx, unsigned bit_idx) const {
    Block value = data_[block_idx];

    // make most significant set bit the first difference
    if (test(block_idx, bit_idx)) {
      value = ~value;
    }

    Block up_to = up_to_mask(bit_idx);

    Block masked_value = value & up_to;

    // if masked_value is zero, all bits are the same
    // also avoids undefined or implementation defined behavior
    if (masked_value == 0) {
      return up_to;
    }

    // if clz is just a loop, it might be faster to just
    // loop over bits and report first difference...
    unsigned start_same =
        (bits_per_block - 1) - count_leading_zeros(masked_value);

    Block eliminate_start = ~up_to_mask(start_same);

    return up_to & eliminate_start;
  }

  HOST_DEVICE Block find_mask_same(unsigned pos) const {
    return find_mask_same(block_index(pos), bit_index(pos));
  }

  HOST_DEVICE Block find_mask_block_end(unsigned block_idx) const {
    return find_mask_same(block_idx, bits_per_block - 1);
  }

  HOST_DEVICE static unsigned block_index(unsigned pos) {
    return pos / bits_per_block;
  }

  HOST_DEVICE static unsigned bit_index(unsigned pos) {
    return pos % bits_per_block;
  }

  HOST_DEVICE static Block bit_mask(unsigned bit_idx) {
    return Block(1) << bit_idx;
  }

  HOST_DEVICE static Block up_to_mask(unsigned n) {
    // in binary, 1 (n + 1) times
    // 0: 1
    // 1: 11
    // ...

    // ternary statement required to avoid undefined behavior
    return n >= bits_per_block - 1 ? std::numeric_limits<Block>::max()
                                   : bit_mask(n + Block(1)) - Block(1);
  }

private:
  Span<Block> data_;
  unsigned num_bits_;
};
