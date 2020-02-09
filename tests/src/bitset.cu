#include "lib/bitset.h"
#include "lib/cuda/managed_mem_vec.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"

#include <gtest/gtest.h>

#include <random>

template <typename T> static void popcount_test(std::mt19937 &gen) {
  const unsigned size = 1000;
  ManangedMemVec<T> values(size);
  std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
  std::generate(values.begin(), values.end(), [&] { return dis(gen); });

  values[0] = 0b100001111100001111001111u;
  values[1] = std::numeric_limits<T>::max();

  ManangedMemVec<T> gpu_out(size);
  ManangedMemVec<T> cpu_out(size);

  auto transform_values = [&](const auto type, ManangedMemVec<T> &out) {
    thrust::transform(type, values.data(), values.data() + values.size(),
                      out.data(),
                      [] __host__ __device__(T v) { return popcount(v); });
  };

  transform_values(thrust::host, cpu_out);
  transform_values(thrust::device, gpu_out);

  for (unsigned i = 0; i < size; i++) {
    EXPECT_EQ(gpu_out[i], cpu_out[i]);
  }

  EXPECT_EQ(gpu_out[0], T(14));
  EXPECT_EQ(cpu_out[0], T(14));
  T max_value_bits = sizeof(T) * CHAR_BIT;
  EXPECT_EQ(gpu_out[1], max_value_bits);
  EXPECT_EQ(cpu_out[1], max_value_bits);
}

TEST(BitSet, popcount) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

  popcount_test<uint32_t>(gen);
  popcount_test<uint64_t>(gen);
}

template <typename T> static void count_leading_zeros_test(std::mt19937 &gen) {
  const unsigned size = 1000;
  ManangedMemVec<T> values(size);
  std::uniform_int_distribution<T> dis(0u, std::numeric_limits<T>::max());
  std::uniform_int_distribution<T> dis_mask(0u, 31);
  std::generate(values.begin(), values.end(),
                [&] { return (dis(gen) & ((1u << dis_mask(gen)) - 1)) | 1u; });

  ManangedMemVec<T> gpu_out(size);
  ManangedMemVec<T> cpu_out(size);

  auto transform_values = [&](const auto type, ManangedMemVec<T> &out) {
    thrust::transform(
        type, values.data(), values.data() + values.size(), out.data(),
        [] __host__ __device__(T v) { return count_leading_zeros(v); });
  };

  transform_values(thrust::host, cpu_out);
  transform_values(thrust::device, gpu_out);

  for (unsigned i = 0; i < size; i++) {
    EXPECT_EQ(gpu_out[i], cpu_out[i]);
  }
}

TEST(BitSet, count_leading_zeros) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

  count_leading_zeros_test<uint32_t>(gen);
  count_leading_zeros_test<uint64_t>(gen);
}

TEST(BitSet, up_to_mask) {
  EXPECT_EQ(BitSetRef<unsigned>::up_to_mask(0), 0b1u);
  EXPECT_EQ(BitSetRef<unsigned>::up_to_mask(3), 0b1111u);
  EXPECT_EQ(BitSetRef<unsigned>::up_to_mask(31),
            0b11111111111111111111111111111111u);
}

TEST(BitSet, num_bits_set_inclusive_up_to) {
  std::vector<unsigned> values = {0b100001111100001111001111u, 0u, 0b11111111u,
                                  0b111100000000000000000001111111u,
                                  0b11111111111111111111111111111111u};
  BitSetRef<unsigned> bit_set(values, 5 * 32);

  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(0, 0), 1u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(0, 8), 7u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(0, 20), 13u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(1, 8), 0u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(1, 31), 0u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(4, 16), 17u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(4, 31), 32u);
}

TEST(BitSet, find_mask_same) {
  std::vector<unsigned> values = {0b100001111100001111001111u, 0u, 0b11111111u,
                                  0b111100000000000000000001111111u,
                                  0b11111111111111111111111111111111u};
  BitSetRef<unsigned> bit_set(values, 5 * 32);

  EXPECT_EQ(bit_set.find_mask_same(0, 0), 1u);
  EXPECT_EQ(bit_set.find_mask_same(0, 8), 0b111000000u);
  EXPECT_EQ(bit_set.find_mask_same(0, 20), 0b110000000000000000000u);
  EXPECT_EQ(bit_set.find_mask_same(1, 8), 0b111111111u);
  EXPECT_EQ(bit_set.find_mask_same(1, 31), 0b11111111111111111111111111111111u);
  EXPECT_EQ(bit_set.find_mask_same(1, 31), 0b11111111111111111111111111111111u);
  EXPECT_EQ(bit_set.find_mask_block_end(1),
            0b11111111111111111111111111111111u);
  EXPECT_EQ(bit_set.find_mask_same(4, 16), 0b11111111111111111u);
  EXPECT_EQ(bit_set.find_mask_same(4, 31), 0b11111111111111111111111111111111u);
  EXPECT_EQ(bit_set.find_mask_block_end(4),
            0b11111111111111111111111111111111u);
}
