#include "data_structure/bitset_ref.h"
#include "execution_model/host_device_vector.h"
#include "lib/span.h"

#include <gtest/gtest.h>

#include <random>

TEST(Bitset, num_bits_set_inclusive_up_to) {
  std::vector<unsigned> values = {0b100001111100001111001111u, 0u, 0b11111111u,
                                  0b111100000000000000000001111111u,
                                  0b11111111111111111111111111111111u};
  BitsetRef<unsigned> bit_set(values);

  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(0, 0), 1u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(0, 8), 7u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(0, 20), 13u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(1, 8), 0u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(1, 31), 0u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(4, 16), 17u);
  EXPECT_EQ(bit_set.num_bits_set_inclusive_up_to(4, 31), 32u);
}

TEST(Bitset, find_mask_same) {
  std::vector<unsigned> values = {0b100001111100001111001111u, 0u, 0b11111111u,
                                  0b111100000000000000000001111111u,
                                  0b11111111111111111111111111111111u};
  BitsetRef<unsigned> bit_set(values);

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
