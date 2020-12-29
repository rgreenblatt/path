#include "data_structure/copyable_to_vec.h"
#include "execution_model/vector_type.h"
#include "lib/assert.h"
#include "lib/cuda/reduce.cuh"
#include "lib/span.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>

template <typename T>
__global__ void sum_sub_blocks(Span<const T> in, Span<T> out,
                               unsigned sub_block_size) {
  unsigned thread_idx = threadIdx.x;
  unsigned block_idx = blockIdx.x;
  unsigned block_size = blockDim.x;
  unsigned overall_idx = thread_idx + block_idx * block_size;
  unsigned sub_block_idx = overall_idx / sub_block_size;
  auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
  const T total = sub_block_reduce<T>(in[overall_idx], add, thread_idx,
                                      block_size, sub_block_size);
  if (thread_idx % sub_block_size == 0) {
    out[sub_block_idx] = total;
  }
}

#define EXPECT_FLOATS_EQ(expected, actual)                                     \
  EXPECT_EQ(expected.size(), actual.size()) << "Sizes differ.";                \
  for (size_t idx = 0; idx < std::min(expected.size(), actual.size());         \
       ++idx) {                                                                \
    EXPECT_FLOAT_EQ(expected[idx], actual[idx]) << "at index: " << idx;        \
  }

TEST(Reduce, sum) {
  auto run_test = [](auto dist, auto check_equality) {
    for (unsigned n_blocks : {1, 2, 3, 7, 17}) {
      for (unsigned block_size : {32, 128, 256, 1024}) {
        for (unsigned sub_block_size = 1; sub_block_size <= block_size;
             sub_block_size *= 2) {
          const unsigned size = n_blocks * block_size;
          std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

          using T = std::decay_t<decltype(dist(gen))>;

          HostVector<T> vals(size);

          std::generate(vals.begin(), vals.end(), [&]() { return dist(gen); });

          DeviceVector<T> gpu_vals;
          copy_to_vec(vals, gpu_vals);

          unsigned num_sub_blocks = size / sub_block_size;

          DeviceVector<T> out_gpu_vals(num_sub_blocks);

          sum_sub_blocks<T><<<n_blocks, block_size>>>(gpu_vals, out_gpu_vals,
                                                      sub_block_size);

          std::vector<T> expected(num_sub_blocks, 0.f);

          for (unsigned sub_block = 0; sub_block < num_sub_blocks;
               ++sub_block) {
            for (unsigned i = sub_block * sub_block_size;
                 i < (sub_block + 1) * sub_block_size; ++i) {
              expected[sub_block] += vals[i];
            }
          }

          std::vector<T> actual;

          copy_to_vec(out_gpu_vals, actual);

          check_equality(expected, actual);
        }
      }
    }
  };

  run_test(std::uniform_real_distribution<double>(0.0, 1.0),
           [](const auto &expected, const auto &actual) {
             EXPECT_FLOATS_EQ(expected, actual);
           });
  run_test(std::uniform_int_distribution<int>(-100, 100),
           [](const auto &expected, const auto &actual) {
             EXPECT_THAT(expected, testing::ElementsAreArray(actual));
           });
}
