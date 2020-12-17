#include "data_structure/copyable.h"
#include "execution_model/vector_type.h"
#include "lib/cuda/reduce.cuh"
#include "lib/span.h"

#include <Eigen/Dense>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <random>

template <typename T>
__global__ void sum_blocks(Span<const T> in, Span<T> out) {
  const unsigned idx = threadIdx.x;
  const unsigned block = blockIdx.x;
  const unsigned block_size = blockDim.x;
  auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
  const T total =
      block_reduce<T>(in[idx + block * block_size], add, 0, idx, block_size);
  if (idx == 0) {
    out[block] = total;
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
    for (unsigned n_blocks : {1, 2, 4, 7, 16}) {
      const unsigned block_size = 256;
      const unsigned size = n_blocks * block_size;
      assert(size % block_size == 0);

      std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

      using T = std::decay_t<decltype(dist(gen))>;

      HostVector<T> vals(size);

      std::generate(vals.begin(), vals.end(), [&]() { return dist(gen); });

      DeviceVector<T> gpu_vals;
      copy_to(vals, gpu_vals);

      DeviceVector<T> out_gpu_vals(n_blocks);

      sum_blocks<T><<<n_blocks, block_size>>>(gpu_vals, out_gpu_vals);

      std::vector<T> expected(n_blocks, 0.f);

      for (unsigned block = 0; block < n_blocks; ++block) {
        for (unsigned i = block * block_size; i < (block + 1) * block_size;
             ++i) {
          expected[block] += vals[i];
        }
      }

      std::vector<T> actual;

      copy_to(out_gpu_vals, actual);

      check_equality(expected, actual);
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
