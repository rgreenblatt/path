#include "execution_model/host_device_vector.h"
#include "lib/assert.h"
#include "lib/cuda/reduce.cuh"
#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "work_division/work_division.h"
#include "work_division/work_division_impl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>

using work_division::WorkDivision;

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

template <typename T>
__global__ void
division_sum_samples(const WorkDivision division, Span<const T> in, Span<T> out,
                     unsigned num_locations, unsigned samples_per) {
  unsigned thread_idx = threadIdx.x;
  unsigned block_idx = blockIdx.x;

  always_assert(blockDim.x == division.block_size());
  always_assert(division.num_sample_blocks() == 1);

  auto [start_sample, end_sample, j, unused] =
      division.get_thread_info(block_idx, thread_idx);

  if (j >= num_locations) {
    return;
  }

  T total = 0;
  for (unsigned i = start_sample; i < end_sample; ++i) {
    total += in[i + j * samples_per];
  }

  auto add = [](auto lhs, auto rhs) { return lhs + rhs; };
  total = division.reduce_samples(total, add, thread_idx);
  if (division.assign_sample(thread_idx)) {
    out[j] = total;
  }
}
#define EXPECT_FLOATS_EQ(expected, actual)                                     \
  EXPECT_EQ(expected.size(), actual.size()) << "Sizes differ.";                \
  for (size_t idx = 0; idx < std::min(expected.size(), actual.size());         \
       ++idx) {                                                                \
    EXPECT_FLOAT_EQ(expected[idx], actual[idx])                                \
        << "at index: " << idx << " and line: " << __LINE__;                   \
  }

TEST(Reduce, sum) {
  auto run_test = [](auto dist, auto check_equality) {
    for (unsigned num_locations : {1, 2, 3, 7, 8, 17, 32, 256}) {
      for (unsigned samples_per : {1, 2, 3, 7, 8, 32, 37, 49, 128, 189, 256})
        for (unsigned block_size : {32, 128, 256, 1024}) {
          for (unsigned base_target_samples_per_thread : {1, 2, 3, 5}) {
            const unsigned size = num_locations * samples_per;

            // avoid this test taking too long
            if (size > 4096) {
              continue;
            }

            const unsigned target_x_block_size = block_size;
            const unsigned target_y_block_size = 1;
            // const unsigned max_samples_per_thread = 16;
            unsigned target_samples_per_thread = base_target_samples_per_thread;
            WorkDivision division;
            do {
              division =
                  WorkDivision({block_size, target_x_block_size,
                                target_y_block_size, target_samples_per_thread},
                               samples_per, size, 1);
              target_samples_per_thread *= 2;
            } while (division.num_sample_blocks() != 1);

            std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

            using T = std::decay_t<decltype(dist(gen))>;

            HostDeviceVector<T> vals(size);

            std::generate(vals.begin(), vals.end(),
                          [&]() { return dist(gen); });

            HostDeviceVector<T> out_vals(num_locations);

            bool use_direct_approach =
                block_size % samples_per == 0 && size % block_size == 0;
            if (use_direct_approach) {
              unsigned num_blocks = size / block_size;
              always_assert(num_blocks * block_size == size);
              sum_sub_blocks<T>
                  <<<num_blocks, block_size>>>(vals, out_vals, samples_per);
            }

            HostDeviceVector<T> out_vals_division(num_locations);

            division_sum_samples<T>
                <<<division.total_num_blocks(), block_size>>>(
                    division, vals, out_vals_division, num_locations,
                    samples_per);

            CUDA_ERROR_CHK(cudaDeviceSynchronize());

            std::vector<T> expected(num_locations, 0.f);

            for (unsigned location = 0; location < num_locations; ++location) {
              for (unsigned i = location * samples_per;
                   i < (location + 1) * samples_per; ++i) {
                expected[location] += vals[i];
              }
            }

            if (use_direct_approach) {
              check_equality(expected, out_vals);
            }
            check_equality(expected, out_vals_division);
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
