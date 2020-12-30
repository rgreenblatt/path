#include "execution_model/host_device_vector.h"
#include "lib/bit_utils.h"
#include "lib/span.h"

#include <gtest/gtest.h>

#include <random>

// These tests mostly exist to verify that the gpu functions
// work as expected. Basic unit testing is included in bit_utils.h
// in the form of static_assert
template <typename T> static void popcount_test(std::mt19937 &gen) {
  const unsigned size = 1000;
  HostDeviceVector<T> values(size);
  std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
  std::generate(values.begin(), values.end(), [&] { return dis(gen); });

  values[0] = 0b10100111u;
  values[1] = std::numeric_limits<T>::max();

  HostDeviceVector<T> gpu_out(size);
  HostDeviceVector<T> cpu_out(size);

  auto transform_values = [&](const auto type, HostDeviceVector<T> &out) {
    thrust::transform(type, values.data(), values.data() + values.size(),
                      out.data(),
                      [] __host__ __device__(T v) { return popcount(v); });
  };

  transform_values(thrust::host, cpu_out);
  transform_values(thrust::device, gpu_out);

  for (unsigned i = 0; i < size; i++) {
    EXPECT_EQ(gpu_out[i], cpu_out[i]);
  }

  EXPECT_EQ(gpu_out[0], T(5));
  EXPECT_EQ(cpu_out[0], T(5));
  T max_value_bits = sizeof(T) * CHAR_BIT;
  EXPECT_EQ(gpu_out[1], max_value_bits);
  EXPECT_EQ(cpu_out[1], max_value_bits);
}

TEST(Bitset, popcount) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

  popcount_test<uint8_t>(gen);
  popcount_test<uint16_t>(gen);
  popcount_test<uint32_t>(gen);
  popcount_test<uint64_t>(gen);
}

template <typename T> static void count_leading_zeros_test(std::mt19937 &gen) {
  const unsigned size = 1000;
  HostDeviceVector<T> values(size);
  std::uniform_int_distribution<T> dis(0u, std::numeric_limits<T>::max());
  std::uniform_int_distribution<T> dis_mask(0u, 31);
  std::generate(values.begin(), values.end(),
                [&] { return (dis(gen) & ((1u << dis_mask(gen)) - 1)) | 1u; });

  HostDeviceVector<T> gpu_out(size);
  HostDeviceVector<T> cpu_out(size);

  auto transform_values = [&](const auto type, HostDeviceVector<T> &out) {
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

TEST(Bitset, count_leading_zeros) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

  count_leading_zeros_test<uint8_t>(gen);
  count_leading_zeros_test<uint16_t>(gen);
  count_leading_zeros_test<uint32_t>(gen);
  count_leading_zeros_test<uint64_t>(gen);
}
