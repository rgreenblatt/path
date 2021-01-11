#ifndef CPU_ONLY
// must be included first to avoid issues with cub...
#include "lib/cuda/utils.h"

#include "execution_model/device_vector.h"
#include "execution_model/thrust_data.h"
#include "lib/span.h"

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <string>

template <typename T> static void standard(benchmark::State &state) {
  ThrustData<ExecutionModel::GPU> thrust_data;
  DeviceVector<T> vals(unsigned(state.range(0)));

  for (auto _ : state) {
    thrust::sort(thrust_data.execution_policy(), vals.begin(), vals.end());
  }
}

template <typename T> static void reverse_order(benchmark::State &state) {
  ThrustData<ExecutionModel::GPU> thrust_data;
  DeviceVector<T> vals(unsigned(state.range(0)));

  for (auto _ : state) {
    thrust::sort(thrust_data.execution_policy(), vals.begin(), vals.end(),
                 thrust::greater<T>());
  }
}

constexpr uint64_t s_range = 1 << 16;
constexpr uint64_t e_range = 1 << 24;

BENCHMARK_TEMPLATE(standard, uint64_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(standard, uint32_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(standard, uint8_t)->Range(s_range, e_range);

BENCHMARK_TEMPLATE(reverse_order, uint64_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(reverse_order, uint32_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(reverse_order, uint8_t)->Range(s_range, e_range);
#endif
