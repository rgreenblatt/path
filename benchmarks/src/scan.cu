#include "data_structure/bitset_ref.h"
#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "lib/span.h"

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <string>

template <typename T> static void standard(benchmark::State &state) {
  ThrustData<ExecutionModel::GPU> thrust_data;
  DeviceVector<T> in(unsigned(state.range(0)));
  DeviceVector<uint32_t> out(unsigned(state.range(0)));

  for (auto _ : state) {
    thrust::exclusive_scan(thrust_data.execution_policy(), in.begin(), in.end(),
                           out.begin());
  }
}

template <typename TVals, typename TKeys>
static void standard_keys(benchmark::State &state) {
  ThrustData<ExecutionModel::GPU> thrust_data;
  DeviceVector<TVals> in(unsigned(state.range(0)));
  DeviceVector<TKeys> keys(unsigned(state.range(0)));
  DeviceVector<uint32_t> out(unsigned(state.range(0)));

  for (auto _ : state) {
    thrust::exclusive_scan_by_key(thrust_data.execution_policy(), keys.begin(),
                                  keys.end(), in.begin(), out.begin());
  }
}

template <std::integral Block> static void bitset_direct(benchmark::State &state) {
  ThrustData<ExecutionModel::GPU> thrust_data;
  uint32_t size_per = BitSetRef<Block>::bits_per_block;
  DeviceVector<Block> in(unsigned(state.range(0)) / size_per);
  DeviceVector<uint32_t> out(unsigned(state.range(0)));

  BitSetRef<Block> bit_set(in, unsigned(state.range(0)));

  auto start_bit_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u),
      [bit_set] __host__ __device__(unsigned pos) { return bit_set[pos]; });
  auto end_bit_iter = start_bit_iter + unsigned(state.range(0));

  for (auto _ : state) {
    thrust::exclusive_scan(thrust_data.execution_policy(), start_bit_iter,
                           end_bit_iter, out.begin());
  }
}

template <std::integral Block> static void bitset_popcount(benchmark::State &state) {
  ThrustData<ExecutionModel::GPU> thrust_data;
  unsigned size_per = BitSetRef<Block>::bits_per_block;
  unsigned num_blocks = unsigned(state.range(0)) / size_per;
  DeviceVector<Block> in(num_blocks);
  DeviceVector<uint32_t> out(num_blocks);

  BitSetRef<Block> bit_set(in, unsigned(state.range(0)));

  auto start_block_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u),
      [bit_set] __host__ __device__(unsigned block) {
        return bit_set.count(block);
      });
  auto end_block_iter = start_block_iter + num_blocks;

  for (auto _ : state) {
    thrust::exclusive_scan(thrust_data.execution_policy(), start_block_iter,
                           end_block_iter, out.begin());
  }
}

template <std::integral Block>
static void bitset_direct_keys(benchmark::State &state) {
  ThrustData<ExecutionModel::GPU> thrust_data;
  unsigned size_per = BitSetRef<Block>::bits_per_block;
  DeviceVector<Block> in(unsigned(state.range(0)) / size_per);
  DeviceVector<Block> keys(unsigned(state.range(0)) / size_per);
  DeviceVector<uint32_t> out(unsigned(state.range(0)));

  BitSetRef<Block> bit_set_in(in, unsigned(state.range(0)));
  BitSetRef<Block> bit_set_keys(keys, unsigned(state.range(0)));

  auto start_key_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u),
      [bit_set_keys] __host__ __device__(unsigned pos) {
        return bit_set_keys[pos];
      });
  auto end_key_iter = start_key_iter + unsigned(state.range(0));

  auto start_in_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u),
      [bit_set_in] __host__ __device__(unsigned pos) {
        return bit_set_in[pos];
      });

  for (auto _ : state) {
    thrust::exclusive_scan_by_key(thrust_data.execution_policy(),
                                  start_key_iter, end_key_iter, start_in_iter,
                                  out.begin());
  }
}

template <std::integral Block>
static void bitset_popcount_keys(benchmark::State &state) {
  ThrustData<ExecutionModel::GPU> thrust_data;
  unsigned size_per = BitSetRef<Block>::bits_per_block;
  DeviceVector<Block> in(unsigned(state.range(0)) / size_per);
  DeviceVector<Block> keys(unsigned(state.range(0)) / size_per);
  DeviceVector<uint32_t> keys_periodic(unsigned(state.range(0)) / size_per);
  DeviceVector<uint32_t> out(unsigned(state.range(0)));

  BitSetRef<Block> bit_set_in(in, unsigned(state.range(0)));
  BitSetRef<Block> bit_set_keys(keys, unsigned(state.range(0)));

  auto start_block_in_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0u),
      [bit_set_in, bit_set_keys] __host__ __device__(unsigned block) {
        return bit_set_in.masked_count(block,
                                       bit_set_keys.find_mask_block_end(block));
      });

  for (auto _ : state) {
    thrust::exclusive_scan_by_key(thrust_data.execution_policy(),
                                  keys_periodic.begin(), keys_periodic.end(),
                                  start_block_in_iter, out.begin());
  }
}

constexpr uint64_t s_range = 1 << 16;
constexpr uint64_t e_range = 1 << 24;

BENCHMARK_TEMPLATE(standard, uint64_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(standard, uint32_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(standard, uint8_t)->Range(s_range, e_range);

BENCHMARK_TEMPLATE(standard_keys, uint64_t, uint64_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(standard_keys, uint32_t, uint32_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(standard_keys, uint32_t, uint8_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(standard_keys, uint8_t, uint8_t)->Range(s_range, e_range);

BENCHMARK_TEMPLATE(bitset_direct, uint32_t)->Range(s_range, e_range);
#if 0
BENCHMARK_TEMPLATE(bitset_direct, uint16_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(bitset_direct, uint8_t)->Range(s_range, e_range);
#endif

BENCHMARK_TEMPLATE(bitset_popcount, uint64_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(bitset_popcount, uint32_t)->Range(s_range, e_range);
#if 0
BENCHMARK_TEMPLATE(bitset_popcount, uint16_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(bitset_popcount, uint8_t)->Range(s_range, e_range);
#endif

BENCHMARK_TEMPLATE(bitset_direct_keys, uint32_t)->Range(s_range, e_range);
#if 0
BENCHMARK_TEMPLATE(bitset_direct_keys, uint16_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(bitset_direct_keys, uint8_t)->Range(s_range, e_range);
#endif

BENCHMARK_TEMPLATE(bitset_popcount_keys, uint64_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(bitset_popcount_keys, uint32_t)->Range(s_range, e_range);
#if 0
// some changes are required to count_leading_zeros to get these to compile
// and be correct
BENCHMARK_TEMPLATE(bitset_popcount_keys, uint16_t)->Range(s_range, e_range);
BENCHMARK_TEMPLATE(bitset_popcount_keys, uint8_t)->Range(s_range, e_range);
#endif
