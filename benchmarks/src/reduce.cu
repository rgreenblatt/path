#ifndef CPU_ONLY
#include "data_structure/copyable_to_vec.h"
#include "execution_model/device_vector.h"
#include "execution_model/host_vector.h"
#include "kernel/kernel_launch.h"
#include "kernel/kernel_launch_impl_gpu.cuh"
#include "kernel/runtime_constants_reducer.h"
#include "kernel/runtime_constants_reducer_impl_gpu.cuh"
#include "kernel/thread_interactor_launchable.h"
#include "kernel/work_division.h"
#include "lib/bit_utils.h"
#include "lib/cuda/reduce.cuh"
#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "lib/tagged_tuple.h"
#include "lib/tagged_union.h"
#include "meta/all_values.h"
#include "meta/all_values_as_tuple.h"
#include "meta/all_values_enum.h"
#include "meta/all_values_integral.h"
#include "meta/all_values_pow_2.h"
#include "meta/as_tuple_macro.h"
#include "meta/dispatch.h"
#include "meta/tuple.h"

#include <benchmark/benchmark.h>
#include <cub/block/block_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <sstream>

enum class FullyDispatchedReduceImpl {
  MineGeneric,
  CUBGeneric,
  CUBSum,
};

constexpr auto sum = [](const auto &l, const auto &r) { return l + r; };

template <FullyDispatchedReduceImpl type, unsigned block_size,
          unsigned items_per_thread, typename T>
__global__ void block_reduce_fully_dispatched(Span<const T> in, Span<T> out) {
  // block load doesn't seem to ever work...
  T thread_data[items_per_thread];
#pragma unroll
  for (unsigned i = 0; i < items_per_thread; ++i) {
    thread_data[i] =
        in[i + items_per_thread * (threadIdx.x + blockIdx.x * blockDim.x)];
  }

  using BlockReduce = cub::BlockReduce<T, block_size>;
  const T item = [&] {
    if constexpr (type == FullyDispatchedReduceImpl::MineGeneric) {
      T value = 0;
#pragma unroll
      for (unsigned i = 0; i < items_per_thread; ++i) {
        value += thread_data[i];
      }
      return block_reduce(value, sum, threadIdx.x, block_size);
    } else if constexpr (type == FullyDispatchedReduceImpl::CUBGeneric) {
      __shared__ typename BlockReduce::TempStorage reduce_storage;
      return BlockReduce(reduce_storage).Reduce(thread_data, sum);
    } else {
      static_assert(type == FullyDispatchedReduceImpl::CUBSum);
      __shared__ typename BlockReduce::TempStorage reduce_storage;
      return BlockReduce(reduce_storage).Sum(thread_data);
    }
  }();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = item;
  }
}

template <FullyDispatchedReduceImpl type, unsigned block_size,
          unsigned sub_warp_size, unsigned items_per_thread, typename T>
__global__ void warp_reduce_fully_dispatched(Span<const T> in, Span<T> out) {
  using WarpReduce = cub::WarpReduce<T, sub_warp_size>;
  T value = 0;
#pragma unroll
  for (unsigned i = 0; i < items_per_thread; ++i) {
    value += in[i + items_per_thread * (threadIdx.x + blockIdx.x * blockDim.x)];
  }

  const T item = [&] {
    if constexpr (type == FullyDispatchedReduceImpl::MineGeneric) {
      return warp_reduce(value, sum, sub_warp_size);
    } else if constexpr (type == FullyDispatchedReduceImpl::CUBGeneric) {
      __shared__ typename WarpReduce::TempStorage reduce_storage;
      return WarpReduce(reduce_storage).Reduce(value, sum);
    } else {
      static_assert(type == FullyDispatchedReduceImpl::CUBSum);
      __shared__ typename WarpReduce::TempStorage reduce_storage;
      return WarpReduce(reduce_storage).Sum(value);
    }
  }();
  if (threadIdx.x % sub_warp_size == 0) {
    constexpr unsigned sub_warps_per_block = block_size / sub_warp_size;
    out[threadIdx.x / sub_warp_size + blockIdx.x * sub_warps_per_block] = item;
  }
}

enum class NotFullyDispatchedReduceImpl {
  Mine,
  GeneralKernelLaunch,
};

template <NotFullyDispatchedReduceImpl type, typename T>
__global__ void block_reduce_not_fully_dispatched(Span<const T> in, Span<T> out,
                                                  unsigned block_size,
                                                  unsigned items_per_thread) {
  // no others for now...
  static_assert(type == NotFullyDispatchedReduceImpl::Mine);
  debug_assert(block_size == blockDim.x);
  debug_assert(threadIdx.x < block_size);
  debug_assert_assume(block_size % warp_size == 0);
  debug_assert_assume(power_of_2(block_size));
  debug_assert_assume(power_of_2(items_per_thread)); // somewhat atypical...
  T value = 0;
  // #pragma unroll
  for (unsigned i = 0; i < items_per_thread; ++i) {
    value += in[i + items_per_thread * (threadIdx.x + blockIdx.x * block_size)];
  }
  value = block_reduce(value, sum, threadIdx.x, block_size);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = value;
  }
}

template <NotFullyDispatchedReduceImpl type, typename T>
__global__ void warp_reduce_not_fully_dispatched(Span<const T> in, Span<T> out,
                                                 unsigned block_size,
                                                 unsigned sub_warp_size,
                                                 unsigned items_per_thread) {
  static_assert(type == NotFullyDispatchedReduceImpl::Mine);
  debug_assert(block_size == blockDim.x);
  debug_assert(threadIdx.x < block_size);
  debug_assert_assume(block_size % warp_size == 0);
  debug_assert_assume(warp_size % sub_warp_size == 0);
  debug_assert_assume(sub_warp_size <= warp_size);
  debug_assert_assume(power_of_2(block_size));
  debug_assert_assume(power_of_2(sub_warp_size));
  debug_assert_assume(power_of_2(items_per_thread)); // somewhat atypical...

  T value = 0;
  // #pragma unroll
  for (unsigned i = 0; i < items_per_thread; ++i) {
    value += in[i + items_per_thread * (threadIdx.x + blockIdx.x * block_size)];
  }

  value = warp_reduce(value, sum, sub_warp_size);
  if (threadIdx.x % sub_warp_size == 0) {
    unsigned sub_warps_per_block = block_size / sub_warp_size;
    out[threadIdx.x / sub_warp_size + blockIdx.x * sub_warps_per_block] = value;
  }
}

enum class ItemType {
  Float,
};

template <ItemType type> auto get_item_type() {
  if constexpr (type == ItemType::Float) {
    return float();
  }
}

template <ItemType type> auto generate_item_dist() {
  if constexpr (type == ItemType::Float) {
    return thrust::uniform_real_distribution(-100.f, 100.f);
  }
}

template <ItemType type> using GetItemType = decltype(get_item_type<type>());

struct BlockSize : public Pow2<32, 512> {
  using Pow2<32, 512>::Pow2Gen;
};

struct ItemsPerThread : public Pow2<1, 64> {
  using Pow2<1, 64>::Pow2Gen;
};

struct SubWarpSize : public Pow2<1, 32> {
  using Pow2<1, 32>::Pow2Gen;
};

template <> struct AllValuesImpl<BlockSize> {
  static constexpr auto values = std::array<BlockSize, 3>{32, 128, 512};
};

template <> struct AllValuesImpl<ItemsPerThread> {
  static constexpr auto values = std::array<ItemsPerThread, 4>{1, 8, 32, 64};
};

template <> struct AllValuesImpl<SubWarpSize> {
  static constexpr auto values = std::array<SubWarpSize, 3>{1, 8, 32};
};

static_assert(AllValuesEnumerable<BlockSize>);
static_assert(AllValuesEnumerable<ItemsPerThread>);
static_assert(AllValuesEnumerable<SubWarpSize>);

template <bool is_block> struct Constants {
  BlockSize block_size;
  std::conditional_t<is_block, EmptyEnumerable, SubWarpSize> sub_warp_size;
  ItemsPerThread items_per_thread;

  AS_TUPLE_STRUCTURAL(Constants, block_size, sub_warp_size, items_per_thread);

  constexpr auto operator<=>(const Constants &other) const = default;
};

template <bool is_fully_dispatched> struct KernelType {
  std::conditional_t<is_fully_dispatched, FullyDispatchedReduceImpl,
                     NotFullyDispatchedReduceImpl>
      reduce_impl;
  ItemType item_type;

  AS_TUPLE_STRUCTURAL(KernelType, reduce_impl, item_type);

  constexpr unsigned numeric_reduce_impl() const {
    auto out = static_cast<unsigned>(reduce_impl);
    if (!is_fully_dispatched) {
      out += magic_enum::enum_count<FullyDispatchedReduceImpl>();
    }
    return out;
  }
  constexpr unsigned numeric_item_type() const {
    return static_cast<unsigned>(item_type);
  }

  constexpr auto operator<=>(const KernelType &other) const = default;
};

struct RunType {
  bool is_block;
  bool is_fully_dispatched;

  AS_TUPLE_STRUCTURAL(RunType, is_block, is_fully_dispatched);

  constexpr auto operator<=>(const RunType &other) const = default;
};

template <RunType type> struct AllVars {
  Constants<type.is_block> constants;
  KernelType<type.is_fully_dispatched> kernel_type;

  AS_TUPLE_STRUCTURAL(AllVars, constants, kernel_type);

  constexpr auto operator<=>(const AllVars &other) const = default;
};

template <RunType type>
constexpr auto get_comp_time_params(AllVars<type> params) {
  if constexpr (type.is_fully_dispatched) {
    return params;
  } else {
    return params.kernel_type;
  }
}

using BenchmarkParams = TaggedUnionPerInstance<RunType, AllVars>;

template <ItemType type>
using HostVectorForType = HostVector<GetItemType<type>>;
template <ItemType type>
using DeviceVectorForType = DeviceVector<GetItemType<type>>;

int main(int argc, char *argv[]) {
  TaggedTuplePerInstance<ItemType, HostVectorForType> cpu_input_vec;
  TaggedTuplePerInstance<ItemType, HostVectorForType> cpu_output_vec;
  TaggedTuplePerInstance<ItemType, HostVectorForType> cpu_output_from_gpu_vec;
  TaggedTuplePerInstance<ItemType, DeviceVectorForType> input_vec;
  TaggedTuplePerInstance<ItemType, DeviceVectorForType> output_vec;

  auto sizes = [] {
    if constexpr (debug_build) {
      return std::array{1u << 12, 1u << 14, 1u << 18};
    } else {
      return std::array{1u << 12, 1u << 18, 1u << 20,
                        1u << 22, 1u << 24, 1u << 26};
    }
  }();
  const unsigned max_size = sizes[sizes.size() - 1];
  for (unsigned size : sizes) {
    for (auto params : AllValues<BenchmarkParams>) {
      params.visit_tagged([&](auto tag, auto params) {
        constexpr RunType run_type = decltype(tag)::value;

        dispatch(get_comp_time_params<run_type>(params), [&](auto tag) {
          constexpr auto type = decltype(tag)::value;
          const auto constants = params.constants;
          const unsigned items_per_block =
              constants.block_size * constants.items_per_thread;
          if (size % items_per_block != 0) {
            return;
          }
          const unsigned num_blocks = size / items_per_block;
          unsigned reduction_factor = items_per_block;
          if constexpr (!run_type.is_block) {
            reduction_factor =
                constants.sub_warp_size * constants.items_per_thread;
          }

          // // TODO
          // if (reduction_factor == 1) {
          //   return;
          // }

          const unsigned iters = [&] {
            if constexpr (debug_build) {
              return 1;
            } else {
              const unsigned base_iters = 8;
              const unsigned max_iters = 128u;
              return std::min(base_iters * max_size / size, max_iters);
            }
          }();

          constexpr KernelType kernel_type = [&]() {
            if constexpr (run_type.is_fully_dispatched) {
              return type.kernel_type;
            } else {
              return type;
            }
          }();
          std::stringstream name;
          name << "reduce_" << size << "_"
               << magic_enum::enum_name(kernel_type.reduce_impl) << "_"
               << magic_enum::enum_name(kernel_type.item_type)
               << "_reduction_factor_" << reduction_factor << "_block_size_"
               << constants.block_size;
          if constexpr (!run_type.is_block) {
            name << "_sub_warp_size_" << constants.sub_warp_size;
          }
          name << "_items_per_thread_" << constants.items_per_thread;
          benchmark::RegisterBenchmark(
              name.str().c_str(),
              [=, &cpu_input = cpu_input_vec.get(TagV<kernel_type.item_type>),
               &cpu_output = cpu_output_vec.get(TagV<kernel_type.item_type>),
               &cpu_output_from_gpu =
                   cpu_output_from_gpu_vec.get(TagV<kernel_type.item_type>),
               &input = input_vec.get(TagV<kernel_type.item_type>),
               &output = output_vec.get(TagV<kernel_type.item_type>)](
                  benchmark::State &st) {
                st.counters["size"] = size;
                st.counters["impl"] = kernel_type.numeric_reduce_impl();
                st.counters["item_type"] = kernel_type.numeric_item_type();
                st.counters["reduction_factor"] = reduction_factor;
                st.counters["block_size"] = constants.block_size();
                if constexpr (run_type.is_block) {
                  st.counters["sub_warp_size"] = 0;
                } else {
                  st.counters["sub_warp_size"] = constants.sub_warp_size();
                }
                st.counters["items_per_thread"] = constants.items_per_thread();

                input.resize(size);
                auto dist = generate_item_dist<kernel_type.item_type>();
                auto counter = thrust::make_counting_iterator(0u);
                thrust::transform(counter, counter + size, input.begin(),
                                  [=](unsigned n) mutable {
                                    thrust::default_random_engine rng;
                                    rng.discard(n);
                                    return dist(rng);
                                  });
                output.resize(size / reduction_factor, 0);
                thrust::fill(output.begin(), output.end(), 0);
                for (auto _ : st) {
                  using ItemT = GetItemType<kernel_type.item_type>;

                  Span<const ItemT> in = input;
                  Span<ItemT> out = output;

                  if constexpr (run_type.is_fully_dispatched) {
                    constexpr Constants constants = type.constants;
                    if constexpr (run_type.is_block) {
                      block_reduce_fully_dispatched<
                          kernel_type.reduce_impl, constants.block_size,
                          constants.items_per_thread, ItemT>
                          <<<num_blocks, constants.block_size()>>>(in, out);
                    } else {
                      warp_reduce_fully_dispatched<
                          kernel_type.reduce_impl, constants.block_size,
                          constants.sub_warp_size, constants.items_per_thread,
                          ItemT>
                          <<<num_blocks, constants.block_size()>>>(in, out);
                    }
                  } else {
                    if constexpr (kernel_type.reduce_impl ==
                                  NotFullyDispatchedReduceImpl::Mine) {
                      if constexpr (run_type.is_block) {
                        block_reduce_not_fully_dispatched<
                            kernel_type.reduce_impl, ItemT>
                            <<<num_blocks, constants.block_size()>>>(
                                in, out, constants.block_size,
                                constants.items_per_thread);
                      } else {
                        warp_reduce_not_fully_dispatched<
                            kernel_type.reduce_impl, ItemT>
                            <<<num_blocks, constants.block_size()>>>(
                                in, out, constants.block_size,
                                constants.sub_warp_size,
                                constants.items_per_thread);
                      }
                    } else {
                      static_assert(
                          kernel_type.reduce_impl ==
                          NotFullyDispatchedReduceImpl::GeneralKernelLaunch);
                      kernel::WorkDivision division(
                          kernel::Settings{
                              .block_size = constants.block_size,
                              .target_x_block_size = constants.block_size,
                              .force_target_samples = true,
                              .forced_target_samples_per_thread =
                                  constants.items_per_thread,
                          },
                          reduction_factor, size / reduction_factor, 1);

                      debug_assert(division.n_threads_per_unit_extra() == 0);
                      debug_assert(num_blocks == division.total_num_blocks());
                      debug_assert(division.base_samples_per_thread() ==
                                   constants.items_per_thread);
                      if constexpr (run_type.is_block) {
                        debug_assert(division.sample_block_size() ==
                                     constants.block_size);
                      } else {
                        debug_assert(division.sample_block_size() ==
                                     constants.sub_warp_size);
                      }

                      using Reducer =
                          kernel::RuntimeConstantsReducer<ExecutionModel::GPU,
                                                          ItemT>;
                      using ThreadRef = typename Reducer::ThreadRef;

                      auto callable = [=](const kernel::WorkDivision &,
                                          const kernel::GridLocationInfo &info,
                                          const unsigned, const unsigned,
                                          const auto &, ThreadRef &interactor) {
                        auto [start_sample, end_sample, x, y] = info;

                    // this substantially speeds things up for this case
#if 0
                        if (reduction_factor == 1) {
                          out[x] = in[x];
                          return;
                        }
#endif

                        ItemT value = 0;
                        // #pragma unroll
                        for (unsigned i = start_sample; i < end_sample; ++i) {
                          value += in[i + x * reduction_factor];
                        }

                        auto op = interactor.reduce(
                            value, sum, division.sample_block_size());

                        if (op.has_value()) {
                          out[x] = *op;
                        }
                      };

                      kernel::ThreadInteractorLaunchableNoExtraInp<
                          Reducer, decltype(callable)>
                          launchable{
                              .interactor = Reducer{},
                              .callable = callable,
                          };

                      kernel::KernelLaunch<ExecutionModel::GPU>::run(
                          division, 0, num_blocks, launchable);
                    }
                  }
                  CUDA_SYNC_CHK();
                }
                // TODO: generalize as needed...
                if constexpr (debug_build &&
                              kernel_type.item_type == ItemType::Float) {
                  copy_to_vec(input, cpu_input);
                  cpu_output.resize(size / reduction_factor);
                  for (unsigned i = 0; i < size; i += reduction_factor) {
                    float &v = cpu_output[i / reduction_factor];
                    v = 0.f;
                    for (unsigned j = i; j < i + reduction_factor; ++j) {
                      v += cpu_input[j];
                    }
                  }

                  copy_to_vec(output, cpu_output_from_gpu);

                  for (unsigned i = 0; i < size / reduction_factor; ++i) {
                    // correct epsilon will depend on distribution
                    // multiplied by reduction factor because non-commutativity
                    // differences accumulate
                    debug_assert((cpu_output[i] - cpu_output_from_gpu[i]) <
                                 reduction_factor * 1e-5);
                  }
                }
              })
              ->Unit(benchmark::kMillisecond)
              ->Iterations(iters);
        });
      });
    }
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
#endif
