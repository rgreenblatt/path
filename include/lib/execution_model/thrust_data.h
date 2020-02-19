#pragma once

#include "lib/execution_model/caching_thrust_allocator.h"

#include <thrust/execution_policy.h>

template <ExecutionModel execution_model> class ThrustData {
public:
  ThrustData() {
    if constexpr (execution_model == ExecutionModel::GPU) {
      cudaStreamCreate(&stream_v); // TODO flag to avoid sync default
    }
  }

  ~ThrustData() {
    if constexpr (execution_model == ExecutionModel::GPU) {
      cudaStreamDestroy(stream_v);
    }
  }

  auto execution_policy() {
    if constexpr (execution_model == ExecutionModel::GPU) {
      return thrust::cuda::par(alloc).on(stream_v);
    } else {
      return thrust::host;
    }
  }

private:
  struct NoStreamType {};

  std::conditional_t<execution_model == ExecutionModel::CPU, NoStreamType,
                     cudaStream_t>
      stream_v;
  CachingThrustAllocator<execution_model> alloc;
};
