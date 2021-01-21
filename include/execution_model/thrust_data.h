#pragma once

#include "execution_model/caching_thrust_allocator.h"
#include "meta/all_values_enum.h"
#include "meta/predicate_for_all_values.h"

#include <thrust/execution_policy.h>

#include <concepts>

class Stream {
public:
  Stream() { cudaStreamCreate(&stream_v_); }

  Stream(const Stream &) = delete;

  Stream(Stream &&) = delete;

  ~Stream() { cudaStreamDestroy(stream_v_); }

  cudaStream_t &stream_v() { return stream_v_; }

  const cudaStream_t &stream_v() const { return stream_v_; }

private:
  cudaStream_t stream_v_;
};

template <ExecutionModel execution_model> class ThrustData {
public:
  ThrustData() {
    if constexpr (execution_model == ExecutionModel::GPU) {
      stream_v = std::make_unique<Stream>();
    }
  }

  auto execution_policy() {
    if constexpr (execution_model == ExecutionModel::GPU) {
      return thrust::cuda::par(alloc).on(stream_v->stream_v());
    } else {
      return thrust::host;
    }
  }

private:
  CachingThrustAllocator<execution_model> alloc;
  struct EmptyT {};
  [[no_unique_address]] std::conditional_t<
      execution_model == ExecutionModel::CPU, EmptyT, std::unique_ptr<Stream>>
      stream_v;
};

template <ExecutionModel exec>
struct ThrustDataMovable : std::bool_constant<std::movable<ThrustData<exec>>> {
};

static_assert(PredicateForAllValues<ExecutionModel>::value<ThrustDataMovable>);
