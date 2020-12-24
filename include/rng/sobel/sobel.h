#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "lib/cuda/curand_utils.h"
#include "lib/span.h"
#include "meta/predicate_for_all_values.h"
#include "rng/rng.h"
#include "rng/rng_from_sequence_gen.h"
#include "rng/sobel/settings.h"

#include <curand.h>
#include <curand_kernel.h>

#include <iostream>

namespace rng {
namespace sobel {
namespace detail {
// TODO: consider impl to impl file...
template <ExecutionModel execution_model> struct SobelSequenceGen {
  Span<const float> operator()(unsigned dimension_bound, unsigned count) {
    if constexpr (execution_model == ExecutionModel::CPU) {
      std::cerr << "Sobel rng is currently not supported for the cpu"
                << std::endl;
      abort();
    } else {
      curandDirectionVectors32_t *host_vectors_32_ptr;
      unsigned *host_scramble_constants_32_ptr;

      CURAND_ERROR_CHK(curandGetDirectionVectors32(
          &host_vectors_32_ptr, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
      CURAND_ERROR_CHK(
          curandGetScrambleConstants32(&host_scramble_constants_32_ptr));

      const unsigned vector_size = 32;

      vectors_32_.resize(dimension_bound * vector_size);
      scramble_constants_32_.resize(dimension_bound);

      const unsigned *host_vectors_ptr =
          reinterpret_cast<const unsigned *>(host_vectors_32_ptr);

      thrust::copy(host_vectors_ptr,
                   host_vectors_ptr + dimension_bound * vector_size,
                   vectors_32_.begin());
      thrust::copy(host_scramble_constants_32_ptr,
                   host_scramble_constants_32_ptr + dimension_bound,
                   scramble_constants_32_.begin());

      Span<unsigned> vectors_32 = vectors_32_;
      Span<const unsigned long long> scramble_constants_32 =
          scramble_constants_32_;

      auto start_it = thrust::make_counting_iterator(0u);

      unsigned val_size = dimension_bound * count;

      vals_.resize(val_size);

      // SPEED: horrific efficiency...
      thrust::transform(
          start_it, start_it + val_size, vals_.begin(),
          [=] HOST_DEVICE(unsigned i) {
            unsigned dimension = i % dimension_bound;
            unsigned offset = i / dimension_bound;
            curandStateScrambledSobol32 state;
            auto vectors_32_v = vectors_32.data() + vector_size * dimension;
            auto scramble_const = scramble_constants_32[dimension];
#ifdef __CUDA_ARCH__
            curand_init(vectors_32_v, scramble_const, offset, &state);

            return patched_curand_uniform(&state);
#else
            // needed to ensure lambda has same size on host and device
            (void)offset;
            (void)state;
            (void)vectors_32_v;
            (void)scramble_const;
            assert(false);

            return i;
#endif
          });

      return vals_;
    }
  }

  void init(const Settings &) {}

private:
  ExecVector<execution_model, unsigned> vectors_32_;
  ExecVector<execution_model, unsigned long long> scramble_constants_32_;
  ExecVector<execution_model, float> vals_;
};
} // namespace detail

template <ExecutionModel exec>
using Sobel = RngFromSequenceGen<detail::SobelSequenceGen<exec>, Settings>;

template <ExecutionModel exec>
struct IsRng : BoolWrapper<Rng<Sobel<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsRng>);
} // namespace sobel
} // namespace rng
