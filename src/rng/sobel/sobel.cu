#include "data_structure/copyable_to_vec.h"
#include "execution_model/device_vector.h"
#include "execution_model/execution_model_vector_type.h"
#include "lib/cuda/curand_utils.h"
#include "lib/cuda/utils.h"
#include "lib/span.h"
#include "rng/sobel/settings.h"
#include "rng/sobel/sobel.h"

#include <curand.h>
#include <curand_kernel.h>

#include <iostream>

namespace rng {
namespace sobel {
namespace detail {
// use curand to generate for simplicity and gpu/cpu consistancy
template <ExecutionModel exec> class SobelSequenceGen<exec>::Generator {
public:
  Span<const float> gen(const SobelSettings &, unsigned dimension_bound,
                        unsigned count) {
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

    gpu_vals_.resize(val_size);

    // SPEED: horrific efficiency...
    thrust::transform(
        start_it, start_it + val_size, gpu_vals_.begin(),
        [=] __device__(unsigned i) {
          unsigned dimension = i % dimension_bound;
          unsigned offset = i / dimension_bound;
          curandStateScrambledSobol32 state;
          auto vectors_32_v = vectors_32.data() + vector_size * dimension;
          auto scramble_const = scramble_constants_32[dimension];
          curand_init(vectors_32_v, scramble_const, offset, &state);

          return patched_curand_uniform(&state);
        });

    if constexpr (exec == ExecutionModel::GPU) {
      return gpu_vals_;
    } else {
      copy_to_vec(gpu_vals_, vals_);

      return vals_;
    }
  }

private:
  DeviceVector<unsigned> vectors_32_;
  DeviceVector<unsigned long long> scramble_constants_32_;
  DeviceVector<float> gpu_vals_;
  ExecVector<exec, float> vals_;
};

template <ExecutionModel exec>
Span<const float> SobelSequenceGen<exec>::gen(const SobelSettings &settings,
                                              unsigned dimension_bound,
                                              unsigned count) {
  return gen_->gen(settings, dimension_bound, count);
}

template <ExecutionModel exec> SobelSequenceGen<exec>::SobelSequenceGen() {
  gen_ = std::make_unique<Generator>();
}

template <ExecutionModel exec>
SobelSequenceGen<exec>::~SobelSequenceGen() = default;

template <ExecutionModel exec>
SobelSequenceGen<exec>::SobelSequenceGen(SobelSequenceGen &&) = default;

template <ExecutionModel exec>
SobelSequenceGen<exec> &
SobelSequenceGen<exec>::operator=(SobelSequenceGen &&) = default;

template class SobelSequenceGen<ExecutionModel::CPU>;
template class SobelSequenceGen<ExecutionModel::GPU>;
} // namespace detail
} // namespace sobel
} // namespace rng
