#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "lib/cuda/curand_utils.h"
#include "lib/span.h"
#include "rng/rng.h"

#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <random>

namespace rng {
namespace sobel_detail {
template <ExecutionModel execution_model> struct SobelSequenceGen {
  Span<const float> operator()(unsigned dimension_bound, unsigned count) {
    unsigned size = dimension_bound * count;
    f_working_mem_.resize(size);
    u_working_mem_.resize(size);
    vals_.resize(size);

    Span<float> f_working_mem = f_working_mem_;
    Span<unsigned> u_working_mem = u_working_mem_;
    Span<float> vals = vals_;

    assert(dimension_bound + 1 < primes.size());

    // SPEED: memory could be reused more efficiently/tiling...
    auto start_it = thrust::make_counting_iterator(0u);
    thrust::for_each(thrust_data_.execution_policy(), start_it,
                     start_it + count, [=] HOST_DEVICE(unsigned i) {
                       const unsigned start = i * dimension_bound;
                       const unsigned end = (i + 1) * dimension_bound;
                       halton(i, f_working_mem.slice(start, end),
                              u_working_mem.slice(start, end),
                              vals.slice(start, end));
                     });

    return vals_;
  }

  using Settings = rng::RngSettings<RngType::Sobel>;

  void init(const Settings &) {}

private:
  ThrustData<execution_model> thrust_data_;

  ExecVector<execution_model, float> f_working_mem_;
  ExecVector<execution_model, unsigned> u_working_mem_;
  ExecVector<execution_model, float> vals_;
};
}
template <ExecutionModel execution_model>
struct RngImpl<RngType::Sobel, execution_model> {
  struct Ref {
    struct State {
      HOST_DEVICE State() = default;

      HOST_DEVICE State(unsigned seed) {
        if constexpr (execution_model == ExecutionModel::GPU) {
          curand_init(seed, 0, 0, &state_);
        } else {
          state_.seed(seed);
          dist_ = CPUDist{0, 1};
        }
      }

      HOST_DEVICE inline float next() {
        if constexpr (execution_model == ExecutionModel::GPU) {
          return curand_uniform(&state_);
        } else {
          return dist_(state_);
        }
      }

    private:
      using GPUState = curandState;
      using CPUState = std::mt19937; // SPEED: maybe try other generators
      std::conditional_t<execution_model == ExecutionModel::GPU, GPUState,
                         CPUState>
          state_;

      using GPUDist = std::tuple<>; // Nothing
      using CPUDist = std::uniform_real_distribution<float>;
      std::conditional_t<execution_model == ExecutionModel::GPU, GPUDist,
                         CPUDist>
          dist_;
    };

    HOST_DEVICE Ref() {}

    Ref(unsigned samples_per, unsigned x_dim, unsigned y_dim)
        : samples_per_(samples_per), x_dim_(x_dim), y_dim_(y_dim) {}

    HOST_DEVICE inline State get_generator(unsigned sample_idx, unsigned x,
                                           unsigned y) const {
      return State(sample_idx + x * samples_per_ + y * samples_per_ * x_dim_);
    }

  private:
    unsigned samples_per_;
    unsigned x_dim_;
    unsigned y_dim_;
  };

  RngImpl() {
    curandDirectionVectors32_t *host_vectors_32_ptr;
    unsigned *host_scramble_constants_32_ptr;

    CURAND_ERROR_CHK(curandGetDirectionVectors32(
        &host_vectors_32_ptr, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    CURAND_ERROR_CHK(
        curandGetScrambleConstants32(&host_scramble_constants_32_ptr));

    const unsigned size = 20000;

    vectors_32_.resize(size);
    scramble_constants_32_.resize(size);

    thrust::transform(host_vectors_32_ptr, host_vectors_32_ptr + size,
                      vectors_32_.begin(),
                      [] HOST_DEVICE(const curandDirectionVectors32_t &vecs) {
                        std::array<unsigned, 32> out;
                        for (unsigned i = 0; i < out.size(); i++) {
                          out[i] = vecs[i];
                        }

                        return out;
                      });
    thrust::copy(host_scramble_constants_32_ptr,
                 host_scramble_constants_32_ptr + size,
                 scramble_constants_32_.begin());
  }

  Ref gen(const RngSettings<RngType::Sobel> &, unsigned samples_per,
          unsigned x_dim, unsigned y_dim, unsigned max_sample_size) {
    if (execution_model == ExecutionModel::CPU) {
      std::cerr << "Sobel rng is currently not supported for the cpu"
                << std::endl;
      abort();
    } else {
#if 0
      unsigned dimension_bound = max_sample_size;
      unsigned next_values = 4;
      auto start_it = thrust::make_counting_iterator(0u);
      unsigned count = next_values * samples_per * dimension_bound;

      states_.resize(count);

      thrust::transform(start_it, start_it + count, states_.begin(),
                        [=] HOST_DEVICE(unsigned i) {
                          unsigned dimension = i % dimension_bound;
                          unsigned offset = i / dimension_bound;
                        });

#endif
      return Ref(samples_per, x_dim, y_dim);
    }
  }

private:
  ExecVector<execution_model, std::array<unsigned, 32>> vectors_32_;
  ExecVector<execution_model, unsigned long long> scramble_constants_32_;
  ExecVector<execution_model, curandStateSobol32_t> states_;
};

static_assert(Rng<RngType::Sobel, ExecutionModel::GPU>);
static_assert(Rng<RngType::Sobel, ExecutionModel::CPU>);
} // namespace rng
