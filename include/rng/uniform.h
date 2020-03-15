#pragma once

#include "rng/rng.h"

#include <curand_kernel.h>

#include <random>

namespace rng {
template <ExecutionModel execution_model>
struct RngImpl<RngType::Uniform, execution_model> {
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

  Ref gen(const RngSettings<RngType::Uniform> &, unsigned samples_per,
          unsigned x_dim, unsigned y_dim, unsigned) {
    return Ref(samples_per, x_dim, y_dim);
  }
};

static_assert(Rng<RngType::Uniform, ExecutionModel::GPU>);
static_assert(Rng<RngType::Uniform, ExecutionModel::CPU>);
} // namespace rng
