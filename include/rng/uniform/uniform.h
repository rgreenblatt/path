#pragma once

#include "execution_model/execution_model.h"
#include "lib/cuda/curand_utils.h"
#include "lib/cuda/utils.h"
#include "meta/predicate_for_all_values.h"
#include "rng/rng.h"
#include "rng/uniform/settings.h"

#include <curand_kernel.h>

#include <random>

namespace rng {
namespace uniform {
template <ExecutionModel exec> struct Uniform {
  struct Ref {
    unsigned samples_per_;

    class State {
    public:
      HOST_DEVICE State() = default;

      HOST_DEVICE State(unsigned seed) {
        if constexpr (exec == ExecutionModel::GPU) {
          curand_init(seed, 0, 0, &state_);
        } else {
          state_.seed(seed);
          dist_ = CPUDist{0, 1};
        }
      }

      HOST_DEVICE inline float next() {
        if constexpr (exec == ExecutionModel::GPU) {
          return patched_curand_uniform(&state_);
        } else {
          return dist_(state_);
        }
      }

    private:
      using GPUState = curandState;
      using CPUState = std::mt19937; // SPEED: maybe try other generators
      std::conditional_t<exec == ExecutionModel::GPU, GPUState, CPUState>
          state_;

      using GPUDist = std::tuple<>; // Nothing
      using CPUDist = std::uniform_real_distribution<float>;
      std::conditional_t<exec == ExecutionModel::GPU, GPUDist, CPUDist> dist_;
    };

    HOST_DEVICE inline State get_generator(unsigned sample_idx,
                                           unsigned location) const {
      return State(sample_idx + location * samples_per_);
    }
  };

  Ref gen(const Settings &, unsigned samples_per, unsigned /*n_locations*/) {
    return Ref{samples_per};
  }
};

static_assert(Rng<Uniform<ExecutionModel::GPU>, Settings>);
static_assert(Rng<Uniform<ExecutionModel::CPU>, Settings>);

template <ExecutionModel exec>
struct IsRng : BoolWrapper<Rng<Uniform<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsRng>);
} // namespace uniform
} // namespace rng
