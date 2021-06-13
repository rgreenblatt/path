#pragma once

#include "execution_model/execution_model.h"
#include "lib/attribute.h"
#include "lib/cuda/curand_utils.h"
#include "lib/cuda/utils.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/predicate_for_all_values.h"
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

      HOST_DEVICE inline State save() const { return *this; }

      HOST_DEVICE inline auto &state() { return state_; }

    private:
      using GPUState = curandState;
      using CPUState = std::mt19937; // SPEED: maybe try other generators
      std::conditional_t<exec == ExecutionModel::GPU, GPUState, CPUState>
          state_;

      struct GPUDist {}; // Nothing
      using CPUDist = std::uniform_real_distribution<float>;
      [[no_unique_address]] std::conditional_t<exec == ExecutionModel::GPU,
                                               GPUDist, CPUDist>
          dist_;
    };

    using SavedState = State;

    ATTR_PURE_NDEBUG HOST_DEVICE inline State
    get_generator(unsigned sample_idx, unsigned location) const {
      return State(sample_idx + location * samples_per_);
    }

    ATTR_PURE_NDEBUG HOST_DEVICE inline State
    state_from_saved(unsigned, unsigned, SavedState state) const {
      return state;
    }
  };

  Ref gen(const Settings &, unsigned samples_per, unsigned /*n_locations*/) {
    return Ref{samples_per};
  }
};

template <ExecutionModel exec>
struct IsRng : std::bool_constant<Rng<Uniform<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsRng>);
} // namespace uniform
} // namespace rng
