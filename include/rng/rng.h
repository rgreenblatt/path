#pragma once

#include "lib/cuda/utils.h"
#include "rng/halton.h"

#include <assert.h>

namespace rng {
// TODO: consider making configurable...
class Rng {
public:
  using StateType = uint16_t;

  HOST_DEVICE Rng() = default;

  HOST_DEVICE Rng(StateType state, StateType max_state)
      : state_(state), max_state_(max_state) {}

  // TODO: should sampling increment???
  template <unsigned n> HOST_DEVICE auto sample_n() {
#ifdef __CUDA_ARCH__
    return halton<n>(state_);
#else
    std::array<float, n> out;
    for (unsigned i = 0; i < n; i++) {
      out[i] = float(double(rand()) / double(RAND_MAX));
    }
    return out;
#endif
  }

  HOST_DEVICE auto sample_1() { return sample_n<1>()[0]; }

  HOST_DEVICE auto sample_2() { return sample_n<2>(); }

  HOST_DEVICE void set_state(StateType state) {
    assert(state_ < max_state_);
    state_ = state;
  }

  HOST_DEVICE void next_state() {
    state_++;

    // wrap
    if (state_ >= max_state_) {
      state_ = 0;
    }
  }

private:
  StateType state_;
  StateType max_state_;
};
} // namespace rng