#pragma once

#include "integrate/term_prob/n_iters/settings.h"
#include "integrate/term_prob/term_prob.h"
#include "lib/cuda/utils.h"

namespace integrate {
namespace term_prob {
namespace n_iters {
struct NIters {
  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &settings) : iters_(settings.iters) {}

    ATTR_PURE HOST_DEVICE float operator()(unsigned iters,
                                           const FloatRGB &) const {
      if (iters >= iters_) {
        return 1.0f;
      } else {
        return 0.0f;
      }
    }

  private:
    unsigned iters_;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};

static_assert(TermProb<NIters, Settings>);
} // namespace n_iters
} // namespace term_prob
} // namespace integrate
