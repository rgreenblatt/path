#pragma once

#include "integrate/term_prob/multiplier_func/settings.h"
#include "integrate/term_prob/term_prob.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"

namespace integrate {
namespace term_prob {
namespace multiplier_func {
struct MultiplierFunc {
  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &settings)
        : exp_(settings.exp), min_prob_(settings.min_prob) {}

    ATTR_PURE_NDEBUG HOST_DEVICE float
    operator()(unsigned, const FloatRGB &multiplier) const {
      // normalization (clamp to deal with cases where multiplier may be
      // negative)
      float squared_norm = std::clamp(
          ((multiplier / (multiplier + 1)) * 0.57).matrix().squaredNorm(), 0.0f,
          1.0f);

      float term_prob = std::abs(std::pow(1 - squared_norm, exp_));

      return std::max(term_prob, min_prob_);
    }

  private:
    float exp_;
    float min_prob_;
  };

  Ref gen(const Settings &settings) { return Ref(settings); }
};

static_assert(TermProb<MultiplierFunc, Settings>);
} // namespace multiplier_func
} // namespace term_prob
} // namespace integrate
