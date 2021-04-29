#pragma once

#include "integrate/term_prob/normalize/settings.h"
#include "integrate/term_prob/term_prob.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"

namespace integrate {
namespace term_prob {
namespace normalize {
struct Normalize {
  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &settings) : min_prob_(settings.min_prob) {}

    ATTR_PURE_NDEBUG HOST_DEVICE float
    operator()(unsigned, const FloatRGB &multiplier) const {
      float term_prob = 1.f - 1.f / (multiplier.sum() / multiplier.size);

      return std::clamp(term_prob, min_prob_, 1.f);
    }

  private:
    float min_prob_;
  };

  Ref gen(const Settings &settings) { return Ref(settings); }
};

static_assert(TermProb<Normalize, Settings>);
} // namespace normalize
} // namespace term_prob
} // namespace integrate
