#pragma once

#include "integrate/term_prob/constant/settings.h"
#include "integrate/term_prob/term_prob.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"

namespace integrate {
namespace term_prob {
namespace constant {
struct Constant {
  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &settings) : prob_(settings.prob) {
      debug_assert(prob_ > 0.);
      debug_assert(prob_ <= 1.);
    }

    ATTR_PURE HOST_DEVICE float operator()(unsigned, const FloatRGB &) const {
      return prob_;
    }

  private:
    float prob_;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};

static_assert(TermProb<Constant, Settings>);
} // namespace constant
} // namespace term_prob
} // namespace integrate
