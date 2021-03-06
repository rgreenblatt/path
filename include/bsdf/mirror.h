#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/utils.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "rng/rng.h"

namespace bsdf {
struct Mirror {
  FloatRGB specular;

  static constexpr bool discrete = true;
  static constexpr bool continuous = false;

  ATTR_PURE constexpr bool is_brdf() const { return true; }

  template <rng::RngState R>
  ATTR_PURE_NDEBUG HOST_DEVICE BSDFSample discrete_sample(
      const UnitVector &incoming_dir, const UnitVector &normal, R &) const {
    return {reflect_over_normal(incoming_dir, normal), specular};
  }
};

static_assert(BSDF<Mirror>);
} // namespace bsdf
