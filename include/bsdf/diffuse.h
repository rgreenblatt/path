#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/bsdf_sample.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "rng/rng.h"

namespace bsdf {
struct Diffuse {
  static constexpr float normalizing_factor = 1 / M_PI;
  FloatRGB diffuse;

  static constexpr bool discrete = false;
  static constexpr bool continuous = true;

  ATTR_PURE constexpr bool is_brdf() const { return true; }

  ATTR_PURE_NDEBUG HOST_DEVICE FloatRGB continuous_eval(
      const UnitVector &, const UnitVector &, const UnitVector &) const {
    return diffuse * normalizing_factor;
  }

  template <rng::RngState R>
  HOST_DEVICE BSDFSample continuous_sample(const UnitVector &incoming_dir,
                                           const UnitVector &normal,
                                           R &rng) const {
    float v0 = rng.next();
    float v1 = rng.next();

    // sample with cos(theta)
    // CDF wrt. theta = sin^2(theta)
    // inverse CDF = asin(sqrt(rand))

    float phi = 2 * M_PI * v0;
    float theta = std::asin(std::sqrt(v1));

    // handle case where we hit back of triangle (the behaviour is a bit ill
    // specified/maybe shouldn't happen in most meshes, but we want to be
    // correct...)
    auto aligned_normal = normal;
    if (incoming_dir->dot(*normal) > 0.) {
      aligned_normal = -normal;
    }

    return {find_relative_vec(aligned_normal, phi, theta), diffuse};
  }
};

static_assert(BSDF<Diffuse>);
} // namespace bsdf
