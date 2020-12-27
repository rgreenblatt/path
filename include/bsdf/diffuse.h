#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/bsdf_sample.h"
#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace bsdf {
struct Diffuse {
  static constexpr float normalizing_factor = 1 / M_PI;
  Eigen::Array3f diffuse_;

  static constexpr bool discrete = false;
  static constexpr bool continuous = true;

  constexpr bool is_brdf() const { return true; }

  HOST_DEVICE Eigen::Array3f continuous_eval(const UnitVector &,
                                             const UnitVector &,
                                             const UnitVector &) const {
    return diffuse_ * normalizing_factor;
  }

  template <rng::RngState R>
  HOST_DEVICE BSDFSample continuous_sample(const UnitVector &,
                                           const UnitVector &normal,
                                           R &rng) const {
    float v0 = rng.next();
    float v1 = rng.next();

    // sample with cos(theta)
    // CDF wrt. theta = sin^2(theta)
    // inverse CDF = asin(sqrt(rand))

    float phi = 2 * M_PI * v0;
    float theta = std::asin(std::sqrt(v1));

    return {find_relative_vec(normal, phi, theta), diffuse_};
  }
};

static_assert(BSDF<Diffuse>);
} // namespace bsdf
