#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/utils.h"
#include "lib/cuda/utils.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace bsdf {
struct Mirror {
  Eigen::Array3f specular_;

  static constexpr bool discrete = true;
  static constexpr bool continuous = false;

  constexpr bool is_brdf() const { return true; }

  template <rng::RngState R>
  HOST_DEVICE BSDFSample discrete_sample(const Eigen::Vector3f &incoming_dir,
                                         const Eigen::Vector3f &normal,
                                         R &) const {
    return {reflect_over_normal(incoming_dir, normal), specular_};
  }
};

static_assert(BSDF<Mirror>);
} // namespace bsdf
