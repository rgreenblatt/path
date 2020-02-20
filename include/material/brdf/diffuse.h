#pragma once

#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "material/brdf_type.h"
#include "render/dir_sample.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace material {
template <> class BRDF<BRDFType::Diffuse> {
public:
  static constexpr bool has_delta_samples = false;
  static constexpr bool has_non_delta_samples = true;
  static constexpr bool is_bsdf = false;

  HOST_DEVICE BRDF() = default;

  HOST_DEVICE BRDF(const Eigen::Array3f &diffuse)
      : normalized_diffuse_(diffuse * normalizing_factor) {}

  HOST_DEVICE Eigen::Array3f brdf(const Eigen::Vector3f &,
                                  const Eigen::Vector3f &,
                                  const Eigen::Vector3f &) const {
    return normalized_diffuse_;
  }

  HOST_DEVICE render::DirSample sample(rng::Rng &rng, const Eigen::Vector3f &,
                                       const Eigen::Vector3f &normal) const {
    auto [v0, v1] = rng.sample_2();

    // sample with cos(theta)
    // CDF wrt. theta = sin^2(theta)
    // inverse CDF = asin(sqrt(rand))

    float phi = 2 * M_PI * v0;
    float theta = std::asin(std::sqrt(v1));

    return {find_relative_vec(normal, phi, theta), std::cos(theta)};
  }

private:
  static constexpr float normalizing_factor = 1 / (2 * M_PI);

  Eigen::Array3f normalized_diffuse_;
};
} // namespace material
