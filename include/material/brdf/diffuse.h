#pragma once

#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "material/brdf.h"
#include "render/dir_sample.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace material {
template <> struct BRDFImpl<BRDFType::Diffuse> {
public:
  static constexpr bool has_delta_samples = false;
  static constexpr bool has_non_delta_samples = true;
  static constexpr bool is_bsdf = false;

  struct Params {
    Eigen::Array3f diffuse;
  };

  HOST_DEVICE BRDFImpl() = default;

  HOST_DEVICE BRDFImpl(const Params &params)
      : normalized_diffuse_(params.diffuse * normalizing_factor) {}

  HOST_DEVICE Eigen::Array3f brdf(const Eigen::Vector3f &,
                                  const Eigen::Vector3f &,
                                  const Eigen::Vector3f &) const {
    return normalized_diffuse_;
  }

  template <rng::RngState R>
  HOST_DEVICE render::DirSample
  sample(const Eigen::Vector3f &, const Eigen::Vector3f &normal, R &rng) const {
    float v0 = rng.next();
    float v1 = rng.next();

    // sample with cos(theta)
    // CDF wrt. theta = sin^2(theta)
    // inverse CDF = asin(sqrt(rand))

    float phi = 2 * M_PI * v0;
    float theta = std::asin(std::sqrt(v1));

    return {find_relative_vec(normal, phi, theta),
            std::cos(theta) * normalizing_factor};
  }

private:
  static constexpr float normalizing_factor = 1 / M_PI;

  Eigen::Array3f normalized_diffuse_;
};
} // namespace material
