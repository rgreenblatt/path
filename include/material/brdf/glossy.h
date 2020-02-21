#pragma once

#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "material/brdf_type.h"
#include "material/utils.h"
#include "render/dir_sample.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace material {
template <> class BRDF<BRDFType::Glossy> {
public:
  static constexpr bool has_delta_samples = false;
  static constexpr bool has_non_delta_samples = true;
  static constexpr bool is_bsdf = false;

  HOST_DEVICE BRDF() = default;

  HOST_DEVICE BRDF(const Eigen::Array3f &specular, float shininess)
      : normalized_specular_(specular * normalizing_factor(shininess)),
        shininess_(shininess) {}

  HOST_DEVICE Eigen::Array3f brdf(const Eigen::Vector3f &incoming_dir,
                                  const Eigen::Vector3f &outgoing_dir,
                                  const Eigen::Vector3f &normal) const {
    return normalized_specular_ *
           std::pow(reflect_over_normal(incoming_dir, normal).dot(outgoing_dir),
                    shininess_);
  }

  HOST_DEVICE render::DirSample sample(rng::Rng &rng,
                                       const Eigen::Vector3f &incoming_dir,
                                       const Eigen::Vector3f &normal) const {
    auto [v0, v1] = rng.sample_2();

    auto reflection = reflect_over_normal(incoming_dir, normal);

    // sample with cos^n(psi)
    // inverse CDF (of psi) = acos(rand^(1/(shininess + 1)))

    float phi = 2 * M_PI * v0;
    float psi = std::acos(std::pow(v1, 1 / (shininess_ + 1)));

    return {find_relative_vec(reflection, phi, psi),
            std::pow(std::cos(psi), shininess_)};
  }

private:
  static constexpr float normalizing_factor(float shininess) {
    return float((shininess + 2) / (2 * M_PI));
  }

  Eigen::Array3f normalized_specular_;
  float shininess_;
};
} // namespace material
