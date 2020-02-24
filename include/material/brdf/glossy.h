#pragma once

#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "material/brdf.h"
#include "material/utils.h"
#include "render/dir_sample.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace material {
template <> struct BRDFImpl<BRDFType::Glossy> {
public:
  static constexpr bool has_delta_samples = false;
  static constexpr bool has_non_delta_samples = true;
  static constexpr bool is_bsdf = false;

  struct Params {
    Eigen::Array3f specular;
    float shininess;
  };

  HOST_DEVICE BRDFImpl() = default;

  HOST_DEVICE BRDFImpl(const Params &params)
      : normalized_specular_(params.specular *
                             normalizing_factor(params.shininess)),
        shininess_(params.shininess) {}

  HOST_DEVICE Eigen::Array3f brdf(const Eigen::Vector3f &incoming_dir,
                                  const Eigen::Vector3f &outgoing_dir,
                                  const Eigen::Vector3f &normal) const {
    return normalized_specular_ *
           std::pow(reflect_over_normal(incoming_dir, normal).dot(outgoing_dir),
                    shininess_);
  }

  template <rng::RngState R>
  HOST_DEVICE render::DirSample sample(const Eigen::Vector3f &incoming_dir,
                                       const Eigen::Vector3f &normal,
                                       R &rng) const {
    float v0 = rng.next();
    float v1 = rng.next();

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
