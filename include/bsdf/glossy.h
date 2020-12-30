#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/bsdf_sample.h"
#include "bsdf/utils.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"
#include "lib/projection.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace bsdf {
struct Glossy {
public:
  static constexpr bool discrete = false;
  static constexpr bool continuous = true;

  HOST_DEVICE Glossy() = default;

  HOST_DEVICE Glossy(const Eigen::Array3f &specular, float shininess)
      : normalizing_factor_((shininess + 2.f) /
                            (2.f * static_cast<float>(M_PI))),
        specular_(specular), shininess_(shininess),
        inv_shininess_p_1_(1.f / (shininess_ + 1.f)) {}

  ATTR_PURE constexpr bool is_brdf() const { return true; }

  ATTR_PURE_NDEBUG HOST_DEVICE Eigen::Array3f
  continuous_eval(const UnitVector &incoming_dir,
                  const UnitVector &outgoing_dir,
                  const UnitVector &normal) const {
    return specular_ * normalizing_factor_ *
           std::pow(std::max(reflect_over_normal(incoming_dir, normal)
                                 ->dot(*outgoing_dir),
                             0.0f),
                    shininess_);
  }

  template <rng::RngState R>
  HOST_DEVICE BSDFSample continuous_sample(const UnitVector &incoming_dir,
                                           const UnitVector &normal,
                                           R &rng) const {
    float v0 = rng.next();
    float v1 = rng.next();

    auto reflection = reflect_over_normal(incoming_dir, normal);

    // sample with cos^n(psi)
    // inverse CDF (of psi) = acos(rand^(1/(shininess + 1)))

    float phi = 2 * M_PI * v0;
    float psi = std::acos(std::pow(v1, inv_shininess_p_1_));

    return {find_relative_vec(reflection, phi, psi), specular_};
  }

private:
  float normalizing_factor_;
  Eigen::Array3f specular_;
  float shininess_;
  float inv_shininess_p_1_; // purely an optimization
};

static_assert(BSDF<Glossy>);
} // namespace bsdf
