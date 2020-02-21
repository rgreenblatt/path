#pragma once

#include "lib/cuda/utils.h"
#include "material/brdf_type.h"
#include "material/utils.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace material {
template <> class BRDF<BRDFType::DielectricRefractive> {
public:
  static constexpr bool has_delta_samples = true;
  static constexpr bool has_non_delta_samples = false;
  static constexpr bool is_bsdf = true;

  HOST_DEVICE BRDF() = default;

  HOST_DEVICE BRDF(const Eigen::Array3f &specular, float ior)
      : specular_(specular), ior_(ior) {
    r_0_ = (ior - 1) / (ior + 1);
    r_0_ *= r_0_;
  }

  HOST_DEVICE std::tuple<Eigen::Vector3f, Eigen::Array3f>
  delta_sample(rng::Rng &rng, const Eigen::Vector3f &incoming_dir,
               const Eigen::Vector3f &normal) const {
    float cos_to_normal = incoming_dir.dot(normal);
    float prop_reflected = r_0_ + (1 - r_0_) * std::pow(1 - cos_to_normal, 5);
    if (rng.sample_1() < prop_reflected) {
      // reflect
      return std::make_tuple(reflect_over_normal(incoming_dir, normal),
                             specular_);
    } else {
      return std::make_tuple(refract_by_normal(ior_, incoming_dir, normal),
                             specular_);
    }
  }

private:
  Eigen::Array3f specular_;
  float ior_;
  float r_0_;
};
} // namespace material
