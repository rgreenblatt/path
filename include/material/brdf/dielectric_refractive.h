#pragma once

#include "lib/cuda/utils.h"
#include "material/brdf.h"
#include "material/utils.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace material {
class DielectricRefractiveBRDF {
public:
  static constexpr bool has_delta_samples = true;
  static constexpr bool has_non_delta_samples = false;
  static constexpr bool is_bsdf = true;

  struct Params {
    Eigen::Array3f specular;
    float ior;
  };

  HOST_DEVICE DielectricRefractiveBRDF() = default;

  HOST_DEVICE DielectricRefractiveBRDF(const Params &params)
      : specular_(params.specular), ior_(params.ior) {
    r_0_ = (ior_ - 1) / (ior_ + 1);
    r_0_ *= r_0_;
  }

  template <rng::RngState R>
  HOST_DEVICE DeltaSample delta_sample(const Eigen::Vector3f &incoming_dir,
                                       const Eigen::Vector3f &normal,
                                       R &rng) const {
    float cos_to_normal = std::abs(incoming_dir.dot(normal));
    float prop_reflected = r_0_ + (1 - r_0_) * std::pow(1 - cos_to_normal, 5);
    if (rng.next() < prop_reflected) {
      // reflect
      return {reflect_over_normal(incoming_dir, normal), specular_};
    } else {
      return {refract_by_normal(ior_, incoming_dir, normal), specular_};
    }
  }

private:
  Eigen::Array3f specular_;
  float ior_;
  float r_0_;
};

static_assert(BRDF<DielectricRefractiveBRDF>);
} // namespace material
