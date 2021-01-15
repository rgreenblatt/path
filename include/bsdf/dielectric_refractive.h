#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/utils.h"
#include "lib/cuda/utils.h"
#include "rng/rng.h"

namespace bsdf {
class DielectricRefractive {
public:
  static constexpr bool discrete = true;
  static constexpr bool continuous = false;

  HOST_DEVICE DielectricRefractive() = default;

  HOST_DEVICE DielectricRefractive(const FloatRGB &specular, float ior)
      : specular_(specular), ior_(ior) {
    r_0_ = (ior_ - 1) / (ior_ + 1);
    r_0_ *= r_0_; // square
  }

  constexpr bool is_brdf() const { return false; }

  template <rng::RngState R>
  HOST_DEVICE BSDFSample discrete_sample(const UnitVector &incoming_dir,
                                         const UnitVector &normal,
                                         R &rng) const {
    float cos_to_normal = std::abs(incoming_dir->dot(*normal));
    float prop_reflected = r_0_ + (1 - r_0_) * std::pow(1 - cos_to_normal, 5);
    if (rng.next() < prop_reflected) {
      // reflect
      return {reflect_over_normal(incoming_dir, normal), specular_};
    } else {
      return {refract_by_normal(ior_, incoming_dir, normal), specular_};
    }
  }

private:
  FloatRGB specular_;
  float ior_;
  float r_0_;
};

static_assert(BSDF<DielectricRefractive>);
} // namespace bsdf
