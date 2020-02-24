#pragma once

#include "lib/cuda/utils.h"
#include "material/brdf.h"
#include "material/utils.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace material {
template <> struct BRDFImpl<BRDFType::Mirror> {
public:
  static constexpr bool has_delta_samples = true;
  static constexpr bool has_non_delta_samples = false;
  static constexpr bool is_bsdf = false;

  struct Params {
    Eigen::Array3f specular;
  };

  HOST_DEVICE BRDFImpl() = default;

  HOST_DEVICE BRDFImpl(const Params &params) : specular_(params.specular) {}

  template <rng::RngState R>
  HOST_DEVICE DeltaSample delta_sample(const Eigen::Vector3f &incoming_dir,
                                       const Eigen::Vector3f &normal,
                                       R &) const {
    return {reflect_over_normal(incoming_dir, normal), specular_};
  }

private:
  Eigen::Array3f specular_;
};
} // namespace material
