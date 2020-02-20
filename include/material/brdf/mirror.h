#pragma once

#include "lib/cuda/utils.h"
#include "material/brdf_type.h"
#include "material/utils.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace material {
template <> class BRDF<BRDFType::Mirror> {
public:
  static constexpr bool has_delta_samples = true;
  static constexpr bool has_non_delta_samples = false;
  static constexpr bool is_bsdf = false;

  HOST_DEVICE BRDF() = default;

  HOST_DEVICE BRDF(const Eigen::Array3f &specular) : specular_(specular) {}

  HOST_DEVICE std::tuple<Eigen::Vector3f, Eigen::Array3f>
  delta_sample(rng::Rng &, const Eigen::Vector3f &incoming_dir,
                  const Eigen::Vector3f &normal) const {
    return std::make_tuple(reflect_over_normal(incoming_dir, normal),
                           specular_);
  }

private:
  Eigen::Array3f specular_;
};
} // namespace material
