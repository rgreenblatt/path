#pragma once

#include "render/dir_sample.h"
// TODO: refactor rng out somehow...
#include "rng/rng.h"
#include "rng/test_rng_state_type.h"

#include <Eigen/Core>

#include <concepts>

namespace material {
struct DeltaSample {
  Eigen::Vector3f direction;
  Eigen::Array3f weight;
};

template <typename T> concept BRDF = requires {
  { T::has_delta_samples }
  ->std::common_with<bool>;
  { T::has_non_delta_samples }
  ->std::common_with<bool>;
  { T::is_bsdf }
  ->std::common_with<bool>;

  typename T::Params;

  requires std::constructible_from<T, typename T::Params>;

  requires requires(const T &brdf, const Eigen::Vector3f &incoming_dir,
                    const Eigen::Vector3f &outgoing_dir,
                    const Eigen::Vector3f &normal) {
    { brdf.brdf(incoming_dir, outgoing_dir, normal) }
    ->std::common_with<Eigen::Array3f>;
  }
  || !T::has_non_delta_samples;

  requires requires(const T &brdf, const Eigen::Vector3f &incoming_dir,
                    const Eigen::Vector3f &normal, rng::DummyRngState &r) {
    { brdf.sample(incoming_dir, normal, r) }
    ->std::common_with<render::DirSample>;
  }
  || !T::has_non_delta_samples;

  requires requires(const T &brdf, const Eigen::Vector3f &incoming_dir,
                    const Eigen::Vector3f &normal, rng::DummyRngState &r) {
    { brdf.delta_sample(incoming_dir, normal, r) }
    ->std::common_with<DeltaSample>;
  }
  || !T::has_delta_samples;

  requires requires(const T &brdf, const Eigen::Vector3f &incoming_dir,
                    const Eigen::Vector3f &normal, rng::DummyRngState &r) {
    { brdf.prob_delta(incoming_dir, normal, r) }
    ->std::common_with<float>;
  }
  || !T::has_delta_samples || !T::has_non_delta_samples;
};
} // namespace material
