#pragma once

#include "render/dir_sample.h"
// TODO: refactor rng out somehow...
#include "rng/rng.h"
#include "rng/test_rng_state_type.h"

#include <Eigen/Core>

#include <concepts>

namespace material {
enum class BRDFType {
  Diffuse,
  Glossy,
  Mirror,
  DielectricRefractive,
};

template <BRDFType type> struct BRDFImpl;

struct DeltaSample {
  Eigen::Vector3f direction;
  Eigen::Array3f weight;
};

template <BRDFType type> concept BRDF = requires {
  typename BRDFImpl<type>;
  { BRDFImpl<type>::has_delta_samples }
  ->std::common_with<bool>;
  { BRDFImpl<type>::has_non_delta_samples }
  ->std::common_with<bool>;
  { BRDFImpl<type>::is_bsdf }
  ->std::common_with<bool>;

  typename BRDFImpl<type>::Params;

  std::constructible_from<BRDFImpl<type>, typename BRDFImpl<type>::Params>;

  requires requires(
      const BRDFImpl<type> &brdf, const Eigen::Vector3f &incoming_dir,
      const Eigen::Vector3f &outgoing_dir, const Eigen::Vector3f &normal) {
    { brdf.brdf(incoming_dir, outgoing_dir, normal) }
    ->std::common_with<Eigen::Array3f>;
  }
  || !BRDFImpl<type>::has_non_delta_samples;

  requires requires(const BRDFImpl<type> &brdf,
                    const Eigen::Vector3f &incoming_dir,
                    const Eigen::Vector3f &normal, rng::TestRngStateT &r) {
    { brdf.sample(incoming_dir, normal, r) }
    ->std::common_with<render::DirSample>;
  }
  || !BRDFImpl<type>::has_non_delta_samples;

  requires requires(const BRDFImpl<type> &brdf,
                    const Eigen::Vector3f &incoming_dir,
                    const Eigen::Vector3f &normal, rng::TestRngStateT &r) {
    { brdf.delta_sample(incoming_dir, normal, r) }
    ->std::common_with<DeltaSample>;
  }
  || !BRDFImpl<type>::has_delta_samples;

  requires requires(const BRDFImpl<type> &brdf,
                    const Eigen::Vector3f &incoming_dir,
                    const Eigen::Vector3f &normal, rng::TestRngStateT &r) {
    { brdf.prob_delta(incoming_dir, normal, r) }
    ->std::common_with<float>;
  }
  || !BRDFImpl<type>::has_delta_samples ||
      !BRDFImpl<type>::has_non_delta_samples;
};

// TODO: move some functionality from material to BRDF trait...
template <BRDFType type> requires BRDF<type> struct BRDFT : BRDFImpl<type> {
  using BRDFImpl<type>::BRDFImpl;
};
} // namespace material
