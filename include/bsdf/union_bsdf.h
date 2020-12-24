#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/dielectric_refractive.h"
#include "bsdf/diffuse.h"
#include "bsdf/glossy.h"
#include "bsdf/mirror.h"
#include "lib/cuda/utils.h"
#include "lib/tagged_union.h"

#include <Eigen/Core>

namespace bsdf {
enum class BSDFType {
  Diffuse,
  Glossy,
  Mirror,
  DielectricRefractive,
};

struct UnionBSDF {
  using Union =
      TaggedUnion<BSDFType, Diffuse, Glossy, Mirror, DielectricRefractive>;
  Union bsdf;

  static constexpr bool discrete = true;
  static constexpr bool continuous = true;

  HOST_DEVICE bool is_brdf() const {
    return bsdf.visit([](const auto &v) { return v.is_brdf(); });
  }

  HOST_DEVICE Eigen::Array3f
  continuous_eval(const Eigen::Vector3f &incoming_dir,
                  const Eigen::Vector3f &outgoing_dir,
                  const Eigen::Vector3f &normal) const {
    return bsdf.visit([&](const auto &v) -> Eigen::Array3f {
      if constexpr (std::decay_t<decltype(v)>::continuous) {
        return v.continuous_eval(incoming_dir, outgoing_dir, normal);
      } else {
        assert(false);
        return {};
      }
    });
  }

  HOST_DEVICE float prob_continuous(const Eigen::Vector3f &incoming_dir,
                                    const Eigen::Vector3f &outgoing_dir) const {
    return bsdf.visit([&](const auto &v) -> float {
      using T = std::decay_t<decltype(v)>;
      if constexpr (T::continuous && !T::discrete) {
        return 1.0f;
      } else if constexpr (T::discrete && !T::continuous) {
        return 0.0f;
      } else {
        return v.prob_continuous(incoming_dir, outgoing_dir);
      }
    });
  }

  template <rng::RngState R>
  HOST_DEVICE auto discrete_sample(const Eigen::Vector3f &incoming_dir,
                                   const Eigen::Vector3f &normal,
                                   R &rng) const {
    return bsdf.visit([&](const auto &v) -> BSDFSample {
      if constexpr (std::decay_t<decltype(v)>::discrete) {
        return v.discrete_sample(incoming_dir, normal, rng);
      } else {
        assert(false);
        return {};
      }
    });
  }

  template <rng::RngState R>
  HOST_DEVICE auto continuous_sample(const Eigen::Vector3f &incoming_dir,
                                     const Eigen::Vector3f &normal,
                                     R &rng) const {
    return bsdf.visit([&](const auto &v) -> BSDFSample {
      if constexpr (std::decay_t<decltype(v)>::continuous) {
        return v.continuous_sample(incoming_dir, normal, rng);
      } else {
        assert(false);
        return {};
      }
    });
  }
};

static_assert(BSDF<UnionBSDF>);
} // namespace bsdf
