#pragma once

#include "bsdf/bsdf.h"
#include "bsdf/dielectric_refractive.h"
#include "bsdf/diffuse.h"
#include "bsdf/glossy.h"
#include "bsdf/mirror.h"
#include "lib/assert.h"
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

  HOST_DEVICE Eigen::Array3f continuous_eval(const UnitVector &incoming_dir,
                                             const UnitVector &outgoing_dir,
                                             const UnitVector &normal) const {
    return bsdf.visit([&](const auto &v) -> Eigen::Array3f {
      if constexpr (std::decay_t<decltype(v)>::continuous) {
        return v.continuous_eval(incoming_dir, outgoing_dir, normal);
      } else {
        unreachable_unchecked();
      }
    });
  }

  HOST_DEVICE float prob_continuous(const UnitVector &incoming_dir,
                                    const UnitVector &outgoing_dir) const {
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
  HOST_DEVICE auto discrete_sample(const UnitVector &incoming_dir,
                                   const UnitVector &normal, R &rng) const {
    return bsdf.visit([&](const auto &v) -> BSDFSample {
      if constexpr (std::decay_t<decltype(v)>::discrete) {
        return v.discrete_sample(incoming_dir, normal, rng);
      } else {
        unreachable_unchecked();
      }
    });
  }

  template <rng::RngState R>
  HOST_DEVICE auto continuous_sample(const UnitVector &incoming_dir,
                                     const UnitVector &normal, R &rng) const {
    return bsdf.visit([&](const auto &v) -> BSDFSample {
      if constexpr (std::decay_t<decltype(v)>::continuous) {
        return v.continuous_sample(incoming_dir, normal, rng);
      } else {
        unreachable_unchecked();
      }
    });
  }
};

static_assert(BSDF<UnionBSDF>);
} // namespace bsdf
