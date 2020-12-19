#pragma once

#include "lib/cuda/utils.h"
#include "lib/tagged_union.h"
#include "material/brdf.h"
#include "material/brdf/dielectric_refractive.h"
#include "material/brdf/diffuse.h"
#include "material/brdf/glossy.h"
#include "material/brdf/mirror.h"

#include <Eigen/Core>

#include <iostream>

namespace material {
enum class BRDFType {
  Diffuse,
  Glossy,
  Mirror,
  DielectricRefractive,
};

using EnumBRDF = TaggedUnion<BRDFType, DiffuseBRDF, GlossyBRDF, MirrorBRDF,
                             DielectricRefractiveBRDF>;

struct Material {
  EnumBRDF brdf;
  Eigen::Array3f emission;

  HOST_DEVICE bool has_non_delta_samples() const {
    return brdf.visit([](const auto &v) {
      return std::decay_t<decltype(v)>::has_non_delta_samples;
    });
  }

  HOST_DEVICE bool is_bsdf() const {
    return brdf.visit(
        [](const auto &v) { return std::decay_t<decltype(v)>::is_bsdf; });
  }

  HOST_DEVICE Eigen::Array3f
  evaluate_brdf(const Eigen::Vector3f &incoming_dir,
                const Eigen::Vector3f &outgoing_dir,
                const Eigen::Vector3f &normal) const {
    return brdf.visit([&](const auto &v) -> Eigen::Array3f {
      if constexpr (std::decay_t<decltype(v)>::has_non_delta_samples) {
        return v.brdf(incoming_dir, outgoing_dir, normal);
      } else {
        assert(false);
        return Eigen::Array3f::Zero();
      }
    });
  }

  HOST_DEVICE float prob_delta() const {
    return brdf.visit([&](const auto &v) -> float {
      using T = std::decay_t<decltype(v)>;
      if constexpr (T::has_non_delta_samples && !T::has_delta_samples) {
        return 0.0f;
      } else if constexpr (T::has_delta_samples && !T::has_non_delta_samples) {
        return 1.0f;
      } else {
        return v.prob_delta();
      }
    });
  }

  HOST_DEVICE float prob_not_delta() const { return 1.0f - prob_delta(); }

  template <rng::RngState R> HOST_DEVICE bool delta_prob_check(R &rng) const {
    return visit([&](const auto &v) -> bool {
      using T = std::decay_t<decltype(v)>;
      if constexpr (T::has_non_delta_samples && !T::has_delta_samples) {
        return false;
      } else if constexpr (T::has_delta_samples && !T::has_non_delta_samples) {
        return true;
      } else {
        return rng.next() < v.prob_delta();
      }
    });
  }

  template <rng::RngState R>
  HOST_DEVICE DeltaSample delta_sample(const Eigen::Vector3f &incoming_dir,
                                       const Eigen::Vector3f &normal,
                                       R &rng) const {
    return visit([&](const auto &v) -> DeltaSample {
      if constexpr (std::decay_t<decltype(v)>::has_delta_samples) {
        return v.delta_sample(incoming_dir, normal, rng);
      } else {
        assert(false);
        return {};
      }
    });
  }

  template <rng::RngState R>
  HOST_DEVICE render::DirSample sample(const Eigen::Vector3f &incoming_dir,
                                       const Eigen::Vector3f &normal,
                                       R &rng) const {
    return visit([&](const auto &v) -> render::DirSample {
      if constexpr (std::decay_t<decltype(v)>::has_non_delta_samples) {
        return v.sample(incoming_dir, normal, rng);
      } else {
        assert(false);
        return {};
      }
    });
  }
};
} // namespace material
