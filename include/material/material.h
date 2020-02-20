#pragma once

#include "lib/cuda/utils.h"
#include "material/brdf.h"

#include <Eigen/Core>

#include <iostream>

namespace material {
// TODO:
class Material {
public:
  HOST_DEVICE Material()
      : Material(BRDF<BRDFType::Diffuse>(), Eigen::Array3f::Zero()) {}

  HOST_DEVICE Material(const Material &other) { copy_in_other(other); }

  HOST_DEVICE Material &operator=(const Material &other) {
    copy_in_other(other);
    
    return *this;
  }

  template <BRDFType type>
  HOST_DEVICE Material(const BRDF<type> &brdf, const Eigen::Array3f &emission)
      : type_(type), emission_(emission) {
    visit([&](auto &val) {
      if constexpr (std::is_same_v<std::decay_t<decltype(val)>, BRDF<type>>) {
        val = brdf;
      }
    });
  }

  HOST_DEVICE inline const Eigen::Array3f &emission() const {
    return emission_;
  }

  template <typename F> HOST_DEVICE auto visit(const F &f) {
    switch (type_) {
    case BRDFType::Diffuse:
      return f(diffuse_);
    case BRDFType::Glossy:
      return f(glossy_);
    case BRDFType::Mirror:
      return f(mirror_);
    case BRDFType::DielectricRefractive:
    default:
      return f(dielectric_refractive_);
    }
  }

  template <typename F> HOST_DEVICE auto visit(const F &f) const {
    switch (type_) {
    case BRDFType::Diffuse:
      return f(diffuse_);
    case BRDFType::Glossy:
      return f(glossy_);
    case BRDFType::Mirror:
      return f(mirror_);
    case BRDFType::DielectricRefractive:
    default:
      return f(dielectric_refractive_);
    }
  }

  HOST_DEVICE bool has_non_delta_samples() const {
    return visit([](const auto &v) {
      return std::decay_t<decltype(v)>::has_non_delta_samples;
    });
  }

  HOST_DEVICE bool is_bsdf() const {
    return visit(
        [](const auto &v) { return std::decay_t<decltype(v)>::is_bsdf; });
  }

  HOST_DEVICE Eigen::Array3f brdf(const Eigen::Vector3f &incoming_dir,
                                  const Eigen::Vector3f &outgoing_dir,
                                  const Eigen::Vector3f &normal) const {
    return visit([&](const auto &v) -> Eigen::Array3f {
      if constexpr (std::decay_t<decltype(v)>::has_non_delta_samples) {
        return v.brdf(incoming_dir, outgoing_dir, normal);
      } else {
        assert(false);
        return Eigen::Array3f::Zero();
      }
    });
  }

  HOST_DEVICE float prob_delta() const {
    return visit([&](const auto &v) -> float {
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

  HOST_DEVICE bool delta_prob_check(rng::Rng &rng) const {
    return visit([&](const auto &v) -> bool {
      using T = std::decay_t<decltype(v)>;
      if constexpr (T::has_non_delta_samples && !T::has_delta_samples) {
        return false;
      } else if constexpr (T::has_delta_samples && !T::has_non_delta_samples) {
        return true;
      } else {
        return rng.sample_1() < v.prob_delta();
      }
    });
  }

  HOST_DEVICE std::tuple<Eigen::Vector3f, Eigen::Array3f>
  delta_sample(rng::Rng &rng, const Eigen::Vector3f &incoming_dir,
               const Eigen::Vector3f &normal) const {
    return visit(
        [&](const auto &v) -> std::tuple<Eigen::Vector3f, Eigen::Array3f> {
          if constexpr (std::decay_t<decltype(v)>::has_delta_samples) {
            return v.delta_sample(rng, incoming_dir, normal);
          } else {
            assert(false);
            return {};
          }
        });
  }

  HOST_DEVICE render::DirSample sample(rng::Rng &rng,
                                       const Eigen::Vector3f &incoming_dir,
                                       const Eigen::Vector3f &normal) const {
    return visit([&](const auto &v) -> render::DirSample {
      if constexpr (std::decay_t<decltype(v)>::has_non_delta_samples) {
        return v.sample(rng, incoming_dir, normal);
      } else {
        assert(false);
        return {};
      }
    });
  }

private:
  HOST_DEVICE void copy_in_other(const Material &other) {
    type_ = other.type_;
    emission_ = other.emission_;
    switch (type_) {
    case BRDFType::Diffuse:
      diffuse_ = other.diffuse_;
      break;
    case BRDFType::Glossy:
      glossy_ = other.glossy_;
      break;
    case BRDFType::Mirror:
      mirror_ = other.mirror_;
      break;
    case BRDFType::DielectricRefractive:
    default:
      dielectric_refractive_ = other.dielectric_refractive_;
      break;
    }
  }

  BRDFType type_;
  Eigen::Array3f emission_;

  union {
    BRDF<BRDFType::Diffuse> diffuse_;
    BRDF<BRDFType::Glossy> glossy_;
    BRDF<BRDFType::Mirror> mirror_;
    BRDF<BRDFType::DielectricRefractive> dielectric_refractive_;
  };
};
} // namespace material
