#pragma once

#include "execution_model/execution_model.h"
#include "lib/projection.h"
#include "material/material.h"
#include "render/dir_sampler.h"
#include "rng/rng.h"
#include "rng/test_rng_state_type.h"

#include <Eigen/Core>

namespace render {
namespace detail {
template <DirSamplerType type, ExecutionModel execution_model>
struct DirSamplerImpl;

template <typename V>
concept DirSamplerRef = requires(const V &dir_sampler,
                                 const Eigen::Vector3f &position,
                                 const material::Material &material,
                                 const Eigen::Vector3f &incoming_dir,
                                 const Eigen::Vector3f &normal,
                                 rng::TestRngStateT &rng) {
  { dir_sampler(position, material, incoming_dir, normal, rng) }
  ->std::common_with<DirSample>;
};

template <DirSamplerType type, ExecutionModel execution_model>
concept DirSampler = requires {
  typename DirSamplerSettings<type>;
  typename DirSamplerImpl<type, execution_model>;

  requires requires(DirSamplerImpl<type, execution_model> & dir_sampler,
                    const DirSamplerSettings<type> &settings) {
    { dir_sampler.gen(settings) }
    ->DirSamplerRef;
  };
};

template <DirSamplerType type, ExecutionModel execution_model>
requires DirSampler<type, execution_model> struct DirSamplerT
    : DirSamplerImpl<type, execution_model> {
  using DirSamplerImpl<type, execution_model>::DirSamplerImpl;
};

template <ExecutionModel execution_model>
struct DirSamplerImpl<DirSamplerType::Uniform, execution_model> {
public:
  using Settings = DirSamplerSettings<DirSamplerType::Uniform>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    template <rng::RngState R>
    HOST_DEVICE DirSample operator()(const Eigen::Vector3f &,
                                     const material::Material &material,
                                     const Eigen::Vector3f &,
                                     const Eigen::Vector3f &normal,
                                     R &rng) const {
      float v0 = rng.next();
      float v1 = rng.next();

      bool need_whole_sphere = material.is_bsdf();

      float phi = 2 * M_PI * v0;
      float theta = std::acos(need_whole_sphere ? 2 * v1 - 1 : v1);

      auto direction = find_relative_vec(normal, phi, theta);

      assert(need_whole_sphere || direction.dot(normal) >= 0);
      assert(need_whole_sphere ||
             std::abs(direction.dot(normal) - std::cos(theta)) < 1e-4);

      return DirSample{direction,
                       1 / float((need_whole_sphere ? 4 : 2) * M_PI)};
    }
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};

template <ExecutionModel execution_model>
struct DirSamplerImpl<DirSamplerType::BRDF, execution_model> {
public:
  using Settings = DirSamplerSettings<DirSamplerType::BRDF>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    template <rng::RngState R>
    HOST_DEVICE DirSample operator()(const Eigen::Vector3f &,
                                     const material::Material &material,
                                     const Eigen::Vector3f &direction,
                                     const Eigen::Vector3f &normal,
                                     R &rng) const {
      return material.sample(direction, normal, rng);
    }
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};
} // namespace detail
} // namespace render
