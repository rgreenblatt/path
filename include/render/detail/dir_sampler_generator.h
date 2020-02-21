#pragma once

#include "lib/execution_model/execution_model.h"
#include "lib/projection.h"
#include "material/material.h"
#include "render/dir_sampler_type.h"
#include "rng/rng.h"

#include <Eigen/Core>

namespace render {
namespace detail {
template <ExecutionModel execution_model, DirSamplerType type>
class DirSamplerGenerator;

template <ExecutionModel execution_model>
class DirSamplerGenerator<execution_model, DirSamplerType::Uniform> {
public:
  using Settings = DirSamplerSettings<DirSamplerType::Uniform>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    HOST_DEVICE DirSample operator()(const Eigen::Vector3f &,
                                     const material::Material &material,
                                     const Eigen::Vector3f &normal,
                                     const Eigen::Vector3f &,
                                     rng::Rng &rng) const {
      auto [v0, v1] = rng.sample_2();

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
class DirSamplerGenerator<execution_model, DirSamplerType::BRDF> {
public:
  using Settings = DirSamplerSettings<DirSamplerType::BRDF>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    HOST_DEVICE DirSample operator()(const Eigen::Vector3f &,
                                     const material::Material &material,
                                     const Eigen::Vector3f &normal,
                                     const Eigen::Vector3f &direction,
                                     rng::Rng &rng) const {
      return material.sample(rng, direction, normal);
    }
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};
} // namespace detail
} // namespace render
