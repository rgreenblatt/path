#pragma once

#include "lib/execution_model/execution_model.h"
#include "render/detail/rng.h"
#include "render/light_sampler_type.h"
#include "scene/material.h"

#include <Eigen/Core>

#include <array>

namespace render {
namespace detail {
template <ExecutionModel execution_model, LightSamplerType type>
class LightSamplerGenerator;

struct LightSample {
  Eigen::Vector3f direction;
  float prob;
};

template <ExecutionModel execution_model>
class LightSamplerGenerator<execution_model,
                            LightSamplerType::NoDirectLighting> {
public:
  class Ref {
  public:
    // TODO
    HOST_DEVICE auto operator()(const Eigen::Vector3f &point,
                                const scene::Material &material,
                                const Eigen::Vector3f &normal,
                                const Eigen::Vector3f &direction,
                                Rng &rng) const {
      return std::array<LightSample, 0>{};
    }

    static const bool performs_samples = false;
  };

  auto gen(LightSamplerSettings<LightSamplerType::NoDirectLighting>) {
    return Ref();
  }
};

template <ExecutionModel execution_model>
class LightSamplerGenerator<execution_model, LightSamplerType::WeightedAABB> {
public:
  class Ref {
  public:
    // TODO
    HOST_DEVICE auto operator()(const Eigen::Vector3f &point,
                                const scene::Material &material,
                                const Eigen::Vector3f &normal,
                                const Eigen::Vector3f &direction,
                                Rng &rng) const {
      return std::array<LightSample, 0>{};
    }

    static const bool performs_samples = false;
  };

  auto gen(LightSamplerSettings<LightSamplerType::WeightedAABB>) {
    return Ref();
  }
};
} // namespace detail
} // namespace render
