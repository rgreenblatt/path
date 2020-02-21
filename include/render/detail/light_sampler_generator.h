#pragma once

#include "lib/execution_model/execution_model.h"
#include "material/material.h"
#include "render/light_sampler_type.h"
#include "rng/rng.h"

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
  using Settings = LightSamplerSettings<LightSamplerType::NoDirectLighting>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    HOST_DEVICE auto operator()(const Eigen::Vector3f &,
                                const material::Material &,
                                const Eigen::Vector3f &,
                                const Eigen::Vector3f &, rng::Rng &) const {
      return std::array<LightSample, 0>{};
    }

    static const bool performs_samples = false;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};

template <ExecutionModel execution_model>
class LightSamplerGenerator<execution_model, LightSamplerType::WeightedAABB> {
public:
  using Settings = LightSamplerSettings<LightSamplerType::WeightedAABB>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    HOST_DEVICE auto operator()(const Eigen::Vector3f &,
                                const material::Material &,
                                const Eigen::Vector3f &,
                                const Eigen::Vector3f &, rng::Rng &) const {
      assert(false);

      return std::array<LightSample, 0>{};
    }

    static const bool performs_samples = true;
  };

  auto gen(const Settings &settings) { return Ref(settings); }
};
} // namespace detail
} // namespace render
