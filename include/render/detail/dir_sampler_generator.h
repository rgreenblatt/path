#pragma once

#include "lib/execution_model/execution_model.h"
#include "render/detail/rng.h"
#include "render/dir_sampler_type.h"
#include "scene/material.h"

#include <Eigen/Core>

namespace render {
namespace detail {
template <ExecutionModel execution_model, DirSamplerType type>
class DirSamplerGenerator;

struct DirSample {
  Eigen::Vector3f direction;
  float prob;
};

template <ExecutionModel execution_model>
class DirSamplerGenerator<execution_model, DirSamplerType::Uniform> {
public:
  class Ref {
  public:
    // TODO
    HOST_DEVICE DirSample operator()(const Eigen::Vector3f &point,
                                     const scene::Material &material,
                                     const Eigen::Vector3f &normal,
                                     const Eigen::Vector3f &direction,
                                     Rng &rng) const {}
  };

  auto gen(DirSamplerSettings<DirSamplerType::Uniform>) { return Ref(); }
};

template <ExecutionModel execution_model>
class DirSamplerGenerator<execution_model, DirSamplerType::BRDF> {
public:
  class Ref {
  public:
    // TODO
    HOST_DEVICE DirSample operator()(const Eigen::Vector3f &point,
                                     const scene::Material &material,
                                     const Eigen::Vector3f &normal,
                                     const Eigen::Vector3f &direction,
                                     Rng &rng) const {}
  };

  auto gen(DirSamplerSettings<DirSamplerType::BRDF>) { return Ref(); }
};
} // namespace detail
} // namespace render
