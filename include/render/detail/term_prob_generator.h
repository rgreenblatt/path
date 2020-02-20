#pragma once

#include "lib/execution_model/execution_model.h"
#include "render/detail/rng.h"
#include "render/term_prob_type.h"
#include "scene/material.h"

#include <Eigen/Core>

namespace render {
namespace detail {
template <ExecutionModel execution_model, TermProbType type>
class TermProbGenerator;

template <ExecutionModel execution_model>
class TermProbGenerator<execution_model, TermProbType::Uniform> {
public:
  class Ref {
  public:
    // TODO
    HOST_DEVICE float operator()(const Eigen::Vector3f &point,
                                 const scene::Material &material,
                                 const Eigen::Vector3f &normal,
                                 const Eigen::Vector3f &direction,
                                 Rng &rng) const {}
  };

  auto gen(TermProbSettings<TermProbType::Uniform>) { return Ref(); }
};

template <ExecutionModel execution_model>
class TermProbGenerator<execution_model, TermProbType::MultiplierNorm> {
public:
  class Ref {
  public:
    // TODO
    HOST_DEVICE float operator()(const Eigen::Vector3f &point,
                                 const scene::Material &material,
                                 const Eigen::Vector3f &normal,
                                 const Eigen::Vector3f &direction,
                                 Rng &rng) const {}
  };

  auto gen(TermProbSettings<TermProbType::MultiplierNorm>) { return Ref(); }
};
} // namespace detail
} // namespace render
