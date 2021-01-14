#pragma once

#include "integrate/rendering_equation_components.h"
#include "meta/pack_element.h"
#include "render/detail/integrate_image_base_items.h"
#include "rng/rng.h"

#include <Eigen/Geometry>

namespace render {
namespace detail {
// allow for named arguments and avoid some repeated definition code
template <ExactSpecializationOf<integrate::RenderingEquationComponents> CIn,
          rng::RngRef RIn>
struct IntegrateImageItems {
  using C = CIn;
  using R = RIn;
  using InfoType = typename C::InfoType;

  IntegrateImageBaseItems base;
  C components;
  R rng;
  Eigen::Affine3f film_to_world;
};
} // namespace detail
} // namespace render
