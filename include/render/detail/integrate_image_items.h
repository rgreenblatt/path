#pragma once

#include "integrate/rendering_equation_components.h"
#include "meta/pack_element.h"
#include "render/detail/integrate_image_base_items.h"
#include "rng/rng.h"

#include <Eigen/Geometry>

namespace render {
namespace detail {
// allow for named arguments and avoid some repeated definition code
template <ExactSpecializationOf<integrate::RenderingEquationComponents> C,
          rng::RngRef R>
struct IntegrateImageItems {
  IntegrateImageBaseItems base;
  C components;
  R rng;
  Eigen::Affine3f film_to_world;
  using InfoType = typename C::InfoType;
};
} // namespace detail
} // namespace render
