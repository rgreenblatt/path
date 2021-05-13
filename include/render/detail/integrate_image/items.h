#pragma once

#include "integrate/rendering_equation_components.h"
#include "integrate/rendering_equation_settings.h"
#include "meta/pack_element.h"
#include "render/detail/integrate_image/base_items.h"
#include "rng/rng.h"

#include <Eigen/Geometry>

namespace render {
namespace detail {
namespace integrate_image {
// allow for named arguments and avoid some repeated definition code
template <ExactSpecializationOf<integrate::RenderingEquationComponents> CIn,
          rng::RngRef RIn>
struct Items {
  using C = CIn;
  using R = RIn;
  using InfoType = typename C::InfoType;

  integrate::RenderingEquationSettings render_settings;
  C components;
  R rng;
};
} // namespace integrate_image
} // namespace detail
} // namespace render
