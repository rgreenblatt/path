#pragma once

#include "integrate/rendering_equation.h"
#include "kernel/grid_location_info.h"
#include "kernel/location_info.h"
#include "kernel/work_division.h"
#include "lib/attribute.h"
#include "render/detail/initial_ray_sample.h"
#include "render/detail/integrate_image_items.h"

namespace render {
namespace detail {
template <ExactSpecializationOf<IntegrateImageItems> Items,
          intersect::Intersectable I>
ATTR_NO_DISCARD_PURE HOST_DEVICE inline Eigen::Array3f
integrate_pixel(const Items &items, const I &intersectable,
                const kernel::WorkDivision &division,
                const integrate::RenderingEquationSettings &settings,
                const kernel::GridLocationInfo &info) {
  auto initial_ray_sampler = [&](auto &rng) {
    return initial_ray_sample(rng, info.x, info.y, division.x_dim(),
                              division.y_dim(), items.film_to_world);
  };

  return integrate::rendering_equation(
      kernel::LocationInfo::from_grid_location_info(info, division.x_dim()),
      settings, initial_ray_sampler, items.rng, intersectable,
      items.components);
}
} // namespace detail
} // namespace render
