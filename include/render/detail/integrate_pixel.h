#pragma once

#include "integrate/rendering_equation.h"
#include "lib/attribute.h"
#include "render/detail/initial_ray_sample.h"
#include "render/detail/integrate_image_items.h"
#include "work_division/grid_location_info.h"
#include "work_division/location_info.h"
#include "work_division/work_division.h"

namespace render {
namespace detail {
template <ExactSpecializationOf<IntegrateImageItems> Items,
          intersect::Intersectable I>
ATTR_NO_DISCARD_PURE HOST_DEVICE inline Eigen::Array3f
integrate_pixel(const Items &items, const I &intersectable,
                const work_division::WorkDivision &division,
                const integrate::RenderingEquationSettings &settings,
                const work_division::GridLocationInfo &info) {
  auto initial_ray_sampler = [&](auto &rng) {
    return initial_ray_sample(rng, info.x, info.y, division.x_dim(),
                              division.y_dim(), items.film_to_world);
  };

  return integrate::rendering_equation(
      work_division::LocationInfo::from_grid_location_info(info,
                                                           division.x_dim()),
      settings, initial_ray_sampler, items.rng, intersectable,
      items.components);
}
} // namespace detail
} // namespace render
