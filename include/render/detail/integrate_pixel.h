#pragma once

#include "integrate/rendering_equation.h"
#include "lib/attribute.h"
#include "render/detail/initial_ray_sample.h"
#include "render/detail/integrate_image_items.h"
#include "work_division/grid_location_info.h"
#include "work_division/location_info.h"

namespace render {
namespace detail {
template <intersect::Intersectable I,
          ExactSpecializationOf<IntegrateImageItems> Items>
ATTR_NO_DISCARD_PURE HOST_DEVICE inline Eigen::Array3f
integrate_pixel(const work_division::GridLocationInfo &info,
                const integrate::RenderingEquationSettings settings,
                const I &intersectable, const Items &items) {
  auto initial_ray_sampler = [&](auto &rng) {
    return initial_ray_sample(rng, info.x, info.y, items.base.x_dim,
                              items.base.y_dim, items.film_to_world);
  };

  return integrate::rendering_equation(
      work_division::LocationInfo::from_grid_location_info(info,
                                                           items.base.x_dim),
      settings, initial_ray_sampler, items.rng, intersectable,
      items.components);
}
} // namespace detail
} // namespace render
