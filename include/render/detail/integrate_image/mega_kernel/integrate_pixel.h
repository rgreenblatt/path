#pragma once

#include "integrate/rendering_equation.h"
#include "kernel/grid_location_info.h"
#include "kernel/location_info.h"
#include "kernel/work_division.h"
#include "lib/attribute.h"
#include "render/detail/integrate_image/initial_ray_sample.h"
#include "render/detail/integrate_image/items.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace mega_kernel {
template <bool output_per_step, ExactSpecializationOf<Items> Items,
          intersect::Intersectable I,
          integrate::Sampler<typename I::InfoType> Sampler>
ATTR_NO_DISCARD_PURE
    HOST_DEVICE inline std::conditional_t<output_per_step, void, FloatRGB>
    integrate_pixel(
        const Items &items, const I &intersectable,
        const kernel::WorkDivision &division,
        const kernel::GridLocationInfo &info, const Sampler &sampler,
        std::conditional_t<output_per_step, SpanSized<FloatRGB>, std::tuple<>>
            step_outputs) {
  return integrate::rendering_equation<output_per_step>(
      kernel::LocationInfo::from_grid_location_info(info, division.x_dim()),
      items.render_settings, sampler, items.rng, intersectable,
      items.components, step_outputs);
}
} // namespace mega_kernel
} // namespace integrate_image
} // namespace detail
} // namespace render
