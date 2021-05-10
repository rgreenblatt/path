#pragma once

#include "execution_model/execution_model.h"
#include "intersectable_scene/intersector.h"
#include "render/detail/integrate_image/inputs.h"
#include "render/detail/integrate_image/streaming/state.h"
#include "render/streaming_settings.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace streaming {
template <ExecutionModel exec> struct Run {
  template <ExactSpecializationOf<Items> Items,
            intersectable_scene::BulkIntersector I>
  requires std::same_as<typename Items::InfoType, typename I::InfoType>
  static Output
  run(Inputs<Items> inp, I &intersector,
      State<exec, Items::C::max_num_light_samples(), typename Items::R> &state,
      const StreamingSettings &settings, ExecVector<exec, BGRA32> &bgra_32,
      ExecVector<exec, FloatRGB> &float_rgb,
      HostVector<ExecVector<exec, FloatRGB>> &output_per_step_rgb);
};
} // namespace streaming
} // namespace integrate_image
} // namespace detail
} // namespace render
