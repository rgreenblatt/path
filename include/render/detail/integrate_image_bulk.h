#pragma once

#include "execution_model/execution_model.h"
#include "intersectable_scene/intersector.h"
#include "render/detail/integrate_image_bulk_state.h"
#include "render/detail/integrate_image_inputs.h"

namespace render {
namespace detail {
template <ExecutionModel exec> struct IntegrateImageBulk {
  template <ExactSpecializationOf<IntegrateImageItems> Items,
            intersectable_scene::BulkIntersector I>
  requires std::same_as<typename Items::InfoType, typename I::InfoType>
  static void run(IntegrateImageInputs<Items> inp, I &intersector,
                  IntegrateImageBulkState<exec, Items::C::L::max_num_samples,
                                          typename Items::R> &state);
};
} // namespace detail
} // namespace render