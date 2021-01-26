#pragma once

#include "execution_model/execution_model.h"
#include "intersect/intersectable.h"
#include "render/detail/integrate_image_inputs.h"

namespace render {
namespace detail {
template <ExecutionModel exec> struct IntegrateImageIndividual {
  template <ExactSpecializationOf<IntegrateImageItems> Items,
            intersect::Intersectable I>
  requires std::same_as<typename Items::InfoType, typename I::InfoType>
  static void run(IntegrateImageInputs<Items> inp, const I &intersectable);
};
} // namespace detail
} // namespace render
