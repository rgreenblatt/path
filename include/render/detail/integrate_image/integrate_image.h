#pragma once

#include "execution_model/execution_model.h"
#include "intersect/intersectable.h"
#include "render/detail/integrate_image/inputs.h"

namespace render {
namespace detail {
namespace integrate_image {
template <ExecutionModel exec> struct IntegrateImage {
  template <ExactSpecializationOf<Items> Items, intersect::Intersectable I>
  requires std::same_as<typename Items::InfoType, typename I::InfoType>
  static void run(Inputs<Items> inp, const I &intersectable);
};
} // namespace integrate_image
} // namespace detail
} // namespace render
