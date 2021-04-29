#pragma once

#include "execution_model/execution_model.h"
#include "intersect/intersectable.h"
#include "render/detail/integrate_image/inputs.h"
#include "render/detail/integrate_image/items.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace mega_kernel {
template <ExecutionModel exec> struct Run {
  template <ExactSpecializationOf<Items> Items, intersect::Intersectable I>
  requires std::same_as<typename Items::InfoType, typename I::InfoType>
  static ExecVector<exec, FloatRGB> *
  run(Inputs<Items> inp, const I &intersectable,
      const MegaKernelSettings &settings, ExecVector<exec, FloatRGB> &float_rgb,
      ExecVector<exec, FloatRGB> &reduced_float_rgb);
};
} // namespace mega_kernel
} // namespace integrate_image
} // namespace detail
} // namespace render
