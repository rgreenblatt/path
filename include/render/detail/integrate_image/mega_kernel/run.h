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
  static Output run(Inputs<Items> inp, const I &intersectable,
                    const MegaKernelSettings &settings,
                    ExecVector<exec, BGRA32> &bgra_32,
                    std::array<ExecVector<exec, FloatRGB>, 2> &float_rgb,
                    std::array<HostVector<ExecVector<exec, FloatRGB>>, 2>
                        &output_per_step_rgb);
};
} // namespace mega_kernel
} // namespace integrate_image
} // namespace detail
} // namespace render
