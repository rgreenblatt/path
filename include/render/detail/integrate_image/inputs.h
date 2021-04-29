#pragma once

#include "kernel/work_division.h"
#include "render/detail/integrate_image/items.h"
#include "render/kernel_approach_settings.h"

namespace render {
namespace detail {
namespace integrate_image {
template <ExactSpecializationOf<Items> Items> struct Inputs {
  const Items &items;

  bool output_as_bgra_32;
  Span<BGRA32> bgra_32;

  unsigned samples_per;
  unsigned x_dim;
  unsigned y_dim;

  bool show_progress;
};
} // namespace integrate_image
} // namespace detail
} // namespace render
