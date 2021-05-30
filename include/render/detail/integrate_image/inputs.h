#pragma once

#include "kernel/work_division.h"
#include "render/detail/integrate_image/items.h"
#include "render/kernel_approach_settings.h"
#include "render/renderer.h"

namespace render {
namespace detail {
namespace integrate_image {
template <ExactSpecializationOf<Items> Items> struct Inputs {
  const Items &items;

  OutputType output_type;
  SampleSpec sample_spec;
  unsigned samples_per;

  // TODO: this might need to change later
  Span<const typename Items::InfoType> idx_to_info;

  bool show_progress;
  bool show_times;
};
} // namespace integrate_image
} // namespace detail
} // namespace render
