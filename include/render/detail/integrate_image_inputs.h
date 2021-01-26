#pragma once

#include "kernel/work_division.h"
#include "render/detail/integrate_image_items.h"
#include "render/general_settings.h"

namespace render {
using kernel::WorkDivision;
namespace detail {
template <ExactSpecializationOf<IntegrateImageItems> Items>
struct IntegrateImageInputs {
  const Items &items;
  const WorkDivision &division;
  const GeneralSettings &settings;
  bool show_progress;
};
} // namespace detail
} // namespace render
