#pragma once

#include "execution_model/execution_model.h"
#include "intersectable_scene/intersector.h"
#include "render/detail/integrate_image_items.h"
#include "render/general_settings.h"
#include "work_division/work_division.h"

namespace render {
using work_division::WorkDivision;
namespace detail {
template <
    ExactSpecializationOf<IntegrateImageItems> Items,
    intersectable_scene::IntersectorForInfoType<typename Items::InfoType> IIn>
struct IntegrateImageInputs {
  using I = IIn;

  const Items &items;
  I &intersector;
  const WorkDivision &division;
  const GeneralSettings &settings;
  bool show_progress;
};

template <ExecutionModel exec> class IntegrateImage {
public:
  template <ExactSpecializationOf<IntegrateImageInputs> Inp>
  static void run(Inp inp) {
    if constexpr (Inp::I::individually_intersectable) {
      IntegrateImage::run_individual(inp);
    } else {
      IntegrateImage::run_bulk(inp);
    }
  }

private:
  template <ExactSpecializationOf<IntegrateImageInputs> Inp>
  requires Inp::I::individually_intersectable static void
  run_individual(Inp inp);

  template <ExactSpecializationOf<IntegrateImageInputs> Inp>
  requires(!Inp::I::individually_intersectable) static void run_bulk(Inp inp) {
    unreachable();
  }
};
} // namespace detail
} // namespace render
