#pragma once

#include "execution_model/execution_model.h"
#include "execution_model/execution_model_vector_type.h"
#include "intersectable_scene/intersector.h"
#include "kernel/work_division.h"
#include "render/detail/integrate_image_bulk_state.h"
#include "render/detail/integrate_image_items.h"
#include "render/general_settings.h"

#include <type_traits>

namespace render {
using kernel::WorkDivision;
namespace detail {
namespace integrate_image {
namespace detail {
template <ExactSpecializationOf<IntegrateImageItems> ItemsIn, typename IIn,
          typename State>
requires std::same_as<typename ItemsIn::InfoType,
                      typename std::decay_t<IIn>::InfoType>
struct IntegrateImageInputs {
  using I = std::decay_t<IIn>;
  using Items = ItemsIn;

  const Items &items;
  [[no_unique_address]] IIn intersector;
  const WorkDivision &division;
  const GeneralSettings &settings;
  bool show_progress;
  [[no_unique_address]] State state;
};
} // namespace detail
} // namespace integrate_image

struct IntegrateImageEmptyState {};

// can't use alias because of this:
// https://stackoverflow.com/questions/30707011/pack-expansion-for-alias-template
template <typename Items, intersect::Intersectable I>
struct IntegrateImageIndividualInputs {
  integrate_image::detail::IntegrateImageInputs<Items, const I &,
                                                IntegrateImageEmptyState>
      val;
};

template <ExecutionModel exec, typename Items,
          intersectable_scene::BulkIntersector I>
struct IntegrateImageBulkInputs {
  integrate_image::detail::IntegrateImageInputs<
      Items, I &,
      IntegrateImageBulkState<exec, Items::C::L::max_num_samples,
                              typename Items::R> &>
      val;
};

template <ExecutionModel exec> struct IntegrateImage {
  template <typename... T>
  static void run(IntegrateImageIndividualInputs<T...> inp);

  template <typename... T>
  static void run(IntegrateImageBulkInputs<exec, T...> inp);
};
} // namespace detail
} // namespace render
