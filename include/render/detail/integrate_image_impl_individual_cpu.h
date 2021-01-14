#pragma once

#include "lib/integer_division_utils.h"
#include "render/detail/assign_output.h"
#include "render/detail/integrate_image.h"
#include "render/detail/integrate_pixel.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <>
template <typename... T>
void IntegrateImage<ExecutionModel::CPU>::run(
    IntegrateImageIndividualInputs<T...> inp) {
  auto val = inp.val;

  ProgressBar progress_bar(val.division.x_dim() * val.division.y_dim(), 70);
  if (val.show_progress) {
    progress_bar.display();
  }
#ifdef NDEBUG
// TODO: better scheduling approach?  depend on number of samples?
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
#endif
  for (unsigned y = 0; y < val.division.y_dim(); y++) {
    for (unsigned x = 0; x < val.division.x_dim(); x++) {
      assign_output(
          val.items.base, val.division, 0, 1, x, y,
          integrate_pixel(val.items, val.intersector, val.division,
                          val.settings.rendering_equation_settings,
                          kernel::GridLocationInfo{
                              .start_sample = 0,
                              .end_sample = val.items.base.samples_per,
                              .x = x,
                              .y = y,
                          }));
      if (val.show_progress) {
#pragma omp critical
        {
          ++progress_bar;
          progress_bar.display();
        }
      }
    }
  }
  if (val.show_progress) {
    progress_bar.done();
  }
}
} // namespace detail
} // namespace render
