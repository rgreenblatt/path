#pragma once

#include "lib/integer_division_utils.h"
#include "render/detail/assign_output.h"
#include "render/detail/integrate_image.h"
#include "render/detail/integrate_pixel.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <> template <ExactSpecializationOf<IntegrateImageInputs> Inp>
requires Inp::I::individually_intersectable void
IntegrateImage<ExecutionModel::CPU>::run_individual(Inp inp) {
  ProgressBar progress_bar(inp.division.x_dim() * inp.division.y_dim(), 70);
  if (inp.show_progress) {
    progress_bar.display();
  }
#ifdef NDEBUG
// TODO: better scheduling approach?  depend on number of samples?
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
#endif
  for (unsigned y = 0; y < inp.division.y_dim(); y++) {
    for (unsigned x = 0; x < inp.division.x_dim(); x++) {
      assign_output(
          inp.items.base, inp.division, 0, 1, x, y,
          integrate_pixel(inp.items, inp.intersector, inp.division,
                          inp.settings.rendering_equation_settings,
                          work_division::GridLocationInfo{
                              .start_sample = 0,
                              .end_sample = inp.items.base.samples_per,
                              .x = x,
                              .y = y,
                          }));
      if (inp.show_progress) {
#pragma omp critical
        {
          ++progress_bar;
          progress_bar.display();
        }
      }
    }
  }
  if (inp.show_progress) {
    progress_bar.done();
  }
}
} // namespace detail
} // namespace render
