#include "lib/integer_division_utils.h"
#include "render/detail/assign_output.h"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image.h"
#include "render/detail/integrate_pixel.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <> template <ExactSpecializationOf<IntegrateImageInputs> Inp>
requires Inp::I::individually_intersectable void
IntegrateImage<ExecutionModel::CPU>::run_individual(Inp inp) {
  ProgressBar progress_bar(inp.items.base.x_dim * inp.items.base.y_dim, 70);
  if (inp.show_progress) {
    progress_bar.display();
  }
#ifdef NDEBUG
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
#endif
  for (unsigned y = 0; y < inp.items.base.y_dim; y++) {
    for (unsigned x = 0; x < inp.items.base.x_dim; x++) {
      assign_output(inp.items.base, 0, 1, x, y,
                    integrate_pixel(
                        work_division::GridLocationInfo{
                            .start_sample = 0,
                            .end_sample = inp.items.base.samples_per,
                            .x = x,
                            .y = y,
                        },
                        inp.settings.rendering_equation_settings,
                        inp.intersector, inp.items));
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

template class Renderer::Impl<ExecutionModel::CPU>;
} // namespace render
