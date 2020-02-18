#include "ray/detail/impl/float_to_bgra.h"
#include "lib/timer.h"
#include "ray/detail/render_impl.h"

namespace ray {
namespace detail {
template <ExecutionModel execution_model>
void RendererImpl<execution_model>::float_to_bgra(
    BGRA *pixels, Span<const scene::Color> colors) {
  auto bgra_span = Span(pixels, real_x_dim_ * real_y_dim_);

  Timer convert_to_bgra_timer;

#pragma omp parallel for collapse(2) schedule(dynamic, 16)
  for (unsigned x = 0; x < real_x_dim_; x++) {
    for (unsigned y = 0; y < real_y_dim_; y++) {
      float_to_bgra_impl(x, y, real_x_dim_, real_y_dim_, super_sampling_rate_,
                         colors, bgra_span);
    }
  }

  if (show_times_) {
    convert_to_bgra_timer.report("convert to bgra");
  }
}

template class RendererImpl<ExecutionModel::CPU>;
} // namespace detail
} // namespace ray
