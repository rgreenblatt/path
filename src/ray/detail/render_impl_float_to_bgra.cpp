#include "ray/detail/impl/float_to_bgra.h"
#include "ray/detail/render_impl.h"

namespace ray {
namespace detail {
template <ExecutionModel execution_model>
void RendererImpl<execution_model>::float_to_bgra(
    BGRA *pixels, Span<const scene::Color> colors) {
  auto bgra_span = Span(pixels, real_x_dim_ * real_y_dim_);

  const auto start_convert = current_time();

#pragma omp parallel for collapse(2) schedule(dynamic, 16)
  for (unsigned x = 0; x < real_x_dim_; x++) {
    for (unsigned y = 0; y < real_y_dim_; y++) {
      float_to_bgra_impl(x, y, real_x_dim_, real_y_dim_, super_sampling_rate_,
                         colors, bgra_span);
    }
  }

  const auto end_convert = current_time();
  double convert_duration = to_secs(start_convert, end_convert);

  if (show_times_) {
    dbg(convert_duration);
  }
}

template class RendererImpl<ExecutionModel::CPU>;
} // namespace detail
} // namespace ray
