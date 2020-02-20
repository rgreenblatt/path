#include "lib/info/timer.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/tone_map.h"

#include <thrust/copy.h>

namespace render {
namespace detail {
template <ExecutionModel execution_model>
RendererImpl<execution_model>::RendererImpl() {}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(
    Span<RGBA> pixels, const scene::Scene &s, unsigned samples_per,
    unsigned x_dim, unsigned y_dim, PerfSettings settings, bool show_times) {
  if (samples_per > std::numeric_limits<uint16_t>::max()) {
    std::cerr << "more samples than allowed" << std::endl;
    return;
  }

  dispatch_compute_intensities(s, samples_per, x_dim, y_dim, settings,
                               show_times);

  SpanSized<RGBA> output_pixels;
  if constexpr (execution_model == ExecutionModel::GPU) {
    bgra_.resize(x_dim * y_dim);
    output_pixels = bgra_;
  } else {
    output_pixels = SpanSized<RGBA>(pixels.data(), x_dim * y_dim);
  }

  tone_map<execution_model>(intensities_, output_pixels);

  if constexpr (execution_model == ExecutionModel::GPU) {
    thrust::copy(bgra_.begin(), bgra_.end(), pixels.begin());
  }
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace render
