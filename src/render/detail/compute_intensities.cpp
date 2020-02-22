#include "render/detail/impl/compute_intensities_impl.h"
#include "render/detail/impl/render.h"

namespace render {
namespace detail {
template <ExecutionModel execution_model, typename Accel, typename LightSampler,
          typename DirSampler, typename TermProb>
void compute_intensities(const WorkDivision &, unsigned samples_per,
                         unsigned x_dim, unsigned y_dim, unsigned,
                         const Accel &accel, const LightSampler &light_sampler,
                         const DirSampler &direction_sampler,
                         const TermProb &term_prob, Span<BGRA> pixels,
                         Span<Eigen::Array3f>,
                         Span<const scene::TriangleData> triangle_data,
                         Span<const material::Material> materials,
                         const Eigen::Affine3f &film_to_world) {
#ifdef NDEBUG
#pragma omp parallel for collapse(2)
#endif
  for (unsigned y = 0; y < y_dim; y++) {
    for (unsigned x = 0; x < x_dim; x++) {
      pixels[x + y * x_dim] = intensity_to_bgr(
          compute_intensities_impl(x, y, 0, samples_per, x_dim, y_dim,
                                   samples_per, accel, light_sampler,
                                   direction_sampler, term_prob, triangle_data,
                                   materials, film_to_world) /
          samples_per);
    }
  }
}

template class RendererImpl<ExecutionModel::CPU>;
} // namespace detail
} // namespace render
