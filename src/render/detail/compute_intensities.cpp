#include "render/detail/impl/compute_intensities_impl.h"
#include "render/detail/impl/render.h"

namespace render {
namespace detail {
template <intersect::accel::AccelRef A, LightSamplerRef L, DirSamplerRef D,
          TermProbRef T, rng::RngRef R>
void compute_intensities(const WorkDivision &division, unsigned samples_per,
                         unsigned x_dim, unsigned y_dim, unsigned block_size,
                         const A &accel, const L &light_sampler,
                         const D &direction_sampler, const T &term_prob,
                         const R &rng, Span<BGRA> pixels,
                         Span<Eigen::Array3f> intensities,
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
                                   direction_sampler, term_prob, rng,
                                   triangle_data, materials, film_to_world) /
          samples_per);
    }
  }
}

template class RendererImpl<ExecutionModel::CPU>;
} // namespace detail
} // namespace render
