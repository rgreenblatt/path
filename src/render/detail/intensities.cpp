#include "render/detail/impl/intensities_impl.h"
#include "render/detail/impl/render_impl.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <intersect::accel::AccelRef MeshAccel,
          intersect::accel::AccelRef TriAccel, LightSamplerRef L,
          DirSamplerRef D, TermProbRef T, rng::RngRef R>
void intensities(const GeneralSettings &settings, bool show_progress,
                 const WorkDivision &, unsigned samples_per, unsigned x_dim,
                 unsigned y_dim, const MeshAccel &mesh_accel,
                 Span<const TriAccel> tri_accels, const L &light_sampler,
                 const D &direction_sampler, const T &term_prob, const R &rng,
                 Span<BGRA> pixels, Span<Eigen::Array3f>,
                 Span<const scene::TriangleData> triangle_data,
                 Span<const material::Material> materials,
                 const Eigen::Affine3f &film_to_world) {
  ProgressBar progress_bar(x_dim * y_dim, 70);
  if (show_progress) {
    progress_bar.display();
  }

#ifdef NDEBUG
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
#endif
  for (unsigned y = 0; y < y_dim; y++) {
    for (unsigned x = 0; x < x_dim; x++) {
      pixels[x + y * x_dim] = intensity_to_bgr(
          intensities_impl(x, y, 0, samples_per, settings, x_dim, y_dim,
                           mesh_accel, tri_accels, light_sampler,
                           direction_sampler, term_prob, rng, triangle_data,
                           materials, film_to_world) /
          samples_per);
      if (show_progress) {
#pragma omp critical
        {
          ++progress_bar;
          progress_bar.display();
        }
      }
    }
  }

  if (show_progress) {
    progress_bar.done();
  }
}

template class RendererImpl<ExecutionModel::CPU>;
} // namespace detail
} // namespace render
