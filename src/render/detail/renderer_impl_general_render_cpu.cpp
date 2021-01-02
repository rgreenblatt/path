#include "render/detail/assign_output.h"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image.h"
#include "render/detail/integrate_pixel.h"

#include <cli/ProgressBar.hpp>

namespace render {
namespace detail {
template <>
template <intersectable_scene::IntersectableScene S,
          LightSamplerRef<typename S::B> L, DirSamplerRef<typename S::B> D,
          TermProbRef T, rng::RngRef R>
void IntegrateImage<ExecutionModel::CPU>::run(
    bool output_as_bgra, const GeneralSettings &settings, bool show_progress,
    const WorkDivision &, unsigned samples_per, unsigned x_dim, unsigned y_dim,
    S &scene, const L &light_sampler, const D &direction_sampler,
    const T &term_prob, const R &rng, Span<BGRA> pixels,
    Span<Eigen::Array3f> intensities, const Eigen::Affine3f &film_to_world) {
  ProgressBar progress_bar(x_dim * y_dim, 70);
  if (show_progress) {
    progress_bar.display();
  }

#ifdef NDEBUG
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
#endif
  for (unsigned y = 0; y < y_dim; y++) {
    for (unsigned x = 0; x < x_dim; x++) {
      assign_output(
          output_as_bgra, pixels, intensities, 0, 1, x, y, x_dim, samples_per,
          integrate_pixel(x, y, 0, samples_per,
                          settings.rendering_equation_settings, x_dim, y_dim,
                          scene.intersectable(), scene.scene(), light_sampler,
                          direction_sampler, term_prob, rng, film_to_world));
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
} // namespace detail

template class Renderer::Impl<ExecutionModel::CPU>;
} // namespace render
