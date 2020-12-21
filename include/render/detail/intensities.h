#pragma once

#include "intersectable_scene/intersectable_scene.h"
#include "lib/bgra.h"
#include "lib/span.h"
#include "render/detail/dir_sampler.h"
#include "render/detail/divide_work.h"
#include "render/detail/light_sampler.h"
#include "render/detail/term_prob.h"
#include "render/general_settings.h"
#include "rng/rng.h"

#include <Eigen/Geometry>

namespace render {
namespace detail {
// generalize initial rays/work division...
template <intersectable_scene::IntersectableScene Scene, LightSamplerRef L,
          DirSamplerRef D, TermProbRef T, rng::RngRef R>
void intensities(const GeneralSettings &settings, bool show_progress,
                 const WorkDivision &division, unsigned samples_per,
                 unsigned x_dim, unsigned y_dim, const Scene &scene,
                 const L &light_sampler, const D &direction_sampler,
                 const T &term_prob, const R &rng, Span<BGRA> pixels,
                 Span<Eigen::Array3f> intensities,
                 const Eigen::Affine3f &film_to_world);
} // namespace detail
} // namespace render
