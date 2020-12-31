#pragma once

#include "bsdf/bsdf.h"
#include "execution_model/execution_model.h"
#include "integrate/dir_sampler/dir_sampler.h"
#include "integrate/light_sampler/light_sampler.h"
#include "integrate/term_prob/term_prob.h"
#include "intersectable_scene/intersectable_scene.h"
#include "lib/bgra.h"
#include "lib/span.h"
#include "render/general_settings.h"
#include "rng/rng.h"
#include "work_division/work_division.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace render {
namespace detail {
using integrate::dir_sampler::DirSamplerRef;
using integrate::light_sampler::LightSamplerRef;
using integrate::term_prob::TermProbRef;
using work_division::WorkDivision;

template <ExecutionModel exec> struct IntegrateImage {
  template <intersectable_scene::IntersectableScene S,
            LightSamplerRef<typename S::B> L, DirSamplerRef<typename S::B> D,
            TermProbRef T, rng::RngRef R>
  static void run(bool output_as_bgra, const GeneralSettings &settings,
                  bool show_progress, const WorkDivision &division,
                  unsigned samples_per, unsigned x_dim, unsigned y_dim,
                  const S &scene, const L &light_sampler,
                  const D &direction_sampler, const T &term_prob, const R &rng,
                  Span<BGRA> pixels, Span<Eigen::Array3f> intensities,
                  const Eigen::Affine3f &film_to_world);
};
} // namespace detail
} // namespace render
