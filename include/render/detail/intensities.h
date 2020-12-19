#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/accel.h"
#include "lib/bgra.h"
#include "lib/span.h"
#include "material/material.h"
#include "render/general_settings.h"
#include "render/detail/dir_sampler.h"
#include "render/detail/divide_work.h"
#include "render/detail/light_sampler.h"
#include "render/detail/term_prob.h"
#include "rng/rng.h"
#include "scene/triangle_data.h"

#include <Eigen/Geometry>

namespace render {
namespace detail {
template <intersect::accel::AccelRef MeshAccel,
          intersect::accel::AccelRef TriAccel, LightSamplerRef L,
          DirSamplerRef D, TermProbRef T, rng::RngRef R>
void intensities(const GeneralSettings &settings, bool show_progress,
                 const WorkDivision &division, unsigned samples_per,
                 unsigned x_dim, unsigned y_dim, const MeshAccel &mesh_accel,
                 Span<const TriAccel> tri_accels, const L &light_sampler,
                 const D &direction_sampler, const T &term_prob, const R &rng,
                 Span<BGRA> pixels, Span<Eigen::Array3f> intensities,
                 Span<const scene::TriangleData> triangle_data,
                 Span<const material::Material> materials,
                 const Eigen::Affine3f &film_to_world);
}
} // namespace render
