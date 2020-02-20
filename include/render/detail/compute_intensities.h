#pragma once

#include "lib/execution_model.h"
#include "lib/span.h"
#include "material/material.h"
#include "render/detail/divide_work.h"
#include "scene/triangle_data.h"

#include <Eigen/Geometry>

namespace render {
namespace detail {
template <ExecutionModel execution_model, typename Accel, typename LightSampler,
          typename DirSampler, typename TermProb>
void compute_intensities(const WorkDivision &division, unsigned samples_per,
                         unsigned x_dim, unsigned y_dim, unsigned block_size,
                         const Accel &accel, const LightSampler &light_sampler,
                         const DirSampler &direction_sampler,
                         const TermProb &term_prob,
                         Span<Eigen::Array3f> intensities,
                         Span<const scene::TriangleData> triangle_data,
                         Span<const material::Material> materials,
                         const Eigen::Affine3f &film_to_world);
}
} // namespace render
