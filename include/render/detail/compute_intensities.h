#pragma once

#include "lib/execution_model.h"
#include "lib/span.h"
#include "render/detail/divide_work.h"

#include <Eigen/Core>

namespace render {
namespace detail {
template <ExecutionModel execution_model, typename Accel>
void compute_intensities(const WorkDivision &division, unsigned x_dim,
                         unsigned y_dim, const Accel &accel,
                         Span<Eigen::Vector3f> intermediate_intensities,
                         Span<Eigen::Vector3f> final_intensities);
}
} // namespace render
