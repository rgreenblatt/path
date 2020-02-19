#pragma once

#include "lib/execution_model.h"
#include "lib/rgba.h"
#include "lib/span.h"

#include <Eigen/Core>

namespace render {
namespace detail {
template <ExecutionModel execution_model>
void tone_map(unsigned x_dim, unsigned y_dim, Span<Eigen::Vector3f> intensities,
              Span<RGBA> rgb);

}
} // namespace render
