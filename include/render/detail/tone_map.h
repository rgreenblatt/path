#pragma once

#include "lib/execution_model.h"
#include "lib/bgra.h"
#include "lib/span.h"

#include <Eigen/Core>

namespace render {
namespace detail {
template <ExecutionModel execution_model>
void tone_map(SpanSized<const Eigen::Array3f> intensities, SpanSized<BGRA> bgr);
}
} // namespace render
