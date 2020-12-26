#pragma once

#include "execution_model/device_vector.h"
#include "lib/bgra.h"
#include "lib/span.h"

#include <Eigen/Core>

namespace render {
namespace detail {
DeviceVector<Eigen::Array3f> *reduce_intensities_gpu(
    bool output_as_bgra, unsigned reduction_factor, unsigned samples_per,
    DeviceVector<Eigen::Array3f> *intensities_in,
    DeviceVector<Eigen::Array3f> *intensities_out, Span<BGRA> bgras);
}
} // namespace render
