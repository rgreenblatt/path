#ifndef CPU_ONLY
#include "kernel/kernel_launch_impl_gpu.cuh"
#include "render/detail/reduce_float_rgb_impl.h"
#pragma message "more impls"

namespace render {
namespace detail {
template struct ReduceFloatRGB<ExecutionModel::GPU>;
} // namespace detail
} // namespace render

#endif
