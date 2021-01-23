#ifndef CPU_ONLY
#include "kernel/kernel_launch_impl_gpu.cuh"
#include "kernel/runtime_constants_reducer_impl_gpu.cuh"
#include "render/detail/reduce_float_rgb_impl.h"

namespace render {
namespace detail {
template struct ReduceFloatRGB<ExecutionModel::GPU>;
} // namespace detail
} // namespace render

#endif
