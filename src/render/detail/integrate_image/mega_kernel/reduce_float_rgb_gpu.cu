#ifndef CPU_ONLY
#include "kernel/kernel_launch_impl_gpu.cuh"
#include "kernel/runtime_constants_reducer_impl_gpu.cuh"
#include "render/detail/integrate_image/mega_kernel/reduce_float_rgb_impl.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace mega_kernel {
template struct ReduceFloatRGB<ExecutionModel::GPU>;
}
} // namespace integrate_image
} // namespace detail
} // namespace render

#endif
