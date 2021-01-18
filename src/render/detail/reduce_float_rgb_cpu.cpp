#include "kernel/kernel_launch_impl_cpu.h"
#include "render/detail/reduce_float_rgb_impl.h"
#pragma message "more impls"

namespace render {
namespace detail {
template struct ReduceFloatRGB<ExecutionModel::CPU>;
} // namespace detail
} // namespace render
