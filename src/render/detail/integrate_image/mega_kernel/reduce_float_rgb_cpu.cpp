#include "kernel/kernel_launch_impl_cpu.h"
#include "render/detail/integrate_image/mega_kernel/reduce_float_rgb_impl.h"

namespace render {
namespace detail {
namespace integrate_image {
namespace mega_kernel {
template struct ReduceFloatRGB<ExecutionModel::CPU>;
}
} // namespace integrate_image
} // namespace detail
} // namespace render
