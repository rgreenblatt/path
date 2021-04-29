#include "kernel/kernel_launch_impl_cpu.h"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image/mega_kernel/run_impl.h"

namespace render {
template class Renderer::Impl<ExecutionModel::CPU>;
} // namespace render
