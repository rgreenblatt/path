#include "kernel/kernel_launch_impl_cpu.h"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image_impl_individual.h"

namespace render {
template class Renderer::Impl<ExecutionModel::CPU>;
} // namespace render
