#ifndef CPU_ONLY
#include "kernel/kernel_launch_impl_gpu.cuh"
#include "kernel/work_division_impl.h"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image/streaming/run_impl.h"

namespace render {
template class Renderer::Impl<ExecutionModel::GPU>;
} // namespace render
#endif
