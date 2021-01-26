#ifndef CPU_ONLY
#include "kernel/kernel_launch_impl_gpu.cuh"
#include "kernel/runtime_constants_reducer_impl_gpu.cuh"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image_individual_impl.h"

namespace render {
template class Renderer::Impl<ExecutionModel::GPU>;
} // namespace render
#endif
