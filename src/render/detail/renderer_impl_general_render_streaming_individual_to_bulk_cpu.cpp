#include "intersectable_scene/to_bulk_impl.h"
#include "kernel/kernel_launch_impl_cpu.h"
#include "kernel/work_division_impl.h"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image/streaming/run_impl.h"

namespace render {
template class Renderer::Impl<ExecutionModel::CPU>;
} // namespace render
