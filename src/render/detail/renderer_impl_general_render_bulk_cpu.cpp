#include "kernel/work_division_impl.h"
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image_bulk_impl.h"

namespace render {
template class Renderer::Impl<ExecutionModel::CPU>;
} // namespace render
