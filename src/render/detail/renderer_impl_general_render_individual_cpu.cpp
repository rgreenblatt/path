#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image_impl_individual_cpu.h"

namespace render {
template class Renderer::Impl<ExecutionModel::CPU>;
} // namespace render
