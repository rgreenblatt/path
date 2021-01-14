#ifndef CPU_ONLY
#include "render/detail/general_render_impl.h"
#include "render/detail/integrate_image_impl_bulk.h"

namespace render {
template class Renderer::Impl<ExecutionModel::GPU>;
} // namespace render
#endif
