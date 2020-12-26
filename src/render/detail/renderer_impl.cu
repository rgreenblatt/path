#include "render/detail/renderer_impl.h"

namespace render {
template <ExecutionModel execution_model>
Renderer::Impl<execution_model>::Impl() {}

template class Renderer::Impl<ExecutionModel::CPU>;
template class Renderer::Impl<ExecutionModel::GPU>;
} // namespace render
