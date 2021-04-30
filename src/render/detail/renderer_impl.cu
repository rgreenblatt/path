#include "render/detail/renderer_impl.h"

namespace render {
template <ExecutionModel exec> Renderer::Impl<exec>::Impl() {}

template class Renderer::Impl<ExecutionModel::CPU>;
#ifndef CPU_ONLY
template class Renderer::Impl<ExecutionModel::GPU>;
#endif
} // namespace render
