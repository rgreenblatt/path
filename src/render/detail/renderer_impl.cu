#include "lib/info/timer.h"
#include "render/detail/renderer_impl.h"

#include <thrust/copy.h>

namespace render {
namespace detail {
template <ExecutionModel execution_model>
RendererImpl<execution_model>::RendererImpl() {}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace render
