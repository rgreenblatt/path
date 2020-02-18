#include "ray/detail/accel/loop_all.h"
#include "ray/detail/impl/render_impl_raytrace.cuh"

namespace ray {
namespace detail {
template class RendererImpl<ExecutionModel::GPU>;

template void
RendererImpl<ExecutionModel::GPU>::raytrace_pass<true, accel::LoopAll>(
    const accel::LoopAll &accel, unsigned current_num_blocks,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights,
    Span<const scene::TextureImageRef> textures);
} // namespace detail
} // namespace ray
