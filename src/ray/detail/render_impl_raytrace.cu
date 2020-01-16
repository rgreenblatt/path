#include "ray/detail/accel/kdtree/kdtree_ref.h"
#include "ray/detail/impl/render_impl_raytrace.cuh"

namespace ray {
namespace detail {
template class RendererImpl<ExecutionModel::GPU>;

template void
RendererImpl<ExecutionModel::GPU>::raytrace_pass<false,
                                                 accel::kdtree::KDTreeRef>(
    const accel::kdtree::KDTreeRef &accel, unsigned current_num_blocks,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights,
    Span<const scene::TextureImageRef> textures);

template void
RendererImpl<ExecutionModel::GPU>::raytrace_pass<true,
                                                 accel::kdtree::KDTreeRef>(
    const accel::kdtree::KDTreeRef &accel, unsigned current_num_blocks,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights,
    Span<const scene::TextureImageRef> textures);
} // namespace detail
} // namespace ray
