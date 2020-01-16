#include "ray/detail/impl/render_impl_raytrace.h"
#include "ray/detail/accel/kdtree/kdtree_ref.h"

namespace ray {
namespace detail {
template class RendererImpl<ExecutionModel::CPU>;

template void
RendererImpl<ExecutionModel::CPU>::raytrace_pass<false,
                                                 accel::kdtree::KDTreeRef>(
    const accel::kdtree::KDTreeRef &accel, unsigned current_num_blocks,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights,
    Span<const scene::TextureImageRef> textures);

template void
RendererImpl<ExecutionModel::CPU>::raytrace_pass<true,
                                                 accel::kdtree::KDTreeRef>(
    const accel::kdtree::KDTreeRef &accel, unsigned current_num_blocks,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights,
    Span<const scene::TextureImageRef> textures);
} // namespace detail
} // namespace ray
