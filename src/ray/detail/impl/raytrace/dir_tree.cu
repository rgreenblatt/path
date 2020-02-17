#include "ray/detail/accel/dir_tree/impl/dir_tree_lookup_ref_impl.h"
#include "ray/detail/accel/kdtree/kdtree_ref_impl.h"
#include "ray/detail/accel/loop_all.h"
#include "ray/detail/impl/render_impl_raytrace.h"

namespace ray {
namespace detail {
template class RendererImpl<ExecutionModel::GPU>;

template void RendererImpl<ExecutionModel::GPU>::raytrace_pass<
    false, accel::dir_tree::DirTreeLookupRef>(
    const accel::dir_tree::DirTreeLookupRef &accel, unsigned current_num_blocks,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights,
    Span<const scene::TextureImageRef> textures);

template void RendererImpl<ExecutionModel::GPU>::raytrace_pass<
    true, accel::dir_tree::DirTreeLookupRef>(
    const accel::dir_tree::DirTreeLookupRef &accel, unsigned current_num_blocks,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights,
    Span<const scene::TextureImageRef> textures);
} // namespace detail
} // namespace ray
