#pragma once

#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/impl/raytrace.h"
#include "ray/detail/intersection/solve.h"
#include "ray/detail/render_impl.h"

namespace ray {
namespace detail {
template <bool is_first, typename Accel>
__global__ void
raytrace_global(const BlockData block_data, const Accel accel,
                Span<const scene::ShapeData> shapes,
                SpanSized<const scene::Light> lights,
                Span<const scene::TextureImageRef> textures,
                Span<Eigen::Vector3f> world_space_eyes,
                Span<Eigen::Vector3f> world_space_directions,
                Span<Eigen::Array3f> color_multipliers,
                Span<scene::Color> colors, Span<unsigned> ignores,
                Span<uint8_t> disables, Span<uint8_t> group_disables,
                SpanSized<const unsigned> group_indexes, unsigned num_shapes) {
  raytrace_impl<is_first>(
      blockIdx.x, threadIdx.x, block_data, accel, shapes, lights, textures,
      world_space_eyes, world_space_directions, color_multipliers, colors,
      ignores, disables, group_disables, group_indexes, num_shapes);
}
template <ExecutionModel execution_model>
template <bool is_first, typename Accel>
void RendererImpl<execution_model>::raytrace_pass(
    const Accel &accel, unsigned current_num_blocks,
    SpanSized<const scene::ShapeData> shapes,
    SpanSized<const scene::Light> lights,
    Span<const scene::TextureImageRef> textures) {
  unsigned general_block_size = block_data_.generalBlockSize();

  // removing below line leads to compiler error (constructor...) :)
  auto needed_for_some_reason = shapes;

  if (current_num_blocks != 0) {
    raytrace_global<is_first><<<current_num_blocks, general_block_size>>>(
        block_data_, accel, shapes, lights, textures, world_space_eyes_,
        world_space_directions_, color_multipliers_, colors_, ignores_,
        disables_, group_disables_, group_indexes_, shapes.size());
  }

  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}
} // namespace detail
} // namespace ray