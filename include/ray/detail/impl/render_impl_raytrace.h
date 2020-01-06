#pragma once

#include "ray/detail/impl/raytrace.h"
#include "ray/detail/intersection/solve.h"
#include "ray/detail/render_impl.h"
#include "ray/detail/render_impl_utils.h"

namespace ray {
namespace detail {
template <bool is_first, typename Accel>
inline void raytrace(const BlockData block_data, const Accel &accel,
                     Span<const scene::ShapeData> shapes,
                     Span<const scene::Light, false> lights,
                     Span<const scene::TextureImageRef> textures,
                     Span<Eigen::Vector3f> world_space_eyes,
                     Span<Eigen::Vector3f> world_space_directions,
                     Span<Eigen::Array3f> color_multipliers,
                     Span<scene::Color> colors, Span<unsigned> ignores,
                     Span<uint8_t> disables, Span<uint8_t> group_disables,
                     Span<const unsigned, false> group_indexes,
                     unsigned current_num_blocks, unsigned num_shapes) {
#pragma omp parallel for collapse(2) schedule(dynamic, 16)
  for (unsigned block_index = 0; block_index < current_num_blocks;
       block_index++) {
    for (unsigned thread_index = 0;
         thread_index < block_data.generalBlockSize(); thread_index++) {
      raytrace_impl<is_first>(
          block_index, thread_index, block_data, accel, shapes, lights,
          textures, world_space_eyes, world_space_directions, color_multipliers,
          colors, ignores, disables, group_disables, group_indexes, num_shapes);
    }
  }
}

template <ExecutionModel execution_model>
template <bool is_first, typename Accel>
void RendererImpl<execution_model>::raytrace_pass(
    const Accel &accel, unsigned current_num_blocks,
    Span<const scene::ShapeData, false> shapes,
    Span<const scene::Light, false> lights,
    Span<const scene::TextureImageRef> textures) {
  auto shapes_span = Span<const scene::ShapeData>(shapes.data(), shapes.size());

  raytrace<is_first>(
      block_data_, accel, shapes_span, lights, textures,
      to_span(world_space_eyes_), to_span(world_space_directions_),
      to_span(color_multipliers_), to_span(colors_), to_span(ignores_),
      to_span(disables_), to_span(group_disables_),
      Span<const unsigned, false>(group_indexes_.data(), group_indexes_.size()),
      current_num_blocks, shapes.size());
}
} // namespace detail
} // namespace ray
