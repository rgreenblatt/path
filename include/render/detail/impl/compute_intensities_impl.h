#pragma once

#include "intersect/accel/impl/loop_all_impl.h"
#include "intersect/impl/ray_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "intersect/ray.h"
#include "render/detail/compute_intensities.h"
#include "render/detail/halton.h"

namespace render {
namespace detail {
HOST_DEVICE inline intersect::Ray
initial_ray(float x, float y, unsigned x_dim, unsigned y_dim,
            const Eigen::Affine3f &film_to_world) {

  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * x) / x_dim - 1.0f, (-2.0f * y) / y_dim + 1.0f, -1.0f);
  const auto world_space_film_plane = film_to_world * camera_space_film_plane;

  intersect::Ray ray;

  ray.origin = film_to_world.translation();
  ray.direction = (world_space_film_plane - ray.origin).normalized();

  return ray;
}

template <typename Accel>
HOST_DEVICE inline void
compute_intensities_impl(unsigned block_idx, unsigned thread_idx,
                         unsigned block_dim, const WorkDivision &division,
                         unsigned x_dim, unsigned y_dim, const Accel &accel,
                         Span<Eigen::Vector3f> intensities,
                         Span<const scene::TriangleData> triangle_data,
                         Span<const scene::Material> materials,
                         const Eigen::Affine3f &film_to_world) {
  const unsigned block_idx_sample = block_idx % division.num_sample_blocks;
  const unsigned block_idx_pixel = block_idx / division.num_sample_blocks;
  const unsigned block_idx_x = block_idx_pixel % division.num_x_blocks;
  const unsigned block_idx_y = block_idx_pixel / division.num_x_blocks;

  const unsigned start_sample = block_idx_sample * division.sample_block_size;
  const unsigned start_x = block_idx_x * division.x_block_size;
  const unsigned start_y = block_idx_y * division.y_block_size;

  // - maintain indexes, store as little as possible
  // - separate trace for lights and for generic intersection (I think this will
  //   be better than trying to mutplex in one loop)
  // - do n loop iterations, then prefix sum over block ***
  //    - see other work on thread compaction....
  //    - warp local prefix sum***
  //      - pros:
  //        - simpler
  //        - no shared memory required (except at end)
  //        -
  //    - maybe some special casing for different cases around num samples
  //    - generally optimize for num samples > warp_size: maybe just
  //    - transmitted state will have to include:
  //      - internal RNG state
  //      - location
  //      - idx (location etc...)
  //      -
  //

  unsigned end_sample = (block_idx_sample + 1) * division.sample_block_size;
  unsigned end_x = (block_idx_x + 1) * division.x_block_size;
  unsigned end_y = (block_idx_y + 1) * division.y_block_size;

  unsigned total_size_per_block = division.sample_block_size *
                                  division.x_block_size * division.y_block_size;
  unsigned high_per_thread = ceil_divide(total_size_per_block, block_dim);
  unsigned extra = high_per_thread * block_dim - total_size_per_block;
  unsigned low_per_thread = high_per_thread - 1;

  unsigned thread_idx_start_low = block_dim - extra;

  unsigned size_before_low = high_per_thread * (block_dim - extra);
  assert(extra % 32 == 0);
  assert(extra < block_dim);
  assert(total_size_per_block ==
         high_per_thread * (block_dim - extra) + low_per_thread * extra);

  bool this_thread_low = thread_idx > (block_dim - extra - 1);
  unsigned this_thread_start =
      this_thread_low ? size_before_low +
                            (thread_idx - thread_idx_start_low) * low_per_thread
                      : high_per_thread * thread_idx;
  unsigned this_thread_end =
      this_thread_start + (this_thread_low ? low_per_thread : high_per_thread);

  unsigned next_idx = this_thread_start;
  bool finished = true;
  bool is_first = true;

  intersect::Ray ray;
  float multiplier;

  uint16_t halton_counter;
  unsigned x;
  unsigned y;
  unsigned sample_idx;

  while (!finished || next_idx != this_thread_end) {
    if (finished) {
      multiplier = 1.0f;
      sample_idx = next_idx % division.sample_block_size + start_sample;
      x = (next_idx / division.sample_block_size) % division.x_block_size;
      y = (next_idx / (division.sample_block_size * division.x_block_size));
      halton_counter = sample_idx;
      assert(y < division.y_block_size);

      auto [x_offset, y_offset] = halton<2>(halton_counter);

      ray =
          initial_ray(x + x_offset, y + y_offset, x_dim, y_dim, film_to_world);

      finished = false;
      is_first = true;
    }

    auto next_intersection = accel(ray);

    if (!next_intersection.has_value()) {
      finished = true;
      continue;
    }

    /* compute_direct_lighting(); */

    is_first = false;
  }
}
} // namespace detail
} // namespace render
