#pragma once

#include "intersect/accel/impl/loop_all_impl.h"
#include "intersect/impl/ray_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "intersect/ray.h"
#include "render/detail/compute_intensities.h"
#include "rng/rng.h"

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

template <typename Accel, typename LightSampler, typename DirSampler,
          typename TermProb>
HOST_DEVICE inline void compute_intensities_impl(
    unsigned block_idx, unsigned thread_idx, unsigned block_dim,
    const WorkDivision &division, unsigned x_dim, unsigned y_dim,
    const Accel &accel, const LightSampler &light_sampler,
    const DirSampler &direction_sampler, const TermProb &term_prob,
    Span<Eigen::Array3f> intensities,
    Span<const scene::TriangleData> triangle_data,
    Span<const material::Material> materials,
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
  bool count_emission = true;

  intersect::Ray ray;
  Eigen::Array3f multiplier;

  unsigned max_sampling_num; // TODO

  rng::Rng rng(0, max_sampling_num);

  constexpr unsigned max_values_covered_by_thread = 4;

  struct IntensityIndexes {
    Eigen::Array3f intensity;
    unsigned x;
    unsigned y;
  };

  std::array<IntensityIndexes, max_values_covered_by_thread> values_covered;
  uint8_t value_idx = 255;

  while (!finished || next_idx != this_thread_end) {
    if (finished) {
      unsigned new_x =
          (next_idx / division.sample_block_size) % division.x_block_size;
      unsigned new_y =
          (next_idx / (division.sample_block_size * division.x_block_size));

      if (value_idx == 255 || values_covered[value_idx].x != new_x ||
          values_covered[value_idx].y != new_y) {
        value_idx++;
        assert(value_idx < values_covered.size());
        values_covered[value_idx] = {Eigen::Vector3f::Zero(), new_x, new_y};
      }

      multiplier = Eigen::Vector3f::Ones();
      unsigned sample_idx =
          next_idx % division.sample_block_size + start_sample;
      rng.set_state(sample_idx);
      assert(new_y < division.y_block_size);

      auto [x_offset, y_offset] = rng.sample_2();

      ray = initial_ray(new_x + x_offset, new_y + y_offset, x_dim, y_dim,
                        film_to_world);

      finished = false;
      count_emission = true;
    }

    auto next_intersection = accel(ray);

    if (!next_intersection.has_value()) {
      finished = true;
      continue;
    }

    unsigned triangle_idx = next_intersection->info[0];
    unsigned mesh_idx = next_intersection->info[1];

    const auto &data = triangle_data[triangle_idx];
    const auto &material = materials[data.material_idx()];

    Eigen::Array3f &intensity = values_covered[value_idx].intensity;

    // count intensity if eye ray
    if (!LightSampler::performs_samples || count_emission) {
      intensity += multiplier * material.emission(); // TODO: check
    }

    Eigen::Vector3f intersection_point =
        next_intersection->intersection_dist * ray.direction + ray.origin;

    const auto &mesh = accel.get(mesh_idx);

    Eigen::Vector3f mesh_space_intersection_point =
        mesh.world_to_mesh() * intersection_point;

    const intersect::Triangle &triangle =
        mesh.accel_triangle().get(triangle_idx);

#if 0
    Eigen::Array3f color =
        data.get_color(mesh_space_intersection_point, triangle);
#endif
    Eigen::Vector3f normal =
        (mesh.mesh_to_world() *
         data.get_normal(mesh_space_intersection_point, triangle))
            .normalized();

    auto direction_multiplier =
        [&](const Eigen::Vector3f &outgoing_dir) -> Eigen::Array3f {
      return material.brdf(ray.direction, outgoing_dir, normal) *
             outgoing_dir.dot(normal);
    };

#if 0
    multiplier *= color; // TODO: check
#endif

    auto compute_direct_lighting = [&]() -> Eigen::Array3f {
      Eigen::Array3f intensity = Eigen::Array3f::Zero();
      const auto samples = light_sampler(intersection_point, material, normal,
                                         ray.direction, rng);
      for (const auto &sample : samples) {
        intersect::Ray light_ray{intersection_point, sample.direction};

        auto light_intersection = accel(light_ray);
        if (!light_intersection.has_value()) {
          continue;
        }

        unsigned triangle_idx = next_intersection->info[0];
        const auto &data = triangle_data[triangle_idx];
        const auto &material = materials[data.material_idx()];

        // TODO: check (prob not delta needed?)
        intensity += material.emission() *
                     material.prob_not_delta() *
                     direction_multiplier(light_ray.direction) / sample.prob;
      }

      return intensity;
    };

    if (material.has_non_delta_samples()) {
      intensity += multiplier * compute_direct_lighting();
    }

    Eigen::Vector3f next_dir;

    bool use_delta_event = material.delta_prob_check(rng);
    count_emission = use_delta_event;

    if (use_delta_event) {
      auto [next_dir_v, m] = material.delta_sample(rng, ray.direction, normal);

      next_dir = next_dir_v;

      multiplier *= m;
    } else {
      auto [next_dir_v, prob_of_next_direction_v] = direction_sampler(
          intersection_point, material, normal, ray.direction, rng);

      next_dir = next_dir_v;

      multiplier *= direction_multiplier(next_dir) / prob_of_next_direction_v;
    }

    float this_term_prob = term_prob(multiplier);

    if (rng.sample_1() < this_term_prob) {
      finished = true;
      continue;
    }

    multiplier /= (1.0f - this_term_prob);

    ray.origin = intersection_point;
    ray.direction = next_dir;

    rng.next_state();
  }

  // TODO: total intensities
}
} // namespace detail
} // namespace render
