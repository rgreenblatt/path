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
HOST_DEVICE inline Eigen::Array3f compute_intensities_impl(
    unsigned x, unsigned y, unsigned start_sample, unsigned end_sample,
    unsigned x_dim, unsigned y_dim, unsigned num_samples, const Accel &accel,
    const LightSampler &light_sampler, const DirSampler &direction_sampler,
    const TermProb &term_prob, Span<const scene::TriangleData> triangle_data,
    Span<const material::Material> materials,
    const Eigen::Affine3f &film_to_world) {
  unsigned sample_idx = start_sample;
  bool finished = true;
  bool count_emission = true;

  intersect::Ray ray;
  Eigen::Array3f multiplier;

  unsigned max_sampling_num = std::max(num_samples, rng::sequence_size);

  rng::Rng rng(0, max_sampling_num);

  Eigen::Array3f intensity = Eigen::Array3f::Zero();

  while (!finished || sample_idx != end_sample) {
    if (finished) {
      multiplier = Eigen::Vector3f::Ones();
      rng.set_state(sample_idx);

      auto [x_offset, y_offset] = rng.sample_2();

      ray =
          initial_ray(x + x_offset, y + y_offset, x_dim, y_dim, film_to_world);

      finished = false;
      count_emission = true;

      sample_idx++;
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
      auto brdf_val = material.brdf(ray.direction, outgoing_dir, normal);
      auto normal_v = outgoing_dir.dot(normal);

      assert(brdf_val.x() >= 0);
      assert(brdf_val.y() >= 0);
      assert(brdf_val.z() >= 0);
      assert(normal_v >= 0);

      return brdf_val * normal_v;
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
        intensity += material.emission() * material.prob_not_delta() *
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

      assert(prob_of_next_direction_v >= 0);
      assert(direction_multiplier(next_dir).x() >= 0);
      assert(direction_multiplier(next_dir).y() >= 0);
      assert(direction_multiplier(next_dir).z() >= 0);

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

  return intensity;
}
} // namespace detail
} // namespace render
