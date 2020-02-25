#pragma once

#include "intersect/accel/impl/kdtree_impl.h"
#include "intersect/accel/impl/loop_all_impl.h"
#include "intersect/impl/ray_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "render/detail/compute_intensities.h"

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

template <intersect::accel::AccelRef MeshAccel,
          intersect::accel::AccelRef TriAccel, LightSamplerRef L,
          DirSamplerRef D, TermProbRef T, rng::RngRef R>
HOST_DEVICE inline Eigen::Array3f compute_intensities_impl(
 unsigned x, unsigned y,
    unsigned start_sample, unsigned end_sample,
    const ComputationSettings &, unsigned x_dim, unsigned y_dim, unsigned,
    const MeshAccel &accel, Span<const TriAccel> &tri_accels,
    const L &light_sampler, const D &dir_sampler, const T &term_prob,
    const R &rng_ref, Span<const scene::TriangleData> triangle_data,
    Span<const material::Material> materials,
    const Eigen::Affine3f &film_to_world) {
  unsigned sample_idx = start_sample;
  bool finished = true;
  bool count_emission = true;

  intersect::Ray ray;
  Eigen::Array3f multiplier;

  typename R::State rng;

  Eigen::Array3f intensity = Eigen::Array3f::Zero();

  while (!finished || sample_idx != end_sample) {
    if (finished) {
      multiplier = Eigen::Vector3f::Ones();
      rng = rng_ref.get_generator(sample_idx, x, y);

      float x_offset = rng.next();
      float y_offset = rng.next();

      ray =
          initial_ray(x + x_offset, y + y_offset, x_dim, y_dim, film_to_world);

      finished = false;
      count_emission = true;

      sample_idx++;
    }

    auto get_intersection = [&](const intersect::Ray &ray) {
      return intersect::IntersectableT<MeshAccel>::intersect(ray, accel,
                                                             tri_accels);
    };

    auto next_intersection = get_intersection(ray);

    if (!next_intersection.has_value()) {
      finished = true;
      continue;
    }

    unsigned triangle_idx = next_intersection->info[0];
    unsigned mesh_idx = next_intersection->info[1];

    const auto &data = triangle_data[triangle_idx];
    const auto &material = materials[data.material_idx()];

    if (!L::performs_samples || count_emission) {
      intensity += multiplier * material.emission(); // TODO: check
    }

    Eigen::Vector3f intersection_point =
        next_intersection->intersection_dist * ray.direction + ray.origin;

    const auto &mesh = accel.get(mesh_idx);

    Eigen::Vector3f mesh_space_intersection_point =
        mesh.world_to_object() * intersection_point;

    const intersect::Triangle &triangle =
        tri_accels[mesh.idx()].get(triangle_idx);

    Eigen::Vector3f normal =
        (mesh.object_to_world() *
         data.get_normal(mesh_space_intersection_point, triangle))
            .normalized();

    auto direction_multiplier =
        [&](const Eigen::Vector3f &outgoing_dir) -> Eigen::Array3f {
      auto brdf_val = material.brdf(ray.direction, outgoing_dir, normal);
      auto normal_v = outgoing_dir.dot(normal);

      return brdf_val * normal_v;
    };

    auto compute_direct_lighting = [&]() -> Eigen::Array3f {
      Eigen::Array3f intensity = Eigen::Array3f::Zero();
      const auto samples = light_sampler(intersection_point, material,
                                         ray.direction, normal, rng);
      for (unsigned i = 0; i < samples.num_samples; i++) {
        const auto &sample = samples.samples[i];
        intersect::Ray light_ray{intersection_point, sample.direction};

        auto light_intersection = get_intersection(light_ray);
        if (!light_intersection.has_value()) {
          continue;
        }

        unsigned triangle_idx = light_intersection->info[0];
        const auto &light_data = triangle_data[triangle_idx];
        const auto &light_material = materials[light_data.material_idx()];

        assert([&] {
          unsigned mesh_idx = light_intersection->info[1];

          const auto &mesh = accel.get(mesh_idx);

          const intersect::Triangle &triangle =
              tri_accels[mesh.idx()].get(triangle_idx);

          auto intersection =
              intersect::IntersectableT<intersect::Triangle>::intersect(
                  light_ray, triangle);

          return intersection.has_value() &&
                 intersection->intersection_dist ==
                     light_intersection->intersection_dist;
        }());

        // TODO: check (prob not delta needed?)
        intensity += light_material.emission() * material.prob_not_delta() *
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
      auto [next_dir_v, m] = material.delta_sample(ray.direction, normal, rng);

      next_dir = next_dir_v;

      multiplier *= m;
    } else {
      auto [next_dir_v, prob_of_next_direction_v] =
          dir_sampler(intersection_point, material, ray.direction, normal, rng);

      float prob_of_next_direction = prob_of_next_direction_v;
      next_dir = next_dir_v;

      assert(prob_of_next_direction_v >= 0);

      multiplier *= direction_multiplier(next_dir) / prob_of_next_direction;
    }

    auto this_term_prob = term_prob(multiplier);

    if (rng.next() <= this_term_prob) {
      finished = true;
      continue;
    }

    multiplier /= (1.0f - this_term_prob);

    if (multiplier.x() < 0 || multiplier.y() < 0 || multiplier.z() < 0) {
      // can be caused by edge cases in Phong brdf
      finished = true;
      continue;
    }

    ray.origin = intersection_point;
    ray.direction = next_dir;
  }

  return intensity;
}
} // namespace detail
} // namespace render
