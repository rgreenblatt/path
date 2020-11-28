#pragma once

#include "intersect/accel/impl/kdtree_impl.h"
#include "intersect/accel/impl/loop_all_impl.h"
#include "intersect/impl/ray_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "render/detail/intensities.h"

#include "lib/info/printf_dbg.h"

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
HOST_DEVICE inline Eigen::Array3f intensities_impl(
    unsigned x, unsigned y, unsigned start_sample, unsigned end_sample,
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
  unsigned iters;

  typename R::State rng;

  Eigen::Array3f intensity = Eigen::Array3f::Zero();

  while (!finished || sample_idx != end_sample) {
    if (finished) {
      multiplier = Eigen::Vector3f::Ones();
      rng = rng_ref.get_generator(sample_idx, x, y);
      iters = 0;

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

    auto compute_direct_lighting = [&]() -> Eigen::Array3f {
      Eigen::Array3f intensity = Eigen::Array3f::Zero();
      const auto samples = light_sampler(intersection_point, material,
                                         ray.direction, normal, rng);
      for (unsigned i = 0; i < samples.num_samples; i++) {
        const auto &sample = samples.samples[i];

        // TODO: BSDF case
        if (sample.direction.dot(normal) <= 0.f) {
          continue;
        }

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

          bool out = intersection.has_value() &&
                     abs(intersection->intersection_dist -
                         light_intersection->intersection_dist) < 1e-7f ;
          if (!out) {
            printf_dbg(intersection.has_value());
            printf_dbg(intersection->intersection_dist);
            printf_dbg(light_intersection->intersection_dist);
            printf_dbg((intersection->intersection_dist -
                       light_intersection->intersection_dist)*1.0e15f);
            printf_dbg(intersection->intersection_dist ==
                       light_intersection->intersection_dist);
          }

          return out;
        }());

        auto multiplier =
            material.brdf(ray.direction, light_ray.direction, normal);

        // TODO: check (prob not delta needed?)
        intensity += light_material.emission() * material.prob_not_delta() *
                     multiplier / sample.prob;
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
      // SPEED: there are advantages to having the sampler compute the
      // multiplier (for instance, the brdf/normal value may cancel)
      auto [next_dir_v, prob_of_next_direction_v] =
          dir_sampler(intersection_point, material, ray.direction, normal, rng);

      float prob_of_next_direction = prob_of_next_direction_v;
      next_dir = next_dir_v;

      assert(prob_of_next_direction_v >= 0);

      auto direction_multiplier =
          [&](const Eigen::Vector3f &outgoing_dir) -> Eigen::Array3f {
        auto normal_v = outgoing_dir.dot(normal);

        // TODO: BSDF case
        assert(normal_v >= 0.f);

        auto brdf_val = material.brdf(ray.direction, outgoing_dir, normal);

        assert(brdf_val.x() >= 0.0f);
        assert(brdf_val.y() >= 0.0f);
        assert(brdf_val.z() >= 0.0f);

        return (brdf_val * normal_v).eval();
      };

      multiplier *= direction_multiplier(next_dir) / prob_of_next_direction;
    }

    auto this_term_prob = term_prob(iters, multiplier);

    if (rng.next() <= this_term_prob) {
      finished = true;
      continue;
    }

    multiplier /= (1.0f - this_term_prob);

    assert(multiplier.x() >= 0.0f);
    assert(multiplier.y() >= 0.0f);
    assert(multiplier.z() >= 0.0f);

    ray.origin = intersection_point;
    ray.direction = next_dir;
    iters++;
  }

  return intensity;
}
} // namespace detail
} // namespace render
