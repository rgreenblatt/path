#pragma once

#include "intersect/accel/kdtree/kdtree_impl.h"
#include "intersect/accel/loop_all/loop_all_impl.h"
#include "intersect/impl/ray_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "render/detail/intensities.h"

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
    const GeneralSettings &settings, unsigned x_dim, unsigned y_dim, 
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

    const auto &mesh = accel.get(mesh_idx);

    const intersect::Triangle &triangle =
        tri_accels[mesh.idx()].get(triangle_idx);

    Eigen::Vector3f intersection_point =
        next_intersection->intersection_dist * ray.direction + ray.origin;

    Eigen::Vector3f mesh_space_intersection_point =
        mesh.world_to_object() * intersection_point;

    Eigen::Vector3f normal =
        (mesh.object_to_world() *
         data.get_normal(mesh_space_intersection_point, triangle))
            .normalized();

    auto is_back_intersection = [&](const intersect::Ray &ray,
                                    const intersect::Triangle &triangle) {
      // basically a copy of triangle intersection, a bit gross...
      // probably can be optimized away by the compiler (because this is
      // repeated computation)
      const auto &vertices = triangle.vertices();

      volatile bool undefined_garbage = false;
      if (undefined_garbage) {
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            printf("%f", vertices[i][j]);
          }
        }
      }

      Eigen::Vector3f edge1 = vertices[1] - vertices[0];
      Eigen::Vector3f edge2 = vertices[2] - vertices[0];

      Eigen::Vector3f h = ray.direction.cross(edge2);
      float a = edge1.dot(h);

      return a < 0.;
    };

    auto include_lighting = [&](const intersect::Ray &ray,
                                const intersect::Triangle &triangle) {
      return !settings.back_cull_emission ||
             !is_back_intersection(ray, triangle);
    };

    auto compute_direct_lighting = [&]() -> Eigen::Array3f {
      Eigen::Array3f intensity = Eigen::Array3f::Zero();
      const auto samples =
          light_sampler(TriangleID{mesh_idx, triangle_idx}, intersection_point,
                        material, ray.direction, normal, rng);
      for (unsigned i = 0; i < samples.num_samples; i++) {
        const auto &[dir_sample, id] = samples.samples[i];

        // TODO: BSDF case
        if (dir_sample.direction.dot(normal) <= 0.f) {
          continue;
        }

        intersect::Ray light_ray{intersection_point, dir_sample.direction};

        auto light_intersection = get_intersection(light_ray);
        if (!light_intersection.has_value()) {
          continue;
        }

        const unsigned light_triangle_idx = light_intersection->info[0];
        const unsigned light_mesh_idx = light_intersection->info[1];

        if (TriangleID{light_mesh_idx, light_triangle_idx} != id) {
          continue;
        }

        const auto &light_data = triangle_data[light_triangle_idx];
        const auto &light_material = materials[light_data.material_idx()];


        const auto &light_mesh = accel.get(light_mesh_idx);

        const intersect::Triangle &light_triangle =
            tri_accels[light_mesh.idx()].get(light_triangle_idx);

        if (!include_lighting(light_ray, light_triangle)) {
          continue;
        }

        const auto light_multiplier =
            material.evaluate_brdf(ray.direction, light_ray.direction, normal);

        intensity += light_material.emission() * material.prob_not_delta() *
                     light_multiplier / dir_sample.prob;
      }

      return intensity;
    };

    if ((!L::performs_samples || count_emission) &&
        include_lighting(ray, triangle)) {
      intensity += multiplier * material.emission();
    }

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
        auto normal_v = abs(outgoing_dir.dot(normal));

        auto brdf_val = material.evaluate_brdf(ray.direction, outgoing_dir, normal);

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
