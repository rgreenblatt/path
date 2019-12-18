#pragma once

#include "lib/span.h"
#include "ray/best_intersection.h"
#include "ray/block_data.h"
#include "ray/by_type_data.h"
#include "ray/cuda_ray_utils.cuh"
#include "ray/intersect.cuh"
#include "ray/ray_utils.h"

namespace ray {
namespace detail {
__host__ __device__ inline void raytrace_impl(
    unsigned block_index, unsigned thread_index, const BlockData &block_data,
    const ByTypeDataRef &by_type_data,
    const TraversalGridsRef &traversal_grids_ref,
    Span<const scene::ShapeData> shapes, Span<const scene::Light, false> lights,
    Span<const scene::TextureImageRef> textures,
    Span<Eigen::Vector3f> world_space_eyes,
    Span<Eigen::Vector3f> world_space_directions,
    Span<Eigen::Array3f> color_multipliers, Span<scene::Color> colors,
    Span<unsigned> ignores, Span<uint8_t> disables,
    Span<uint8_t> group_disables, Span<const unsigned, false> group_indexes,
    bool is_first, bool use_kd_tree, bool use_traversals) {
  uint8_t disable = false;

  auto [x, y, index, outside_bounds] =
      block_data.getIndexes(group_indexes, block_index, thread_index);

  unsigned group_index = group_indexes[block_index];

  if (outside_bounds) {
    disable = true;
  } else {
    {
      if (!is_first && disables[index]) {
        disable = true;
        goto set_disable;
      }

      thrust::optional<BestIntersectionNormalUV> best_normals_uv;

      {
        thrust::optional<BestIntersection> best = thrust::nullopt;

        const auto &world_space_direction = world_space_directions[index];
        const auto &world_space_eye = world_space_eyes[index];

        bool use_camera_traversals = use_traversals && is_first;

        solve_general_intersection(
            by_type_data,
            use_camera_traversals
                ? traversal_grids_ref.getCameraTraversal(group_index)
                : Traversal(),
            traversal_grids_ref.actions, shapes, world_space_eye,
            world_space_direction, ignores[index], disables[index], best,
            is_first, use_camera_traversals, use_kd_tree,
            [&](const thrust::optional<BestIntersection> &new_best) {
              best = optional_min(best, new_best);

              return false;
            });

        if (best.has_value()) {
          // TODO: why required
          auto out = get_shape_intersection<true>(
              shapes, best->shape_idx, world_space_eye, world_space_direction);
          best_normals_uv = out;
        } else {
          disable = true;
          goto set_disable;
        }
      }

      auto &best = *best_normals_uv;
      auto &shape = shapes[best.shape_idx];
      Eigen::Vector3f prod =
          shape.get_object_normal_to_world() * best.intersection.normal;
      const Eigen::Vector3f world_space_normal = prod.normalized();

      const float intersection = best.intersection.intersection;

      auto &world_space_eye = world_space_eyes[index];
      auto &world_space_direction = world_space_directions[index];

      const auto world_space_intersection =
          (world_space_direction * intersection + world_space_eye).eval();

      scene::Color diffuse_lighting(0, 0, 0);
      scene::Color specular_lighting(0, 0, 0);

      auto reflect_over_normal = [&](const Eigen::Vector3f &vec) {
        return (vec + 2.0f * -vec.dot(world_space_normal) * world_space_normal)
            .normalized()
            .eval();
      };

      const auto &material = shape.get_material();

      for (unsigned light_idx = 0; light_idx < lights.size(); light_idx++) {
        Eigen::Vector3f light_direction;
        float light_distance = std::numeric_limits<float>::max();
        float attenuation = 1.0f;
        const auto &light = lights[light_idx];
        light.visit([&](auto &&light) {
          using T = std::decay_t<decltype(light)>;
          if constexpr (std::is_same<T, scene::DirectionalLight>::value) {
            light_direction = -light.direction;
          } else {
            light_direction = light.position - world_space_intersection;
            light_distance = light_direction.norm();
            attenuation =
                1.0f / ((Eigen::Array3f(1, light_distance,
                                        light_distance * light_distance) *
                         light.attenuation_function)
                            .sum());
          }
        });

        light_direction.normalize();

#if 1
        bool shadowed = false;

        thrust::optional<BestIntersection> holder = thrust::nullopt;

        auto traversal = use_traversals ? traversal_grids_ref.getLightTraversal(
                                              light_idx, light_direction,
                                              world_space_intersection)
                                        : Traversal();

        solve_general_intersection(
            by_type_data, traversal, traversal_grids_ref.actions, shapes,
            world_space_intersection, light_direction, best.shape_idx,
            !is_first && disables[index], holder, false, use_traversals,
            use_kd_tree,
            [&](const thrust::optional<BestIntersection>
                    &possible_intersection) {
              if (possible_intersection.has_value() &&
                  possible_intersection->intersection < light_distance) {
                shadowed = true;
                return true;
              }

              return false;
            });

        if (shadowed) {
          continue;
        }
#endif

        scene::Color light_factor = light.color * attenuation;

        const float diffuse_factor =
            std::clamp(world_space_normal.dot(light_direction), 0.0f, 1.0f);

        diffuse_lighting += light_factor * diffuse_factor;

        const Eigen::Vector3f reflection_vec =
            reflect_over_normal(-light_direction);

        const float specular_factor = std::pow(
            std::clamp(world_space_direction.dot(-reflection_vec), 0.0f, 1.0f),
            material.shininess);

        specular_lighting += light_factor * specular_factor;
      }

      auto get_blend_multiplier = [&](float blend) {
        return material.texture_data.has_value() ? (1.0f - blend) : 1.0f;
      };

      scene::Color color =
          get_blend_multiplier(material.ambient_blend) * material.ambient +
          get_blend_multiplier(material.diffuse_blend) * material.diffuse *
              diffuse_lighting +
          material.specular * specular_lighting;

      if (material.texture_data.has_value()) {
        auto tex_lighting =
            material.texture_data->sample(textures, best.intersection.uv);

        color += material.diffuse_blend * tex_lighting * diffuse_lighting;
        color += material.ambient_blend * tex_lighting;
      }

      colors[index] += color_multipliers[index] * color;

      if (material.reflective[0] >= 1e-5f || material.reflective[1] >= 1e-5f ||
          material.reflective[2] >= 1e-5f) {
        const auto reflection_vec = reflect_over_normal(world_space_direction);
        world_space_eye = world_space_intersection;
        world_space_direction = reflection_vec;
        ignores[index] = best.shape_idx;
        color_multipliers[index] *= material.reflective;
      } else {
        disable = true;
      }
    }

  set_disable:
    disables[index] = disable;
  }

#if defined(__CUDA_ARCH__)
  uint8_t block_disable = block_reduce_cond(disable, threadIdx.x, blockDim.x);

  if (threadIdx.x == 0) {
    group_disables[group_index] = block_disable;
  }
#else
  group_disables[group_index] = group_disables[group_index] && disable;
#endif
}

__global__ void raytrace(const BlockData block_data,
                         const ByTypeDataRef by_type_data,
                         const TraversalGridsRef traversal_grids_ref,
                         Span<const scene::ShapeData> shapes,
                         Span<const scene::Light, false> lights,
                         Span<const scene::TextureImageRef> textures,
                         Span<Eigen::Vector3f> world_space_eyes,
                         Span<Eigen::Vector3f> world_space_directions,
                         Span<Eigen::Array3f> color_multipliers,
                         Span<scene::Color> colors, Span<unsigned> ignores,
                         Span<uint8_t> disables, Span<uint8_t> group_disables,
                         Span<const unsigned, false> group_indexes,
                         bool is_first, bool use_kd_tree, bool use_traversals) {
  raytrace_impl(blockIdx.x, threadIdx.x, block_data, by_type_data,
                traversal_grids_ref, shapes, lights, textures, world_space_eyes,
                world_space_directions, color_multipliers, colors, ignores,
                disables, group_disables, group_indexes, is_first, use_kd_tree,
                use_traversals);
}

inline void raytrace_cpu(const BlockData block_data,
                         const ByTypeDataRef by_type_data,
                         const TraversalGridsRef traversal_grids_ref,
                         Span<const scene::ShapeData> shapes,
                         Span<const scene::Light, false> lights,
                         Span<const scene::TextureImageRef> textures,
                         Span<Eigen::Vector3f> world_space_eyes,
                         Span<Eigen::Vector3f> world_space_directions,
                         Span<Eigen::Array3f> color_multipliers,
                         Span<scene::Color> colors, Span<unsigned> ignores,
                         Span<uint8_t> disables, Span<uint8_t> group_disables,
                         Span<const unsigned, false> group_indexes,
                         bool is_first, bool use_kd_tree, bool use_traversals) {
  for (unsigned block_index = 0; block_index < group_indexes.size();
       block_index++) {
    for (unsigned thread_index = 0;
         thread_index < block_data.generalBlockSize(); thread_index++) {
      raytrace_impl(block_index, thread_index, block_data, by_type_data,
                    traversal_grids_ref, shapes, lights, textures,
                    world_space_eyes, world_space_directions, color_multipliers,
                    colors, ignores, disables, group_disables, group_indexes,
                    is_first, use_kd_tree, use_traversals);
    }
  }
}
} // namespace detail
} // namespace ray
