#pragma once

#include "lib/cuda/reduce.h"
#include "lib/printf_dbg.h"
#include "ray/detail/block_data.h"
#include "ray/detail/intersection/solve.h"
#include "scene/light.h"

namespace ray {
namespace detail {
template <bool is_first, typename Accel>
HOST_DEVICE inline void
raytrace_impl(unsigned block_index, unsigned thread_index,
              const BlockData &block_data, const Accel &accel,
              Span<const scene::ShapeData> shapes,
              SpanSized<const scene::Light> lights,
              Span<const scene::TextureImageRef> textures,
              Span<Eigen::Vector3f> world_space_eyes,
              Span<Eigen::Vector3f> world_space_directions,
              Span<Eigen::Array3f> color_multipliers, Span<scene::Color> colors,
              Span<unsigned> ignores, Span<uint8_t> disables,
              Span<uint8_t> group_disables,
              SpanSized<const unsigned> group_indexes, unsigned num_shapes) {
  uint8_t disable = false;

  auto [x, y, index, outside_bounds] =
      block_data.getIndexes(group_indexes, block_index, thread_index);

  unsigned group_index = group_indexes[block_index];

  if (outside_bounds) {
    disable = true;
  } else {
    {
      if constexpr (!is_first) {
        if (disables[index]) {
          disable = true;
          goto set_disable;
        }
      }

      thrust::optional<BestIntersectionNormalUV> best_normals_uv;

      {
        thrust::optional<BestIntersection> best = thrust::nullopt;

        const auto &world_space_direction = world_space_directions[index];
        const auto &world_space_eye = world_space_eyes[index];

        unsigned ignore_value;

        if constexpr (is_first) {
          ignore_value = std::numeric_limits<unsigned>::max();
        } else {
          ignore_value = ignores[index];
        }

        intersection::solve(
            accel, shapes, world_space_eye, world_space_direction, best,
            ignore_value,
            [&](const thrust::optional<BestIntersection> &new_best) {
              best = optional_min(best, new_best);

              return false;
            });

        if (best.has_value()) {
          // TODO: why required
          auto out = intersection::shapes::get_intersection<true>(
              shapes, best->shape_idx, world_space_eye, world_space_direction);
          best_normals_uv = out;
        } else {
          disable = true;
          goto set_disable;
        }
      }

      auto &best = *best_normals_uv;

      // if this is commented out I get a segfault!?!?!?
      if (best.shape_idx >= num_shapes) {
        printf_dbg(best.shape_idx);
        printf_dbg(num_shapes);
      }
      auto &shape = shapes[best.shape_idx];
      const Eigen::Vector3f world_space_normal =
          (shape.get_object_normal_to_world() * best.intersection.normal)
              .normalized();

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

#if 0
        float point_dist;
        bool is_toward_max = true;
        float min_dist_bound;
        float max_dist_bound;

        Traversal traversal;
        if (use_traversals) {
          auto [traversal_v, dist_v] = traversal_grids_ref.getTraversalFromIdx(
              light_idx, light_direction, world_space_intersection);
          traversal = traversal_v;
          point_dist = std::abs(dist_v);
          is_toward_max = dist_v > 0;
          if (is_toward_max) {
            min_dist_bound = point_dist;
            max_dist_bound = std::numeric_limits<float>::lowest();
          } else {
            max_dist_bound = point_dist;
            min_dist_bound = std::numeric_limits<float>::max();
          }
        }
#endif

        intersection::solve(accel, shapes, world_space_intersection,
                            light_direction, holder, best.shape_idx,
                            [&](const thrust::optional<BestIntersection>
                                    &possible_intersection) {
                              if (possible_intersection.has_value()
#if 0
                  // internal point lights not allowed at the moment
                  &&
                  possible_intersection->intersection < light_distance
#endif
                              ) {
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

#ifdef __CUDA_ARCH__
  uint8_t block_disable = block_reduce_cond(disable, threadIdx.x, blockDim.x);

  if (threadIdx.x == 0) {
    group_disables[group_index] = block_disable;
  }
#else
  group_disables[group_index] = group_disables[group_index] && disable;
#endif
}
} // namespace detail
} // namespace ray
