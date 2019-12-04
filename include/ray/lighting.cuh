#pragma once

#include "lib/bgra.h"
#include "ray/best_intersection.h"
#include "ray/by_type_data.h"
#include "ray/intersect.cuh"
#include "ray/ray_utils.h"
#include "scene/light.h"
#include "scene/shape_data.h"

namespace ray {
namespace detail {
constexpr uint32_t full_mask = 0xffffffff;

__inline__ __device__ uint8_t warp_reduce_cond(uint8_t val) {
  for (unsigned offset = warpSize / 2; offset > 0; offset /= 2) {
    val = __shfl_down_sync(full_mask, val, offset) && val;
  }

  return val;
}

__inline__ __device__ uint8_t block_reduce_cond(uint8_t val,
                                                unsigned thread_block_index,
                                                unsigned thread_block_size) {
  static __shared__ uint8_t shared[32]; // Shared mem for 32 partial sums

  unsigned lane = thread_block_index % warpSize;
  unsigned wid = thread_block_index / warpSize;

  val = warp_reduce_cond(val); // Each warp performs partial reduction

  if (lane == 0) {
    shared[wid] = val; // Write reduced value to shared memory
  }

  __syncthreads(); // Wait for all partial reductions

  // read from shared memory only if that warp existed
  val = (thread_block_index < thread_block_size / warpSize) ? shared[lane] : 1;

  val = warp_reduce_cond(val); // Final reduce within first warp

  return val;
}

__host__ __device__ void compute_color(
    unsigned x, unsigned y, unsigned width, unsigned height,
    const std::array<ByTypeDataRef, scene::shapes_size> &by_type_data,
    Eigen::Vector3f *world_space_eyes, Eigen::Vector3f *world_space_directions,
    unsigned *ignores, Eigen::Array3f *color_multipliers_, uint8_t *disables,
    uint8_t &group_disable,
    const thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::ShapeData *shapes, const scene::Light *lights,
    unsigned num_lights, const scene::TextureImageRef *textures,
    scene::Color *colors, bool use_kd_tree, bool is_first) {
  unsigned index = x + y * width;

  uint8_t disable;
  if (x >= width || y >= height) {
    disable = true;
  } else {
    auto &best_op = best_intersections[index];

    if ((!is_first && disables[index]) || !best_op.has_value()) {
      disable = true;
    } else {
      auto &best = *best_op;
      auto &shape = shapes[best.shape_idx];

      const Eigen::Vector3f world_space_normal =
          (shape.get_object_normal_to_world() * best.intersection.normal)
              .normalized();

      const float intersection = best.intersection.intersection;

      auto &world_space_eye = world_space_eyes[index];
      auto &world_space_direction = world_space_directions[index];

      const auto world_space_intersection =
          world_space_direction * intersection + world_space_eye;

      scene::Color diffuse_lighting(0, 0, 0);
      scene::Color specular_lighting(0, 0, 0);

      auto reflect_over_normal = [&](const Eigen::Vector3f &vec) {
        return (vec + 2.0f * -vec.dot(world_space_normal) * world_space_normal)
            .normalized();
      };

      const auto &material = shape.get_material();

      for (unsigned light_idx = 0; light_idx < num_lights; light_idx++) {
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

        for (auto &data : by_type_data) {
          thrust::optional<BestIntersection> holder = thrust::nullopt;

          auto solve = [&]<scene::Shape shape_type>() {
            solve_general_intersection<shape_type>(
                data, shapes, world_space_intersection, light_direction,
                best.shape_idx, !is_first && disables[index], holder, false,
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
          };
          switch (data.shape_type) {
          case scene::Shape::Sphere:
            solve.template operator()<scene::Shape::Sphere>();
            break;
          case scene::Shape::Cylinder:
            solve.template operator()<scene::Shape::Cylinder>();
            break;
          case scene::Shape::Cube:
            solve.template operator()<scene::Shape::Cube>();
            break;
          case scene::Shape::Cone:
            solve.template operator()<scene::Shape::Cone>();
            break;
          }
          if (shadowed) {
            break;
          }
        }
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

      colors[index] += color_multipliers_[index] * color;

      if (material.reflective[0] >= 1e-5f || material.reflective[1] >= 1e-5f ||
          material.reflective[2] >= 1e-5f) {
        const auto reflection_vec = reflect_over_normal(world_space_direction);
        world_space_eye = world_space_intersection;
        world_space_direction = reflection_vec;
        ignores[index] = best.shape_idx;
        color_multipliers_[index] *= material.reflective;
        disable = false;
      } else {
        disable = true;
      }
    }
    disables[index] = disable;
  }

#if 1
#if defined(__CUDA_ARCH__)
  uint8_t block_disable =
      block_reduce_cond(disable, threadIdx.x, blockDim.x);

  if (threadIdx.x == 0) {
    group_disable = block_disable;
  }
#endif
#endif
}

__global__ void compute_colors(
    unsigned width, unsigned height,
    std::array<ByTypeDataRef, scene::shapes_size> by_type_data,
    Eigen::Vector3f *world_space_eyes, Eigen::Vector3f *world_space_directions,
    unsigned *ignores, Eigen::Array3f *color_multipliers_, uint8_t *disables,
    uint8_t *group_disables, const unsigned *group_indexes,
    unsigned num_blocks_x, unsigned block_dim_x, unsigned block_dim_y,
    const thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::ShapeData *shapes, const scene::Light *lights,
    unsigned num_lights, const scene::TextureImageRef *textures,
    scene::Color *colors, bool use_kd_tree, bool is_first) {
  auto [x, y] = get_indexes(group_indexes, !is_first, num_blocks_x, block_dim_x,
                            block_dim_y);

  compute_color(x, y, width, height, by_type_data, world_space_eyes,
                world_space_directions, ignores, color_multipliers_, disables,
                group_disables[group_indexes[blockIdx.x]], best_intersections,
                shapes, lights, num_lights, textures, colors, use_kd_tree,
                is_first);
}

void compute_colors_cpu(
    unsigned width, unsigned height,
    std::array<ByTypeDataRef, scene::shapes_size> by_type_data,
    Eigen::Vector3f *world_space_eyes, Eigen::Vector3f *world_space_directions,
    unsigned *ignores, Eigen::Array3f *color_multipliers_, uint8_t *disables,
    const thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::ShapeData *shapes, const scene::Light *lights,
    unsigned num_lights, const scene::TextureImageRef *textures,
    scene::Color *colors, bool use_kd_tree, bool is_first) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < height; y++) {
      uint8_t discard;
      compute_color(x, y, width, height, by_type_data, world_space_eyes,
                    world_space_directions, ignores, color_multipliers_,
                    disables, discard, best_intersections, shapes,
                    lights, num_lights, textures, colors, use_kd_tree,
                    is_first);
    }
  }
}

inline __host__ __device__ void float_to_bgra(unsigned x, unsigned y,
                                              unsigned width, unsigned height,
                                              unsigned super_sampling_rate,
                                              const scene::Color *colors,
                                              BGRA *bgra) {
  if (x >= width || y >= height) {
    return;
  }

  unsigned index = x + y * width;

  scene::Color color(0, 0, 0);

  unsigned effective_width = super_sampling_rate * width;
  unsigned start_x = x * super_sampling_rate;
  unsigned start_y = y * super_sampling_rate;

  for (unsigned color_y = start_y; color_y < start_y + super_sampling_rate;
       color_y++) {
    for (unsigned color_x = start_x; color_x < start_x + super_sampling_rate;
         color_x++) {
      color += colors[color_x + color_y * effective_width];
    }
  }

  color /= super_sampling_rate * super_sampling_rate;

  bgra[index].head<3>() = (color * 255.0f + 0.5f)
                              .cast<int>()
                              .cwiseMax(0)
                              .cwiseMin(255)
                              .cast<uint8_t>();
}

__global__ void floats_to_bgras(unsigned width, unsigned height,
                                unsigned num_blocks_x, unsigned block_dim_x,
                                unsigned block_dim_y,
                                unsigned super_sampling_rate,
                                const scene::Color *colors, BGRA *bgra) {
  auto [x, y] = get_non_sparse_indexes(num_blocks_x, block_dim_x, block_dim_y);

  float_to_bgra(x, y, width, height, super_sampling_rate, colors, bgra);
}

void floats_to_bgras_cpu(unsigned width, unsigned height,
                         unsigned super_sampling_rate,
                         const scene::Color *colors, BGRA *bgra) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < width; y++) {
      float_to_bgra(x, y, width, height, super_sampling_rate, colors, bgra);
    }
  }
}
} // namespace detail
} // namespace ray
