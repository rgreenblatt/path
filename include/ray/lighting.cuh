#pragma once

#include "lib/bgra.h"
#include "ray/best_intersection.h"
#include "ray/ray_utils.h"
#include "scene/light.h"
#include "scene/shape_data.h"

namespace ray {
namespace detail {
__host__ __device__ void compute_color(
    unsigned x, unsigned y, unsigned width, unsigned height,
    Eigen::Vector3f *world_space_eyes, Eigen::Vector3f *world_space_directions,
    unsigned *ignores, Eigen::Array3f *color_multipliers_, uint8_t *disables,
    const thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::ShapeData *shapes, const scene::Light *lights,
    unsigned num_lights, scene::Color *colors) {
  unsigned index = x + y * width;

  if (index >= width * height) {
    return;
  }

  auto &best_op = best_intersections[index];
  scene::Color color;
  if (!best_op.has_value()) {
    color = Eigen::Vector3f::Zero();
    disables[index] = true;
  } else {
    auto &best = *best_op;
    auto &shape = shapes[best.shape_idx];
    /* printf("idx: %u\n", best.shape_idx); */

    const Eigen::Vector3f world_space_normal =
        (shape.get_object_normal_to_world() * best.intersection.normal)
            .normalized();

    const float intersection = best.intersection.intersection;

    auto &world_space_eye = world_space_eyes[index];
    auto &world_space_direction = world_space_directions[index];

    const auto world_space_intersection =
        world_space_direction * intersection + world_space_eye;

    scene::Color diffuse_lighting(0);
    scene::Color specular_lighting(0);

    auto reflect_over_normal = [&](const Eigen::Vector3f &vec) {
      return (vec + 2.0f * -vec.dot(world_space_normal) * world_space_normal)
          .normalized();
    };

    const auto &material = shape.get_material();

    for (unsigned light_idx = 0; light_idx < num_lights; light_idx++) {
      Eigen::Vector3f light_direction;
      float attenuation = 1.0f;
      const auto &light = lights[light_idx];
      light.visit([&](auto &&light) {
        using T = std::decay_t<decltype(light)>;
        if constexpr (std::is_same<T, scene::DirectionalLight>::value) {
          light_direction = -light.direction;
        } else {
          light_direction = light.position - world_space_intersection;
          attenuation = 1.0f / (Eigen::Array3f(1, light_direction.norm(),
                                               light_direction.squaredNorm()) *
                                light.attenuation_function)
                                   .sum();
        }
      });

      light_direction.normalize();

      /* bool shadows = true; */
      /* if (shadows && */
      /*     // could be faster to simply check for first intersection */
      /*     solve_intersection(world_space_intersection, light_direction, */
      /*                        boost::make_optional(shape_index)) */
      /*         .is_initialized()) { */
      /*   continue; */
      /* } */

      scene::Color light_factor = light.color * attenuation;

      const float diffuse_factor =
          std::clamp(world_space_normal.dot(light_direction), 0.0f, 1.0f);

      diffuse_lighting += light_factor * diffuse_factor;

#if 0
      printf("diffuse_lighting:\n");
      printf("b: %f, g: %f, r: %f\n", diffuse_lighting[0], diffuse_lighting[1],
             diffuse_lighting[2]);
#endif

      const Eigen::Vector3f reflection_vec =
          reflect_over_normal(-light_direction);

      const float specular_factor = std::pow(
          std::clamp(world_space_direction.dot(-reflection_vec), 0.0f, 1.0f),
          material.shininess);

      specular_lighting += light_factor * specular_factor;

#if 0
      printf("specular_lighting:\n");
      printf("b: %f, g: %f, r: %f\n", specular_lighting[0],
             specular_lighting[1], specular_lighting[2]);
#endif
    }
#if 0
      printf("material.ambient:\n");
      printf("b: %f, g: %f, r: %f\n", material.ambient[0], material.ambient[1],
             material.ambient[2]);
      printf("material.diffuse:\n");
      printf("b: %f, g: %f, r: %f\n", material.diffuse[0], material.diffuse[1],
             material.diffuse[2]);
      printf("material.specular:\n");
      printf("b: %f, g: %f, r: %f\n", material.specular[0],
             material.specular[1], material.specular[2]);
#endif

    color = material.ambient +
            (material.texture_map_index.has_value() ? (1.0f - material.blend)
                                                    : 1.0f) *
                material.diffuse * diffuse_lighting +
            material.specular * specular_lighting;

    if (material.texture_map_index.has_value()) {
      assert(false);
      /* auto tex_lighting = shape.texture.get().sample(textures_, solution.uv);
       */

      /* lighting += shape.material.blend * tex_lighting * diffuse_lighting; */
    }

    if (material.reflective[0] >= 1e-5f || material.reflective[1] >= 1e-5f ||
        material.reflective[2] >= 1e-5f) {
      const auto reflection_vec = reflect_over_normal(world_space_direction);
      world_space_eye = world_space_intersection;
      world_space_direction = reflection_vec;
      ignores[index] = best.shape_idx;
      color_multipliers_[index] = material.reflective;
      disables[index] = false;
    } else {
      disables[index] = true;
    }

#if 0
      printf("color:\n");
      printf("b: %f, g: %f, r: %f\n", color[0], color[1], color[2]);
#endif
  }

  colors[index] += color_multipliers_[index] * color;
}

__global__ void compute_colors(
    unsigned width, unsigned height, Eigen::Vector3f *world_space_eyes,
    Eigen::Vector3f *world_space_directions, unsigned *ignores,
    Eigen::Array3f *color_multipliers_, uint8_t *disables,
    const thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::ShapeData *shapes, const scene::Light *lights,
    unsigned num_lights, scene::Color *colors) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  compute_color(x, y, width, height, world_space_eyes, world_space_directions,
                ignores, color_multipliers_, disables, best_intersections,
                shapes, lights, num_lights, colors);
}

void compute_colors_cpu(
    unsigned width, unsigned height, Eigen::Vector3f *world_space_eyes,
    Eigen::Vector3f *world_space_directions, unsigned *ignores,
    Eigen::Array3f *color_multipliers_, uint8_t *disables,
    const thrust::optional<BestIntersectionNormalUV> *best_intersections,
    const scene::ShapeData *shapes, const scene::Light *lights,
    unsigned num_lights, scene::Color *colors) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < height; y++) {
      compute_color(x, y, width, height, world_space_eyes,
                    world_space_directions, ignores, color_multipliers_,
                    disables, best_intersections, shapes, lights, num_lights,
                    colors);
    }
  }
}

inline __host__ __device__ void float_to_bgra(unsigned x, unsigned y,
                                              unsigned width, unsigned height,
                                              const scene::Color *colors,
                                              BGRA *bgra) {
  unsigned index = x + y * width;

  if (index >= width * height) {
    return;
  }

  bgra[index].head<3>() = (colors[index] * 255.0f + 0.5f)
                              .cast<int>()
                              .cwiseMax(0)
                              .cwiseMin(255)
                              .cast<uint8_t>();
}

__global__ void floats_to_bgras(unsigned width, unsigned height,
                                const scene::Color *colors, BGRA *bgra) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  float_to_bgra(x, y, width, height, colors, bgra);
}

void floats_to_bgras_cpu(unsigned width, unsigned height,
                         const scene::Color *colors, BGRA *bgra) {
  for (unsigned x = 0; x < width; x++) {
    for (unsigned y = 0; y < width; y++) {
      float_to_bgra(x, y, width, height, colors, bgra);
    }
  }
}
} // namespace detail
} // namespace ray
