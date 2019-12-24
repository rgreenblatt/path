#include "ray/render_impl.h"
#include "ray/render_impl_utils.h"
#include "ray/intersect.cuh"

namespace ray {
using namespace detail;
__host__ __device__ inline void raytrace_impl(
    unsigned block_index, unsigned thread_index, const BlockData &block_data,
    const KDTreeNodesRef &kdtree_nodes_ref,
    const TraversalGridsRef &traversal_grids_ref,
    Span<const scene::ShapeData> shapes, Span<const scene::Light, false> lights,
    Span<const scene::TextureImageRef> textures,
    Span<Eigen::Vector3f> world_space_eyes,
    Span<Eigen::Vector3f> world_space_directions,
    Span<Eigen::Array3f> color_multipliers, Span<scene::Color> colors,
    Span<unsigned> ignores, Span<uint8_t> disables,
    Span<uint8_t> group_disables, Span<const unsigned, false> group_indexes,
    bool is_first, bool use_kd_tree, bool use_traversals,
    bool use_traversal_dists) {
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
        
        float point_dist;
        bool is_toward_max = true;
        float min_dist_bound;
        float max_dist_bound;

        Traversal traversal;
        if (use_traversals) {
          /* if (is_first) { */
          /*   traversal = traversal_grids_ref.getCameraTraversal(group_index); */
          /* } else { */
            auto [traversal_v, dist_v] =
                traversal_grids_ref.getGeneralTraversal(world_space_direction,
                                                        world_space_eye);
            traversal = traversal_v;
            point_dist = std::abs(dist_v);
            is_toward_max = dist_v > 0;
            max_dist_bound = std::numeric_limits<float>::lowest();
            min_dist_bound = std::numeric_limits<float>::max();
            if (!is_first) {
              if (is_toward_max) {
                min_dist_bound = point_dist;
              } else {
                max_dist_bound = point_dist;
              }
            }
          /* } */
        }

        /* bool use_traversal_dists_now = !is_first && use_traversal_dists; */
        bool use_traversal_dists_now = use_traversal_dists;

        solve_general_intersection(
            kdtree_nodes_ref, traversal,
            is_toward_max ? traversal_grids_ref.max_sorted_actions()
                          : traversal_grids_ref.min_sorted_actions(),
            shapes, world_space_eye, world_space_direction, ignores[index],
            disables[index], best, is_first, use_traversals,
            use_traversal_dists_now, min_dist_bound, max_dist_bound,
            is_toward_max, use_kd_tree,
            [&](const thrust::optional<BestIntersection> &new_best) {
              best = optional_min(best, new_best);
              if (new_best.has_value() && use_traversal_dists_now) {
                if (is_toward_max) {
                  max_dist_bound = point_dist - best->intersection;
                } else {
                  min_dist_bound = point_dist + best->intersection;
                }
              }

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

        solve_general_intersection(
            kdtree_nodes_ref, traversal,
            is_toward_max ? traversal_grids_ref.max_sorted_actions()
                          : traversal_grids_ref.min_sorted_actions(),
            shapes, world_space_intersection, light_direction, best.shape_idx,
            !is_first && disables[index], holder, false, use_traversals,
            use_traversal_dists, min_dist_bound, max_dist_bound, is_toward_max,
            use_kd_tree,
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
                         const KDTreeNodesRef kdtree_nodes_ref,
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
                         bool is_first, bool use_kd_tree, bool use_traversals,
                         bool use_traversal_dists) {
  raytrace_impl(blockIdx.x, threadIdx.x, block_data, kdtree_nodes_ref,
                traversal_grids_ref, shapes, lights, textures, world_space_eyes,
                world_space_directions, color_multipliers, colors, ignores,
                disables, group_disables, group_indexes, is_first, use_kd_tree,
                use_traversals, use_traversal_dists);
}

inline void raytrace_cpu(
    const BlockData block_data, const KDTreeNodesRef kdtree_nodes_ref,
    const TraversalGridsRef traversal_grids_ref,
    Span<const scene::ShapeData> shapes, Span<const scene::Light, false> lights,
    Span<const scene::TextureImageRef> textures,
    Span<Eigen::Vector3f> world_space_eyes,
    Span<Eigen::Vector3f> world_space_directions,
    Span<Eigen::Array3f> color_multipliers, Span<scene::Color> colors,
    Span<unsigned> ignores, Span<uint8_t> disables,
    Span<uint8_t> group_disables, Span<const unsigned, false> group_indexes,
    bool is_first, bool use_kd_tree, bool use_traversals,
    bool use_traversal_dists, unsigned current_num_blocks) {
  for (unsigned block_index = 0; block_index < current_num_blocks;
       block_index++) {
    for (unsigned thread_index = 0;
         thread_index < block_data.generalBlockSize(); thread_index++) {
      raytrace_impl(block_index, thread_index, block_data, kdtree_nodes_ref,
                    traversal_grids_ref, shapes, lights, textures,
                    world_space_eyes, world_space_directions, color_multipliers,
                    colors, ignores, disables, group_disables, group_indexes,
                    is_first, use_kd_tree, use_traversals, use_traversal_dists);
    }
  }
}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::raytrace_pass(
    bool is_first, bool use_kd_tree, bool use_traversals, bool use_traversal_dists,
    unsigned current_num_blocks, Span<const scene::ShapeData, false> shapes,
    Span<const scene::Light, false> lights,
    Span<const scene::TextureImageRef> textures,
    const TraversalGridsRef &traversal_grids_ref) {
  KDTreeNodesRef kdtree_nodes_ref(kdtree_nodes_.data(), kdtree_nodes_.size(),
                                  shapes.size());

  unsigned general_block_size = block_data_.generalBlockSize();

  auto shapes_span = Span<const scene::ShapeData>(shapes.data(), shapes.size());

  if constexpr (execution_model == ExecutionModel::GPU) {
    if (current_num_blocks != 0) {
      raytrace<<<current_num_blocks, general_block_size>>>(
          block_data_, kdtree_nodes_ref, traversal_grids_ref, shapes_span,
          lights, textures, to_span(world_space_eyes_),
          to_span(world_space_directions_), to_span(color_multipliers_),
          to_span(colors_), to_span(ignores_), to_span(disables_),
          to_span(group_disables_),
          Span<const unsigned, false>(group_indexes_.data(),
                                      group_indexes_.size()),
          is_first, use_kd_tree, use_traversals, use_traversal_dists);
    }
  } else {
    raytrace_cpu(block_data_, kdtree_nodes_ref, traversal_grids_ref,
                 shapes_span, lights, textures, to_span(world_space_eyes_),
                 to_span(world_space_directions_), to_span(color_multipliers_),
                 to_span(colors_), to_span(ignores_), to_span(disables_),
                 to_span(group_disables_),
                 Span<const unsigned, false>(group_indexes_.data(),
                                             group_indexes_.size()),
                 is_first, use_kd_tree, use_traversals, use_traversal_dists,
                 current_num_blocks);
  }

  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace ray
