#include "lib/unified_memory_vector.h"
#include "ray/intersect.cuh"
#include "ray/kdtree.h"
#include "ray/lighting.cuh"
#include "ray/render_impl.h"
#include "ray/render_impl.h"
#include "ray/ray.cuh"

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/combine.hpp>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <chrono>
#include <dbg.h>

namespace ray {
using namespace detail;

unsigned num_blocks(unsigned size, unsigned block_size) {
  return (size + block_size - 1) / block_size;
};

template <ExecutionModel execution_model>
RendererImpl<execution_model>::RendererImpl(unsigned width, unsigned height,
                                            unsigned super_sampling_rate,
                                            unsigned recursive_iterations)
    : effective_width_(width * super_sampling_rate),
      effective_height_(height * super_sampling_rate),
      super_sampling_rate_(super_sampling_rate),
      pixel_size_(effective_width_ * effective_height_),
      recursive_iterations_(recursive_iterations), by_type_data_(invoke([&] {
        auto get_by_type = [&](scene::Shape shape_type) {
          return ByTypeData(shape_type);
        };

        return std::array{
            get_by_type(scene::Shape::Sphere),
            /* get_by_type(scene::Shape::Cylinder), */
            /* get_by_type(scene::Shape::Cube), */
            /* get_by_type(scene::Shape::Cone), */
        };
      })),
      world_space_eyes_(pixel_size_), world_space_directions_(pixel_size_),
      ignores_(pixel_size_), color_multipliers_(pixel_size_),
      disables_(pixel_size_), colors_(pixel_size_), bgra_(width * height) {}

template <typename T> T *to_ptr(thrust::device_vector<T> &vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T> const T *to_ptr(const thrust::device_vector<T> &vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T> T *to_ptr(std::vector<T> &vec) { return vec.data(); }

template <typename T> const T *to_ptr(const std::vector<T> &vec) {
  return vec.data();
}

template <ExecutionModel execution_model>
ByTypeDataRef RendererImpl<execution_model>::ByTypeData::initialize(
    const scene::Scene &scene, scene::ShapeData *shapes) {
  const uint16_t num_shapes = scene.getNumShapes();

  auto kdtree = construct_kd_tree(shapes, num_shapes);
  nodes.resize(kdtree.size());
  std::copy(kdtree.begin(), kdtree.end(), nodes.begin());

  return ByTypeDataRef(nodes.data(), nodes.size(), shape_type, 0, num_shapes);
}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(
    const scene::Scene &scene, BGRA *pixels,
    const scene::Transform &m_film_to_world, bool use_kd_tree,
    bool show_times) {
  namespace chr = std::chrono;

  const auto lights = scene.getLights();
  const unsigned num_lights = scene.getNumLights();
  const unsigned num_pixels = effective_width_ * effective_height_;
  const auto textures = scene.getTextures();

  const unsigned block_dim_x = 32;
  const unsigned block_dim_y = 8;

  const unsigned num_blocks_x = num_blocks(effective_width_, block_dim_x);
  const unsigned num_blocks_y = num_blocks(effective_height_, block_dim_y);

  const unsigned general_num_blocks = num_blocks_x * num_blocks_y;
  const unsigned general_block_size = block_dim_x * block_dim_y;

  group_disables_.resize(general_num_blocks);
  group_indexes_.resize(general_num_blocks);

  unsigned current_num_blocks = general_num_blocks;

  const Eigen::Vector3f world_space_eye = m_film_to_world.translation();

  const scene::Color initial_multiplier = scene::Color::Ones();

  const scene::Color initial_color = scene::Color::Zero();

  const auto start_fill = chr::high_resolution_clock::now();

  // could be made async until...
  if constexpr (execution_model == ExecutionModel::GPU) {
    const unsigned fill_block_size = 256;
    const unsigned fill_num_blocks = num_blocks(num_pixels, fill_block_size);
    fill<<<fill_num_blocks, fill_block_size>>>(to_ptr(world_space_eyes_),
                                               num_pixels, world_space_eye);
    fill<<<fill_num_blocks, fill_block_size>>>(to_ptr(color_multipliers_),
                                               num_pixels, initial_multiplier);
    fill<<<fill_num_blocks, fill_block_size>>>(to_ptr(colors_), num_pixels,
                                               initial_color);

    initial_world_space_directions<<<general_num_blocks, general_block_size>>>(
        effective_width_, effective_height_, num_blocks_x, block_dim_x,
        block_dim_y, world_space_eye, m_film_to_world,
        to_ptr(world_space_directions_));

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
  } else {
    std::fill(world_space_eyes_.begin(), world_space_eyes_.end(),
              world_space_eye);
    std::fill(color_multipliers_.begin(), color_multipliers_.end(),
              initial_multiplier);
    std::fill(colors_.begin(), colors_.end(), initial_color);

    initial_world_space_directions_cpu(effective_width_, effective_height_,
                                       world_space_eye, m_film_to_world,
                                       world_space_directions_.data());
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - start_fill)
            .count());
  }

  const unsigned num_shapes = scene.numShapes();
  ManangedMemVec<scene::ShapeData> shapes(num_shapes);

  {
    auto start_shape = scene.getShapes();
    std::copy(start_shape, start_shape + num_shapes, shapes.begin());
  }

  const auto start_kdtree = chr::high_resolution_clock::now();

  std::array<ByTypeDataRef, scene::shapes_size> by_type_data_gpu;

#pragma omp parallel for
  for (unsigned i = 0; i < 1; i++) {
    by_type_data_gpu[i] = by_type_data_[i].initialize(scene, shapes.data());
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - start_kdtree)
            .count());
  }

  const Eigen::Vector3f camera_loc = m_film_to_world.translation();

  auto get_projection_info = [&](bool is_loc,
                                 const Eigen::Vector3f &loc_or_dir) {
    auto &last_node = by_type_data_[0].nodes[by_type_data_[0].nodes.size() - 1];
    const auto &min_bound = last_node.get_contents().get_min_bound();
    const auto &max_bound = last_node.get_contents().get_max_bound();
    const auto center = (max_bound + min_bound) / 2;
    const auto dims = max_bound - min_bound;
    const auto dir = is_loc ? loc_or_dir - center : loc_or_dir;
    const auto normalized_directions = dir.array() / dims.array();
    unsigned axis;
    float max_axis_v = std::numeric_limits<float>::lowest();
    for (unsigned test_axis = 0; test_axis < 3; test_axis++) {
      float abs_v = std::abs(normalized_directions[test_axis]);
      if (abs_v > max_axis_v) {
        axis = test_axis;
        max_axis_v = abs_v;
      }
    }

    bool is_min = normalized_directions[axis] < 0;

    float projection_value = is_min ? min_bound[axis] : max_bound[axis];

    Eigen::Vector2f min_point(std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max());
    Eigen::Vector2f max_point(std::numeric_limits<float>::lowest(),
                              std::numeric_limits<float>::lowest());
    std::vector<KDTreeNode<ProjectedAABBInfo>> projected_aabb(
        by_type_data_[0].nodes.size());
    std::transform(by_type_data_[0].nodes.begin(), by_type_data_[0].nodes.end(),
                   projected_aabb.begin(), [&](const KDTreeNode<AABB> &node) {
                     return node.transform([&](const AABB &aa_bb) {
                       auto out =
                           aa_bb
                               .project_to_axis(is_loc, axis, projection_value,
                                                loc_or_dir)
                               .get_info();
                       min_point = min_point.cwiseMin(out.flattened_min);
                       max_point = max_point.cwiseMax(out.flattened_max);

                       return out;
                     });
                   });

    return std::make_tuple(axis, projection_value, min_point, max_point,
                           projected_aabb);
  };

  unsigned num_blocks_light_x = 1;
  unsigned num_blocks_light_y = 1;

  if constexpr (execution_model == ExecutionModel::GPU) {
    auto [axis, value_to_project_to, min_p, max_p, projected_aabb] =
        get_projection_info(true, camera_loc);

    const auto start_traversal_grid = chr::high_resolution_clock::now();

    auto [traversals, disables, actions] = get_traversal_grid_from_transform(
        projected_aabb, effective_width_, effective_height_, m_film_to_world,
        block_dim_x, block_dim_y, num_blocks_x, num_blocks_y, axis,
        value_to_project_to);

    std::copy(disables.begin(), disables.end(), group_disables_.begin());

    camera_traversals_.resize(traversals.size());
    camera_actions_.resize(actions.size());

    thrust::copy(traversals.begin(), traversals.end(),
                 camera_traversals_.begin());
    thrust::copy(actions.begin(), actions.end(), camera_actions_.begin());

    light_traversals_cpu_.resize(num_blocks_light_x * num_blocks_light_y *
                                 num_lights);
    light_actions_cpu_.clear();
    light_traversal_data_cpu_.resize(num_lights);
    unsigned last_offset = 0;

    for (unsigned light_idx = 0; light_idx < num_lights; light_idx++) {
      const auto &light = lights[light_idx];
      light.visit([&](auto &&light_data) {
        using T = std::decay_t<decltype(light_data)>;
        auto copy_in_light = [&](bool is_loc,
                                 const Eigen::Vector3f &loc_or_dir) {
          auto [axis, value_to_project_to, min_p, max_p, projected_aabb] =
              get_projection_info(is_loc, loc_or_dir);

          auto [traversals, _, actions] = get_traversal_grid_from_bounds(
              projected_aabb, min_p, max_p, num_blocks_light_x,
              num_blocks_light_y);

          std::copy(traversals.data(), traversals.data() + traversals.size(),
                    light_traversals_cpu_.begin() +
                        num_blocks_light_x * num_blocks_light_y * light_idx);
          light_actions_cpu_.insert(light_actions_cpu_.begin() + last_offset,
                                    actions.begin(), actions.end());
          light_traversal_data_cpu_[light_idx] = LightTraversalData(
              last_offset, min_p, max_p, axis, value_to_project_to);
          last_offset += actions.size();
        };
        if constexpr (std::is_same<T, scene::DirectionalLight>::value) {
          copy_in_light(false, light_data.direction);
        } else {
          copy_in_light(true, light_data.position);
        }
      });
    }

    light_traversals_.resize(light_traversals_cpu_.size());
    thrust::copy(light_traversals_cpu_.data(),
                 light_traversals_cpu_.data() + light_traversals_cpu_.size(),
                 light_traversals_.begin());

    light_actions_.resize(light_actions_cpu_.size());
    thrust::copy(light_actions_cpu_.data(),
                 light_actions_cpu_.data() + light_actions_cpu_.size(),
                 light_actions_.begin());

    light_traversal_data_.resize(light_traversal_data_cpu_.size());
    thrust::copy(light_traversal_data_cpu_.data(),
                 light_traversal_data_cpu_.data() +
                     light_traversal_data_cpu_.size(),
                 light_traversal_data_.begin());

    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(
              chr::high_resolution_clock::now() - start_traversal_grid)
              .count());
    }
  }

  for (unsigned depth = 0; depth < recursive_iterations_; depth++) {
    bool is_first = depth == 0;

    // do this on gpu????
    if (execution_model == ExecutionModel::GPU) {
      current_num_blocks = 0;

      for (unsigned i = 0; i < group_disables_.size(); i++) {
        if (!group_disables_[i]) {
          group_indexes_[current_num_blocks] = i;
          current_num_blocks++;
        }
      }
    }

    const auto start_intersect = chr::high_resolution_clock::now();

    /* #pragma omp parallel for */
    for (auto &data : by_type_data_gpu) {
      if constexpr (execution_model == ExecutionModel::GPU) {
        if (current_num_blocks != 0) {
          raytrace<<<current_num_blocks, general_block_size>>>(
              effective_width_, effective_height_, num_blocks_x, block_dim_x,
              block_dim_y, data, to_ptr(camera_traversals_),
              to_ptr(camera_actions_), to_ptr(light_traversals_),
              to_ptr(light_actions_), to_ptr(light_traversal_data_),
              num_blocks_light_x, num_blocks_light_y, shapes.data(), lights,
              num_lights, textures, to_ptr(world_space_eyes_),
              to_ptr(world_space_directions_), to_ptr(color_multipliers_),
              to_ptr(colors_), to_ptr(ignores_), to_ptr(disables_),
              group_disables_.data(), group_indexes_.data(), is_first,
              use_kd_tree);
        }
      } else {
        raytrace_cpu(effective_width_, effective_height_, data, shapes.data(),
                     lights, num_lights, textures, world_space_eyes_.data(),
                     world_space_directions_.data(), color_multipliers_.data(),
                     colors_.data(), ignores_.data(), disables_.data(),
                     is_first, use_kd_tree);
      }
    }

    CUDA_ERROR_CHK(cudaDeviceSynchronize());

    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(
              chr::high_resolution_clock::now() - start_intersect)
              .count());
    }
  }

  const unsigned width = effective_width_ / super_sampling_rate_;
  const unsigned height = effective_height_ / super_sampling_rate_;

  if constexpr (execution_model == ExecutionModel::GPU) {
    const auto start_convert = chr::high_resolution_clock::now();
    floats_to_bgras<<<general_num_blocks, general_block_size>>>(
        width, height, num_blocks_x, block_dim_x, block_dim_y,
        super_sampling_rate_, to_ptr(colors_), to_ptr(bgra_));

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(
              chr::high_resolution_clock::now() - start_convert)
              .count());
    }

    const auto start_copy_to_cpu = chr::high_resolution_clock::now();
    thrust::copy(bgra_.begin(), bgra_.end(), pixels);
    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(
              chr::high_resolution_clock::now() - start_copy_to_cpu)
              .count());
    }
  } else {
    floats_to_bgras_cpu(width, height, super_sampling_rate_, colors_.data(),
                        pixels);
  }
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace ray
