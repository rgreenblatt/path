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

  auto find_projection_info = [&](const Eigen::Vector3f &loc) {
    auto &last_node = by_type_data_[0].nodes[by_type_data_[0].nodes.size() - 1];
    const auto &min_bound = last_node.get_contents().get_min_bound();
    const auto &max_bound = last_node.get_contents().get_max_bound();
    const auto center = (max_bound + min_bound) / 2;
    const auto dims = max_bound - min_bound;
    const auto dir = loc - center;
    const auto normalized_directions = dir.array() / dims.array();
    unsigned max_axis;
    float max_axis_v = std::numeric_limits<float>::lowest();
    for (unsigned axis = 0; axis < 3; axis++) {
      float abs_v = std::abs(normalized_directions[axis]);
      if (abs_v > max_axis_v) {
        max_axis = axis;
        max_axis_v = abs_v;
      }
    }

    float projection_value = normalized_directions[max_axis] > 0
                                 ? max_bound[max_axis]
                                 : min_bound[max_axis];

    return std::make_tuple(max_axis, projection_value);
  };

  auto [axis_v, value_to_project_to_v] = find_projection_info(camera_loc);
  const uint8_t axis = axis_v;
  const float value_to_project_to = value_to_project_to_v;

  std::vector<KDTreeNode<ProjectedAABBInfo>> projection_nodes(
      by_type_data_[0].nodes.size());

  std::transform(by_type_data_[0].nodes.begin(), by_type_data_[0].nodes.end(),
                 projection_nodes.begin(), [&](const KDTreeNode<AABB> &node) {
                   return node.transform([&](const AABB &aa_bb) {
                     return aa_bb
                         .project_to_axis(axis, value_to_project_to, camera_loc)
                         .get_info();
                   });
                 });

  const auto start_traversal_grid = chr::high_resolution_clock::now();

  auto [traversals, disables, actions] =
      get_traversal_grid(projection_nodes, effective_width_, effective_height_,
                         m_film_to_world, block_dim_x, block_dim_y,
                         num_blocks_x, num_blocks_y, axis, value_to_project_to);

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - start_traversal_grid)
            .count());
  }

  std::copy(disables.begin(), disables.end(), group_disables_.begin());

  if constexpr (execution_model == ExecutionModel::GPU) {
    camera_traversals_.resize(traversals.size());
    camera_actions_.resize(actions.size());

    thrust::copy(traversals.begin(), traversals.end(),
                 camera_traversals_.begin());
    thrust::copy(actions.begin(), actions.end(), camera_actions_.begin());
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
              to_ptr(camera_actions_), shapes.data(), lights, num_lights,
              textures, to_ptr(world_space_eyes_),
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
