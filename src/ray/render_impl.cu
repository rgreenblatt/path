#include "lib/unified_memory_vector.h"
#include "ray/action_grid.h"
#include "ray/intersect.cuh"
#include "ray/kdtree.h"
#include "ray/lighting.cuh"
#include "ray/projection_impl.h"
#include "ray/ray.cuh"
#include "ray/render_impl.h"
#include "ray/scene_ref.h"
#include "scene/camera.h"

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/combine.hpp>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <chrono>
#include <dbg.h>

namespace ray {
using namespace detail;
template <ExecutionModel execution_model>
RendererImpl<execution_model>::RendererImpl(unsigned width, unsigned height,
                                            unsigned super_sampling_rate,
                                            unsigned recursive_iterations,
                                            std::unique_ptr<scene::Scene> &s)
    : block_data_(super_sampling_rate * width, super_sampling_rate * height, 32,
                  8),
      super_sampling_rate_(super_sampling_rate),
      recursive_iterations_(recursive_iterations), scene_(std::move(s)),
      by_type_data_(invoke([&] {
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
      world_space_eyes_(block_data_.totalSize()),
      world_space_directions_(block_data_.totalSize()),
      ignores_(block_data_.totalSize()),
      color_multipliers_(block_data_.totalSize()),
      disables_(block_data_.totalSize()), colors_(block_data_.totalSize()),
      bgra_(width * height) {}

template <typename T> T *to_ptr(thrust::device_vector<T> &vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T> const T *to_ptr(const thrust::device_vector<T> &vec) {
  return thrust::raw_pointer_cast(vec.data());
}

template <typename T> T *to_ptr(std::vector<T> &vec) { return vec.data(); }

template <typename T> Span<T> to_span(thrust::device_vector<T> &vec) {
  return Span(thrust::raw_pointer_cast(vec.data()), vec.size());
}

template <typename T>
Span<const T> to_const_span(const thrust::device_vector<T> &vec) {
  return Span(thrust::raw_pointer_cast(vec.data()), vec.size());
}

template <typename T, typename A> Span<T> to_span(std::vector<T, A> &vec) {
  return Span(vec.data(), vec.size());
}

template <typename T, typename A>
Span<const T> to_const_span(const std::vector<T, A> &vec) {
  return Span(vec.data(), vec.size());
}

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
    BGRA *pixels, const scene::Transform &m_film_to_world,
    const Eigen::Projective3f &world_to_film,
    bool use_kd_tree, bool use_traversals, bool show_times) {
  namespace chr = std::chrono;

  const auto lights = scene_->getLights();
  const unsigned num_lights = scene_->getNumLights();
  const auto textures = scene_->getTextures();
  const unsigned num_textures = scene_->getNumTextures();

  const unsigned general_num_blocks = block_data_.generalNumBlocks();
  const unsigned general_block_size = block_data_.generalBlockSize();

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
    const unsigned fill_num_blocks =
        num_blocks(block_data_.totalSize(), fill_block_size);
    fill<<<fill_num_blocks, fill_block_size>>>(
        to_ptr(world_space_eyes_), block_data_.totalSize(), world_space_eye);
    fill<<<fill_num_blocks, fill_block_size>>>(to_ptr(color_multipliers_),
                                               block_data_.totalSize(),
                                               initial_multiplier);
    fill<<<fill_num_blocks, fill_block_size>>>(
        to_ptr(colors_), block_data_.totalSize(), initial_color);

    initial_world_space_directions<<<general_num_blocks, general_block_size>>>(
        block_data_, world_space_eye, m_film_to_world,
        to_span(world_space_directions_));

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
  } else {
    std::fill(world_space_eyes_.begin(), world_space_eyes_.end(),
              world_space_eye);
    std::fill(color_multipliers_.begin(), color_multipliers_.end(),
              initial_multiplier);
    std::fill(colors_.begin(), colors_.end(), initial_color);

    initial_world_space_directions_cpu(block_data_, world_space_eye,
                                       m_film_to_world,
                                       to_span(world_space_directions_));
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - start_fill)
            .count());
  }

  const unsigned num_shapes = scene_->getNumShapes();
  ManangedMemVec<scene::ShapeData> shapes(num_shapes);

  {
    auto start_shape = scene_->getShapes();
    std::copy(start_shape, start_shape + num_shapes, shapes.begin());
  }

  const auto start_kdtree = chr::high_resolution_clock::now();

  std::array<ByTypeDataRef, scene::shapes_size> by_type_data_gpu;

  for (unsigned i = 0; i < 1; i++) {
    by_type_data_gpu[i] = by_type_data_[i].initialize(*scene_, shapes.data());
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - start_kdtree)
            .count());
  }

  const auto start_traversal_grid = chr::high_resolution_clock::now();

  TraversalGrid camera_grid(TriangleProjector(world_to_film), shapes.data(),
                            shapes.size(), Eigen::Array2f(-1, -1),
                            Eigen::Array2f(1, 1), block_data_.num_blocks_x,
                            block_data_.num_blocks_y, false, true);

  unsigned traversals_offset = 0;

  traversals_cpu_.clear();
  actions_cpu_.clear();

  camera_grid.copy_into(traversals_cpu_, actions_cpu_);

  if (use_traversals) {
    std::transform(
        traversals_cpu_.begin(), traversals_cpu_.end(), group_disables_.begin(),
        [&](const Traversal &traversal) { return traversal.size == 0; });
  } else {
    for (auto &disable : group_disables_) {
      disable = false;
    }
  }

  traversals_offset = traversals_cpu_.size();

  traversal_data_cpu_.resize(num_lights);

  unsigned projection_index = 0;

  unsigned num_division_light_x = 32;
  unsigned num_division_light_y = 32;

  auto add_projection = [&](bool is_loc, const Eigen::Vector3f &loc_or_dir) {
    auto &max_bound = scene_->getMaxBound();
    auto &min_bound = scene_->getMinBound();
    if (is_loc && (loc_or_dir.array() <= max_bound.array()).all() &&
        (loc_or_dir.array() >= min_bound.array()).all()) {
      dbg("INTERNAL POINT LIGHTS NOT SUPPORTED");
      abort();
    }
    const auto center = ((min_bound + max_bound) / 2).eval();
    const auto dims = (max_bound - min_bound).eval();
    const auto dir = is_loc ? (center - loc_or_dir).eval() : loc_or_dir;
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

    std::vector<ProjectedTriangle> triangles;
    Eigen::Affine3f bounding_transform = Eigen::Translation3f(center) *
                                         Eigen::Scaling(dims) *
                                         Eigen::Affine3f::Identity();
    TriangleProjector projector(
        DirectionPlane(loc_or_dir, is_loc, projection_value, axis));
    project_shape(scene::ShapeData(bounding_transform, scene::Material(),
                                   scene::Shape::Cube),
                  projector, triangles);
    Eigen::Array2f projected_min = Eigen::Array2f(
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Eigen::Array2f projected_max =
        Eigen::Array2f(std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());
    for (const auto &triangle : triangles) {
      for (const auto &point : triangle.points()) {
        projected_min = projected_min.cwiseMin(point);
        projected_max = projected_max.cwiseMax(point);
      }
    }

    TraversalGrid grid(projector, shapes.data(), shapes.size(), projected_min,
                       projected_max, num_division_light_x,
                       num_division_light_y);

    grid.copy_into(traversals_cpu_, actions_cpu_);

    Eigen::Vector2f intersected_min(std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max());
    Eigen::Vector2f intersected_max(std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::lowest());
    for (auto x : {-0.5f, 0.5f}) {
      for (auto y : {-0.5f, 0.5f}) {
        for (auto z : {-0.5f, 0.5f}) {
          Eigen::Vector3f transformed_point =
              bounding_transform * Eigen::Vector3f(x, y, z);
          auto intersection = get_intersection_point(
              is_loc ? (loc_or_dir - transformed_point).eval() : loc_or_dir,
              projection_value, transformed_point, axis);
          intersected_min = intersected_min.cwiseMin(intersection);
          intersected_max = intersected_max.cwiseMax(intersection);
        }
      }
    }

    traversal_data_cpu_[projection_index] = TraversalData(
        traversals_offset, axis, projection_value, intersected_min,
        intersected_max, num_division_light_x, num_division_light_y);

#if 0
    dbg("intersected min max");
    std::cout << intersected_min << std::endl;
    std::cout << intersected_max << std::endl;
    std::cout << min_bound << std::endl;
    std::cout << max_bound << std::endl;
    std::cout << loc_or_dir << std::endl;
    dbg(axis);
    dbg(projection_value);
    std::cout << traversal_data_cpu_[projection_index].convert_space_coords
              << std::endl;
    std::cout << bounding_transform.matrix() << std::endl;
#endif

    traversals_offset = traversals_cpu_.size();
    projection_index++;
  };

  for (unsigned light_idx = 0; light_idx < num_lights; light_idx++) {
    const auto &light = lights[light_idx];
    light.visit([&](auto &&light_data) {
      using T = std::decay_t<decltype(light_data)>;
      if constexpr (std::is_same<T, scene::DirectionalLight>::value) {
        add_projection(false, light_data.direction);
      } else {
        add_projection(true, light_data.position);
      }
    });
  }

  auto copy = [](auto start, auto end, auto start_copy) {
    if constexpr (execution_model == ExecutionModel::GPU) {
      thrust::copy(start, end, start_copy);
    } else {
      std::copy(start, end, start_copy);
    }
  };

  traversals_.resize(traversals_cpu_.size());
  copy(traversals_cpu_.data(), traversals_cpu_.data() + traversals_cpu_.size(),
       traversals_.begin());

  actions_.resize(actions_cpu_.size());
  copy(actions_cpu_.data(), actions_cpu_.data() + actions_cpu_.size(),
       actions_.begin());

  traversal_data_.resize(traversal_data_cpu_.size());
  std::copy(traversal_data_cpu_.begin(), traversal_data_cpu_.end(),
            traversal_data_.begin());

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - start_traversal_grid)
            .count());
  }

  for (unsigned depth = 0; depth < recursive_iterations_; depth++) {
    bool is_first = depth == 0;

    current_num_blocks = 0;

    for (unsigned i = 0; i < group_disables_.size(); i++) {
      if (!group_disables_[i]) {
        group_indexes_[current_num_blocks] = i;
        current_num_blocks++;
      }
    }

    const auto start_intersect = chr::high_resolution_clock::now();

    TraversalGridsRef traveral_grids_ref(to_const_span(actions_),
                                         to_const_span(traversal_data_),
                                         to_const_span(traversals_));

    /* #pragma omp parallel for */
    for (auto &data : by_type_data_gpu) {
      if constexpr (execution_model == ExecutionModel::GPU) {
        if (current_num_blocks != 0) {
          raytrace<<<current_num_blocks, general_block_size>>>(
              block_data_, data, traveral_grids_ref, to_const_span(shapes),
              Span<const scene::Light, false>(lights, num_lights),
              Span(textures, num_textures), to_span(world_space_eyes_),
              to_span(world_space_directions_), to_span(color_multipliers_),
              to_span(colors_), to_span(ignores_), to_span(disables_),
              to_span(group_disables_),
              Span<const unsigned, false>(group_indexes_.data(),
                                          group_indexes_.size()),
              is_first, use_kd_tree, use_traversals);
        }
      } else {
        raytrace_cpu(
            block_data_, data, traveral_grids_ref, to_const_span(shapes),
            Span<const scene::Light, false>(lights, num_lights),
            Span(textures, num_lights), to_span(world_space_eyes_),
            to_span(world_space_directions_), to_span(color_multipliers_),
            to_span(colors_), to_span(ignores_), to_span(disables_),
            to_span(group_disables_),
            Span<const unsigned, false>(group_indexes_.data(),
                                        group_indexes_.size()),
            is_first, use_kd_tree, use_traversals);
      }
    }

    CUDA_ERROR_CHK(cudaDeviceSynchronize());

    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(
              chr::high_resolution_clock::now() - start_intersect)
              .count());
    }
  }

  const unsigned width = block_data_.x_dim / super_sampling_rate_;
  const unsigned height = block_data_.y_dim / super_sampling_rate_;

  if constexpr (execution_model == ExecutionModel::GPU) {
    const auto start_convert = chr::high_resolution_clock::now();
    const unsigned x_block_size = 32;
    const unsigned y_block_size = 8;
    dim3 block(x_block_size, y_block_size);
    dim3 grid(num_blocks(width, x_block_size),
              num_blocks(height, y_block_size));

    floats_to_bgras<<<grid, block>>>(width, height, super_sampling_rate_,
                                     to_const_span(colors_), to_span(bgra_));

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
    floats_to_bgras_cpu(width, height, super_sampling_rate_,
                        to_const_span(colors_), Span(pixels, width * height));
  }

#if 0
  auto draw_point = [&](unsigned x, unsigned y, BGRA color) {
    for (unsigned y_draw = std::max(y, 6u) - 6; y_draw < std::min(y, height - 6);
         y_draw++) {
      for (unsigned x_draw = std::max(x, 6u) - 6;
           x_draw < std::min(x, width - 6); x_draw++) {
        pixels[x_draw + y_draw * width] = color;
      }
    }
  };

  for (const auto &triangle : output_triangles) {
    for (const auto &point : triangle.points()) {
      draw_point(((point[0] + 1) / 2) * width, ((point[1] + 1) / 2) * height,
                 BGRA(200, 200, 200, 0));
    }
  }
#endif
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace ray
