#include "lib/unified_memory_vector.h"
#include "ray/intersect.cuh"
#include "ray/lighting.cuh"
#include "ray/render.h"
#include "ray/kdtree.h"

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/combine.hpp>
#include <thrust/fill.h>

#include <dbg.h>
#include <chrono>

namespace ray {
using namespace detail;

unsigned num_blocks(unsigned size, unsigned block_size) {
  return (size + block_size - 1) / block_size;
};

template <ExecutionModel execution_model>
Renderer<execution_model>::Renderer(unsigned width, unsigned height,
                                    unsigned recursive_iterations,
                                    unsigned x_special, unsigned y_special)
    : width_(width), height_(height), x_special_(x_special),
      y_special_(y_special), recursive_iterations_(recursive_iterations),
      by_type_data_(invoke([&] {
        auto get_by_type = [&](scene::Shape shape_type) {
          return ByTypeData(width, height, shape_type);
        };

        return std::array{
            get_by_type(scene::Shape::Sphere),
            get_by_type(scene::Shape::Cylinder),
            get_by_type(scene::Shape::Cube),
            get_by_type(scene::Shape::Cone),
        };
      })),
      world_space_eyes_(width * height),
      world_space_directions_(width * height), ignores_(width * height),
      color_multipliers_(width * height), disables_(width * height),
      colors_(width * height), bgra_(width * height) {}

template <ExecutionModel execution_model>
template <typename... T>
void Renderer<execution_model>::minimize_intersections(unsigned size,
                                                       T &... values) {
  if constexpr (execution_model == ExecutionModel::GPU) {
    const unsigned minimize_block_size = 256;
    minimize_all_intersections<<<num_blocks(size, minimize_block_size),
                                 minimize_block_size>>>(
        size, values.intersections.data()...);

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
  } else {
    minimize_all_intersections_cpu(size, values.intersections.data()...);
  }
}

template <ExecutionModel execution_model>
template <typename... T>
void Renderer<execution_model>::minimize_intersections(ByTypeData &first,
                                                       const T &... rest) {
  minimize_intersections(static_cast<unsigned>(first.intersections.size()),
                         first, rest...);
}

template <ExecutionModel execution_model>
void Renderer<execution_model>::render(
    const scene::Scene &scene, BGRA *pixels,
    const scene::Transform &m_film_to_world) {
  const auto lights = scene.get_lights();
  const unsigned num_lights = scene.get_num_lights();
  const unsigned num_pixels = width_ * height_;

  /* light_shadowed_.resize(num_lights * num_pixels); */

  // TODO
  /* unsigned width_block_size = 4; */
  /* unsigned height_block_size = 4; */
  /* unsigned shapes_block_size = 64; */

  const unsigned width_block_size = 32;
  const unsigned height_block_size = 32;

  const dim3 grid(num_blocks(width_, width_block_size),
                  num_blocks(height_, height_block_size), 1);
  const dim3 block(width_block_size, height_block_size, 1);

  const Eigen::Vector3f world_space_eye = m_film_to_world.translation();

  const scene::Color initial_multiplier = scene::Color::Ones();

  const scene::Color initial_color = scene::Color::Zero();

  // could be made async until...
  if constexpr (execution_model == ExecutionModel::GPU) {
    const unsigned fill_block_size = 256;
    fill<<<num_blocks(num_pixels, fill_block_size), fill_block_size>>>(
        world_space_eyes_.data(), num_pixels, world_space_eye);
    fill<<<num_blocks(num_pixels, fill_block_size), fill_block_size>>>(
        color_multipliers_.data(), num_pixels, initial_multiplier);
    fill<<<num_blocks(num_pixels, fill_block_size), fill_block_size>>>(
        colors_.data(), num_pixels, initial_color);

    initial_world_space_directions<<<grid, block>>>(
        width_, height_, world_space_eye, m_film_to_world,
        world_space_directions_.data());

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
  } else {
    std::fill(world_space_eyes_.begin(), world_space_eyes_.end(),
              world_space_eye);
    std::fill(color_multipliers_.begin(), color_multipliers_.end(),
              initial_multiplier);
    std::fill(colors_.begin(), colors_.end(), initial_color);

    initial_world_space_directions_cpu(width_, height_, world_space_eye,
                                       m_film_to_world,
                                       world_space_directions_.data());
  }
    
  auto start_shape = scene.get_shapes();
  const unsigned num_shapes = scene.num_shapes();
  DataType<scene::ShapeData> shapes(num_shapes);

  std::copy(start_shape, start_shape + num_shapes, shapes.begin());

/* #pragma omp parallel for */
  for (auto &data : by_type_data_) {
    const unsigned num_shape = scene.get_num_shape(data.shape_type);
    const unsigned start_shape = scene.get_start_shape(data.shape_type);

    auto kdtree =
        construct_kd_tree(shapes.data() + start_shape, num_shape);
    data.nodes.resize(kdtree.size());
    std::copy(kdtree.begin(), kdtree.end(), data.nodes.begin());
  }

  for (unsigned depth = 0; depth < recursive_iterations_; depth++) {
    bool is_first = depth == 0;

    const auto start_intersect = std::chrono::high_resolution_clock::now();
/* #pragma omp parallel for */
    for (auto &data : by_type_data_) {
      const unsigned start_shape = scene.get_start_shape(data.shape_type);
      if constexpr (execution_model == ExecutionModel::GPU) {
        solve_intersections<<<grid, block>>>(
            width_, height_, start_shape, shapes.data(),
            world_space_eyes_.data(), world_space_directions_.data(),
            data.nodes.data(), data.nodes.size(), ignores_.data(),
            disables_.data(), data.intersections.data(), data.shape_type,
            is_first);
      } else {
        solve_intersections_cpu(
            width_, height_, start_shape, shapes.data(),
            world_space_eyes_.data(), world_space_directions_.data(),
            data.nodes.data(), data.nodes.size(), ignores_.data(),
            disables_.data(), data.intersections.data(), data.shape_type,
            is_first);
      }
    }

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
    
    dbg(std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - start_intersect).count());

    // fuse kernel???
    minimize_intersections(by_type_data_[0], by_type_data_[1], by_type_data_[2],
                           by_type_data_[3]);

    auto &best_intersections = by_type_data_[0].intersections;
    
#if 0
    for (auto &data : by_type_data_) {
      const unsigned num_shape = scene.get_num_shape(data.shape_type);
      const unsigned start_shape = scene.get_start_shape(data.shape_type);

      if constexpr (execution_model == ExecutionModel::GPU) {
        check_intersections<<<grid, block>>>(
            width_, height_, num_shape, start_shape, shapes,
            world_space_eyes_.data(), world_space_directions_.data(),
            ignores_.data(), disables_.data(), data.intersections.data(),
            data.shape_type, is_first);
      } else {
        solve_intersections_cpu(width_, height_, num_shape, start_shape, shapes,
                                world_space_eyes_.data(),
                                world_space_directions_.data(), ignores_.data(),
                                disables_.data(), data.intersections.data(),
                                data.shape_type, is_first);
      }
    }
#endif

    const auto start_color = std::chrono::high_resolution_clock::now();
    if constexpr (execution_model == ExecutionModel::GPU) {
      // TODO block etc...
      compute_colors<<<grid, block>>>(
          width_, height_, world_space_eyes_.data(),
          world_space_directions_.data(), ignores_.data(),
          color_multipliers_.data(), disables_.data(),
          best_intersections.data(), shapes.data(), lights, num_lights,
          colors_.data(), is_first, x_special_, y_special_);

      CUDA_ERROR_CHK(cudaDeviceSynchronize());
    } else {
      compute_colors_cpu(width_, height_, world_space_eyes_.data(),
                         world_space_directions_.data(), ignores_.data(),
                         color_multipliers_.data(), disables_.data(),
                         best_intersections.data(), shapes.data(), lights,
                         num_lights, colors_.data(), is_first, x_special_,
                         y_special_);
    }
    dbg(std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - start_color)
            .count());
  }

  if constexpr (execution_model == ExecutionModel::GPU) {
    floats_to_bgras<<<grid, block>>>(width_, height_, colors_.data(),
                                     bgra_.data(), x_special_, y_special_);

    CUDA_ERROR_CHK(cudaDeviceSynchronize());

    std::copy(bgra_.begin(), bgra_.end(), pixels);
  } else {
    floats_to_bgras_cpu(width_, height_, colors_.data(), pixels, x_special_, y_special_);
  }
}

template class Renderer<ExecutionModel::CPU>;
template class Renderer<ExecutionModel::GPU>;
} // namespace ray
