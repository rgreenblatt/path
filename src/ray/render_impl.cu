#include "lib/unified_memory_vector.h"
#include "ray/intersect.cuh"
#include "ray/kdtree.h"
#include "ray/lighting.cuh"
#include "ray/render_impl.h"

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
          return ByTypeData(pixel_size_, shape_type);
        };

        return std::array{
            get_by_type(scene::Shape::Sphere),
            get_by_type(scene::Shape::Cylinder),
            get_by_type(scene::Shape::Cube),
            get_by_type(scene::Shape::Cone),
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
  const unsigned num_shape = scene.getNumShape(shape_type);
  const unsigned start_shape = scene.getStartShape(shape_type);

  auto kdtree = construct_kd_tree(shapes + start_shape, num_shape);
  nodes.resize(kdtree.size());
  std::copy(kdtree.begin(), kdtree.end(), nodes.begin());
  return ByTypeDataRef(to_ptr(intersections), nodes.data(), nodes.size(),
                       shape_type, start_shape, num_shape);
}

template <ExecutionModel execution_model>
template <typename... T>
void RendererImpl<execution_model>::minimize_intersections(
    unsigned size, bool is_first, const DataType<uint8_t> &disables,
    T &... values) {
  if constexpr (execution_model == ExecutionModel::GPU) {
    const unsigned minimize_block_size = 256;
    minimize_all_intersections<<<num_blocks(size, minimize_block_size),
                                 minimize_block_size>>>(
        size, is_first, to_ptr(disables), to_ptr(values.intersections)...);

    CUDA_ERROR_CHK(cudaDeviceSynchronize());
  } else {
    minimize_all_intersections_cpu(size, is_first, disables.data(),
                                   values.intersections.data()...);
  }
}

template <ExecutionModel execution_model>
template <typename... T>
void RendererImpl<execution_model>::minimize_intersections(
    bool is_first, const DataType<uint8_t> &disables, ByTypeData &first,
    const T &... rest) {
  minimize_intersections(static_cast<unsigned>(first.intersections.size()),
                         is_first, disables, first, rest...);
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

  const unsigned width_block_size = 8;
  const unsigned height_block_size = 8;

  const dim3 grid(num_blocks(effective_width_, width_block_size),
                  num_blocks(effective_height_, height_block_size), 1);
  const dim3 block(width_block_size, height_block_size, 1);

  const Eigen::Vector3f world_space_eye = m_film_to_world.translation();

  const scene::Color initial_multiplier = scene::Color::Ones();

  const scene::Color initial_color = scene::Color::Zero();

  const auto start_fill = chr::high_resolution_clock::now();

  // could be made async until...
  if constexpr (execution_model == ExecutionModel::GPU) {
    const unsigned fill_block_size = 256;
    fill<<<num_blocks(num_pixels, fill_block_size), fill_block_size>>>(
        to_ptr(world_space_eyes_), num_pixels, world_space_eye);
    fill<<<num_blocks(num_pixels, fill_block_size), fill_block_size>>>(
        to_ptr(color_multipliers_), num_pixels, initial_multiplier);
    fill<<<num_blocks(num_pixels, fill_block_size), fill_block_size>>>(
        to_ptr(colors_), num_pixels, initial_color);

    initial_world_space_directions<<<grid, block>>>(
        effective_width_, effective_height_, world_space_eye, m_film_to_world,
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
  for (unsigned i = 0; i < scene::shapes_size; i++) {
    by_type_data_gpu[i] = by_type_data_[i].initialize(scene, shapes.data());
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - start_kdtree)
            .count());
  }

  for (unsigned depth = 0; depth < recursive_iterations_; depth++) {
    bool is_first = depth == 0;

    const auto start_intersect = chr::high_resolution_clock::now();

    for (auto &data : by_type_data_gpu) {
      auto solve = [&]<scene::Shape shape_type>() {
        if constexpr (execution_model == ExecutionModel::GPU) {
          solve_intersections<shape_type><<<grid, block>>>(
              effective_width_, effective_height_, data, shapes.data(),
              to_ptr(world_space_eyes_), to_ptr(world_space_directions_),
              to_ptr(ignores_), to_ptr(disables_), is_first, use_kd_tree);
        } else {
          solve_intersections_cpu<shape_type>(
              effective_width_, effective_height_, data, shapes.data(),
              world_space_eyes_.data(), world_space_directions_.data(),
              ignores_.data(), disables_.data(), is_first, use_kd_tree);
        }
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
    }

    CUDA_ERROR_CHK(cudaDeviceSynchronize());

    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(
              chr::high_resolution_clock::now() - start_intersect)
              .count());
    }

    const auto start_minimize = chr::high_resolution_clock::now();

    // fuse kernel???
    minimize_intersections(is_first, disables_, by_type_data_[0],
                           by_type_data_[1], by_type_data_[2],
                           by_type_data_[3]);

    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(
              chr::high_resolution_clock::now() - start_minimize)
              .count());
    }

    auto &best_intersections = by_type_data_[0].intersections;

    const auto start_color = chr::high_resolution_clock::now();
    if constexpr (execution_model == ExecutionModel::GPU) {
      compute_colors<<<grid, block>>>(
          effective_width_, effective_height_, by_type_data_gpu,
          to_ptr(world_space_eyes_), to_ptr(world_space_directions_),
          to_ptr(ignores_), to_ptr(color_multipliers_), to_ptr(disables_),
          to_ptr(best_intersections), shapes.data(), lights, num_lights,
          textures, to_ptr(colors_), use_kd_tree, is_first);

      CUDA_ERROR_CHK(cudaDeviceSynchronize());
    } else {
      compute_colors_cpu(
          effective_width_, effective_height_, by_type_data_gpu,
          world_space_eyes_.data(), world_space_directions_.data(),
          ignores_.data(), color_multipliers_.data(), disables_.data(),
          best_intersections.data(), shapes.data(), lights, num_lights,
          textures, colors_.data(), use_kd_tree, is_first);
    }
    const auto end_color = chr::high_resolution_clock::now();
    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(end_color - start_color)
              .count());
    }
  }

  const unsigned width = effective_width_ / super_sampling_rate_;
  const unsigned height = effective_height_ / super_sampling_rate_;

  if constexpr (execution_model == ExecutionModel::GPU) {
    const dim3 to_bgra_grid(num_blocks(width, width_block_size),
                            num_blocks(height, height_block_size), 1);
    const dim3 to_bgra_block(width_block_size, height_block_size, 1);

    const auto start_convert = chr::high_resolution_clock::now();
    floats_to_bgras<<<to_bgra_grid, to_bgra_block>>>(
        width, height, super_sampling_rate_, to_ptr(colors_), to_ptr(bgra_));

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
