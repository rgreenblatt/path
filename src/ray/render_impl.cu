#include "ray/floats_to_bgras.cuh"
#include "ray/render_impl.h"
#include "ray/render_impl_utils.h"
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
      world_space_eyes_(block_data_.totalSize()),
      world_space_directions_(block_data_.totalSize()),
      ignores_(block_data_.totalSize()),
      color_multipliers_(block_data_.totalSize()),
      disables_(block_data_.totalSize()), colors_(block_data_.totalSize()),
      bgra_(width * height) {}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(
    BGRA *pixels, const scene::Transform &m_film_to_world,
    const Eigen::Projective3f &world_to_film, bool use_kd_tree,
    bool use_traversals, bool use_traversal_dists, bool show_times) {
  namespace chr = std::chrono;

  const auto lights = scene_->getLights();
  const unsigned num_lights = scene_->getNumLights();
  const auto textures = scene_->getTextures();
  const unsigned num_textures = scene_->getNumTextures();

  const unsigned general_num_blocks = block_data_.generalNumBlocks();

  group_disables_.resize(general_num_blocks);
  group_indexes_.resize(general_num_blocks);

  unsigned current_num_blocks = general_num_blocks;

  const auto start_fill = chr::high_resolution_clock::now();

  // could be made async until...
  fill(m_film_to_world);

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - start_fill)
            .count());
  }

  const unsigned num_shapes = scene_->getNumShapes();
  ManangedMemVec<scene::ShapeData> moved_shapes_(num_shapes);

  {
    auto start_shape = scene_->getShapes();
    std::copy(start_shape, start_shape + num_shapes, moved_shapes_.begin());
  }

  if (use_kd_tree && !use_traversals) {
    const auto start_kdtree = chr::high_resolution_clock::now();

    auto kdtree = construct_kd_tree(moved_shapes_.data(), num_shapes, 25, 3);
    kdtree_nodes_.resize(kdtree.size());
    std::copy(kdtree.begin(), kdtree.end(), kdtree_nodes_.begin());

    if (show_times) {
      dbg(chr::duration_cast<chr::duration<double>>(
              chr::high_resolution_clock::now() - start_kdtree)
              .count());
    }
  }

  Span<const scene::Light, false> lights_span(lights, num_lights);

  TraversalGridsRef traversal_grids_ref;

  if (use_traversals) {
    traversal_grids_ref =
        traversal_grids(show_times, world_to_film,
                        Span<const scene::ShapeData, false>(
                            moved_shapes_.data(), moved_shapes_.size()),
                        lights_span);
  } else {
    for (auto &disable : group_disables_) {
      disable = false;
    }
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

    raytrace_pass(is_first, use_kd_tree, use_traversals, use_traversal_dists,
                  current_num_blocks,
                  Span<const scene::ShapeData, false>(moved_shapes_.data(),
                                                      moved_shapes_.size()),
                  Span<const scene::Light, false>(lights, num_lights),
                  Span(textures, num_textures), traversal_grids_ref);

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
    dim3 grid(num_blocks(width, x_block_size),
              num_blocks(height, y_block_size));
    dim3 block(x_block_size, y_block_size);

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
