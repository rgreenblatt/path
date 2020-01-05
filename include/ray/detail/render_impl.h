#pragma once

#include "lib/bgra.h"
#include "lib/cuda/unified_memory_vector.h"
#include "ray/detail/block_data.h"
#include "ray/execution_model.h"
#include "ray/detail/accel/kdtree/kdtree.h"
#include "ray/detail/accel/dir_tree/dir_tree.h"
#include "scene/scene.h"

#include <thrust/device_vector.h>

#include <chrono>
#include <dbg.h>

namespace ray {
namespace detail {
template <ExecutionModel execution_model, typename T> struct get_vector_type;

template <typename T> struct get_vector_type<ExecutionModel::CPU, T> {
  using type = std::vector<T>;
};

template <typename T> struct get_vector_type<ExecutionModel::GPU, T> {
  using type = thrust::device_vector<T>;
};

template <ExecutionModel execution_model, typename T>
using DataType = typename get_vector_type<execution_model, T>::type;

template <ExecutionModel execution_model> class RendererImpl {
public:
  void render(BGRA *pixels, const Eigen::Affine3f &m_film_to_world,
              const Eigen::Projective3f &world_to_film, bool use_kd_tree,
              bool use_traversals, bool use_traversal_dists, bool show_times);

  RendererImpl(unsigned x_dim, unsigned y_dim, unsigned super_sampling_rate,
               unsigned recursive_iterations, std::unique_ptr<scene::Scene> &s);

  scene::Scene &get_scene() { return *scene_; }

  const scene::Scene &get_scene() const { return *scene_; }

private:
  template <bool is_first, typename Accel>
  void raytrace_pass(const Accel &accel, unsigned current_num_blocks,
                     Span<const scene::ShapeData, false> shapes,
                     Span<const scene::Light, false> lights,
                     Span<const scene::TextureImageRef> textures);

  std::chrono::high_resolution_clock::time_point current_time() {
    return std::chrono::high_resolution_clock::now();
  }

  double to_secs(std::chrono::high_resolution_clock::time_point start,
                 std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                     start)
        .count();
  }
  void fill(const scene::Color &initial_multiplier,
            const scene::Color &initial_color,
            const Eigen::Affine3f &m_film_to_world);

#if 0
  void initial_world_space_directions(const Eigen::Affine3f &m_film_to_world);
#endif

  void float_to_bgra(BGRA *pixels, Span<const scene::Color> colors);

#if 0
  TraversalGridsRef
  traversal_grids(bool show_times, const Eigen::Projective3f &world_to_film,
                  Span<const scene::ShapeData, false> shapes,
                  Span<const scene::Light, false> lights);
#endif

  template <typename T> using DataType = DataType<execution_model, T>;

  const BlockData block_data_;
  unsigned real_x_dim_;
  unsigned real_y_dim_;
  const unsigned super_sampling_rate_;

  unsigned recursive_iterations_;

  bool show_times_;

  std::unique_ptr<scene::Scene> scene_;

  ManangedMemVec<accel::kdtree::KDTreeNode<accel::AABB>> kdtree_nodes_;
  ManangedMemVec<accel::kdtree::KDTreeNode<accel::AABB>> sort_nodes_;

  DataType<Eigen::Vector3f> world_space_eyes_;
  DataType<Eigen::Vector3f> world_space_directions_;
  DataType<unsigned> ignores_;
  DataType<scene::Color> color_multipliers_;
  DataType<uint8_t> disables_;
  DataType<scene::Color> colors_;
  DataType<BGRA> bgra_;
  ManangedMemVec<uint8_t> group_disables_;
  ManangedMemVec<unsigned> group_indexes_;
#if 0
  DataType<Traversal> traversals_;
  DataType<Action> min_sorted_actions_;
  DataType<Action> max_sorted_actions_;
#endif
  DataType<unsigned> segments_;
#if 0
  ManangedMemVec<TraversalData> traversal_data_;
  ManangedMemVec<TraversalGrid> traversal_grids_;
  ManangedMemVec<BoundingPoints> shape_bounds_;
  DataType<ShapePossibles> shape_grids_;
#endif
  DataType<int> action_starts_;
  DataType<int> action_ends_;
};
} // namespace detail
} // namespace ray
