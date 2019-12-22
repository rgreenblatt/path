#pragma once

#include "lib/bgra.h"
#include "lib/unified_memory_vector.h"
#include "ray/traversal_grid.h"
#include "ray/best_intersection.h"
#include "ray/kdtree_nodes_ref.h"
#include "ray/execution_model.h"
#include "ray/kdtree.h"
#include "scene/scene.h"
#include "ray/block_data.h"

#include <thrust/device_vector.h>
#include <thrust/optional.h>

#include <functional>

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
} // namespace detail

template <ExecutionModel execution_model> class RendererImpl {
public:
  void render(BGRA *pixels, const scene::Transform &m_film_to_world,
              const Eigen::Projective3f &world_to_film, bool use_kd_tree,
              bool use_traversals, bool use_traversal_dists, bool show_times);

  RendererImpl(unsigned width, unsigned height, unsigned super_sampling_rate,
               unsigned recursive_iterations, std::unique_ptr<scene::Scene> &s);

  scene::Scene &get_scene() { return *scene_; }

  const scene::Scene &get_scene() const { return *scene_; }

private:
  void raytrace_pass(bool is_first, bool use_kd_tree, bool use_traversals,
                     bool use_traversal_dists, unsigned current_num_blocks,
                     Span<const scene::ShapeData, false> shapes,
                     Span<const scene::Light, false> lights,
                     Span<const scene::TextureImageRef> textures,
                     const detail::TraversalGridsRef &traversal_grids_ref);
  void fill(const Eigen::Affine3f &m_film_to_world);
  detail::TraversalGridsRef
  traversal_grids(bool show_times, const Eigen::Projective3f &world_to_film,
                  Span<const scene::ShapeData, false> shapes,
                  Span<const scene::Light, false> lights);

  template <typename T> using DataType = detail::DataType<execution_model, T>;

  const detail::BlockData block_data_;
  const unsigned super_sampling_rate_;

  unsigned recursive_iterations_;

  std::unique_ptr<scene::Scene> scene_;

  ManangedMemVec<detail::KDTreeNode<detail::AABB>> kdtree_nodes_;

  DataType<Eigen::Vector3f> world_space_eyes_;
  DataType<Eigen::Vector3f> world_space_directions_;
  DataType<unsigned> ignores_;
  DataType<scene::Color> color_multipliers_;
  DataType<uint8_t> disables_;
  DataType<scene::Color> colors_;
  DataType<BGRA> bgra_;
  ManangedMemVec<uint8_t> group_disables_;
  ManangedMemVec<unsigned> group_indexes_;
  DataType<detail::Traversal> traversals_;
  DataType<detail::Action> actions_;
  ManangedMemVec<detail::TraversalData> traversal_data_;
  ManangedMemVec<detail::TraversalGrid> traversal_grids_;
  ManangedMemVec<detail::BoundingPoints> shape_bounds_;
  DataType<detail::ShapePossibles> shape_grids_;
  DataType<int> action_starts_;
  DataType<int> action_ends_;
};
} // namespace ray
