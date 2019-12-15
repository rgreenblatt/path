#pragma once

#include "lib/bgra.h"
#include "lib/unified_memory_vector.h"
#include "ray/action_grid.h"
#include "ray/best_intersection.h"
#include "ray/by_type_data.h"
#include "ray/execution_model.h"
#include "ray/kdtree.h"
#include "scene/scene.h"

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
              const Eigen::Affine3f &world_to_film,
              const Eigen::Projective3f &unhinging, bool use_kd_tree,
              bool show_times);

  RendererImpl(unsigned width, unsigned height, unsigned super_sampling_rate,
               unsigned recursive_iterations, std::unique_ptr<scene::Scene> &s);

  scene::Scene &get_scene() { return *scene_; }

  const scene::Scene &get_scene() const { return *scene_; }

private:
  template <typename T> using DataType = detail::DataType<execution_model, T>;

  struct ByTypeData {
    ManangedMemVec<detail::KDTreeNode<detail::AABB>> nodes;
    scene::Shape shape_type;

    detail::ByTypeDataRef initialize(const scene::Scene &scene,
                                     scene::ShapeData *shapes);

    ByTypeData(scene::Shape shape_type) : shape_type(shape_type) {}
  };

  unsigned effective_width_;
  unsigned effective_height_;
  unsigned super_sampling_rate_;
  unsigned pixel_size_;

  unsigned recursive_iterations_;

  std::unique_ptr<scene::Scene> scene_;

  std::array<ByTypeData, 1> by_type_data_;

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
  DataType<detail::TraversalData> traversal_data_;
  std::vector<detail::Traversal> traversals_cpu_;
  std::vector<detail::Action> actions_cpu_;
  std::vector<detail::TraversalData> traversal_data_cpu_;
};
} // namespace ray
