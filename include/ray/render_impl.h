#pragma once

#include "lib/bgra.h"
#include "lib/unified_memory_vector.h"
#include "ray/best_intersection.h"
#include "ray/by_type_data.h"
#include "ray/kdtree.h"
#include "ray/execution_model.h"
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
  void render(const scene::Scene &scene, BGRA *pixels,
              const scene::Transform &m_film_to_world, bool use_kd_tree,
              bool show_times);

  RendererImpl(unsigned width, unsigned height, unsigned super_sampling_rate,
               unsigned recursive_iterations);

private:
  template <typename T> using DataType = detail::DataType<execution_model, T>;

  struct ByTypeData {
    DataType<thrust::optional<detail::BestIntersectionNormalUV>> intersections;
    ManangedMemVec<detail::KDTreeNode> nodes;
    scene::Shape shape_type;

    detail::ByTypeDataRef initialize(const scene::Scene &scene,
                                     scene::ShapeData *shapes);

    ByTypeData(unsigned pixel_size, scene::Shape shape_type)
        : intersections(pixel_size), shape_type(shape_type) {}
  };

  unsigned effective_width_;
  unsigned effective_height_;
  unsigned super_sampling_rate_;
  unsigned pixel_size_;

  unsigned recursive_iterations_;

  std::array<ByTypeData, scene::shapes_size> by_type_data_;

  DataType<Eigen::Vector3f> world_space_eyes_;
  DataType<Eigen::Vector3f> world_space_directions_;
  DataType<unsigned> ignores_;
  DataType<scene::Color> color_multipliers_;
  DataType<uint8_t> disables_;
  DataType<scene::Color> colors_;
  DataType<BGRA> bgra_;
  ManangedMemVec<uint8_t> group_disables_;
  ManangedMemVec<unsigned> group_indexes_;
};
} // namespace ray
