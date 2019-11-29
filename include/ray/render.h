#pragma once

#include "lib/bgra.h"
#include "lib/unified_memory_vector.h"
#include "ray/best_intersection.h"
#include "ray/kdtree.h"
#include "ray/by_type_data.h"
#include "scene/scene.h"

#include <thrust/optional.h>
#include <thrust/device_vector.h>

#include <functional>

namespace ray {
enum class ExecutionModel {
  CPU,
  GPU,
};

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

template <ExecutionModel execution_model> class Renderer {
public:
  void render(const scene::Scene &scene, BGRA *pixels,
              const scene::Transform &m_film_to_world, bool use_kd_tree);

  Renderer(unsigned width, unsigned height, unsigned super_sampling_rate,
           unsigned recursive_iterations);

private:
  template <typename T> using DataType = detail::DataType<execution_model, T>;

  struct ByTypeData {
    DataType<thrust::optional<detail::BestIntersectionNormalUV>> intersections;
    ManangedMemVec<detail::KDTreeNode> nodes;
    scene::Shape shape_type;

    detail::ByTypeDataGPU initialize(const scene::Scene &scene,
                                     scene::ShapeData *shapes);

    ByTypeData(unsigned pixel_size, scene::Shape shape_type)
        : intersections(pixel_size), shape_type(shape_type) {}
  };

  template <typename... T>
  void minimize_intersections(unsigned size, bool is_first,
                              const DataType<uint8_t> &disables, T &... values);

  template <typename... T>
  void minimize_intersections(bool is_first, const DataType<uint8_t> &disables,
                              ByTypeData &first, const T &... rest);

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
};
} // namespace ray
