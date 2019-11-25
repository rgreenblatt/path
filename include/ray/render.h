#pragma once

#include "lib/bgra.h"
#include "lib/unified_memory_vector.h"
#include "ray/best_intersection.h"
#include "ray/kdtree.h"
#include "scene/scene.h"

#include <thrust/optional.h>

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
  using type = ManangedMemVec<T>;
};

template <ExecutionModel execution_model, typename T>
using DataType = typename get_vector_type<execution_model, T>::type;
}

template <ExecutionModel execution_model> class Renderer {
public:
  void render(const scene::Scene &scene, BGRA *pixels,
              const scene::Transform &m_film_to_world);

  Renderer(unsigned width, unsigned height, unsigned recursive_iterations,
           unsigned x_special, unsigned y_special);

private:
  template <typename... T>
  void minimize_intersections(unsigned size, T &... values);

  template <typename T> using DataType = detail::DataType<execution_model, T>;

  struct ByTypeData {
    DataType<thrust::optional<detail::BestIntersectionNormalUV>> intersections;
    DataType<detail::KDTreeNode> nodes;
    scene::Shape shape_type;

    ByTypeData(unsigned width, unsigned height, scene::Shape shape_type)
        : intersections(width * height), shape_type(shape_type) {}
  };

  template <typename... T>
  void minimize_intersections(ByTypeData &first, const T &... rest);

  unsigned width_;
  unsigned height_;
  
  unsigned x_special_;
  unsigned y_special_;

  unsigned recursive_iterations_;

  std::array<ByTypeData, scene::shapes_size> by_type_data_;

  DataType<Eigen::Vector3f> world_space_eyes_;
  DataType<Eigen::Vector3f> world_space_directions_;
  DataType<unsigned> ignores_;
  DataType<scene::Color> color_multipliers_;
  DataType<uint8_t> disables_;
  DataType<scene::Color> colors_;
  DataType<BGRA> bgra_;
  /* DataType<uint8_t> light_shadowed_; */
};
} // namespace ray
