#pragma once

#include "BGRA.h"
#include "lib/unified_memory_vector.h"
#include "ray/best_intersection.h"
#include "scene/scene.h"

#include <functional>
#include <optional>

namespace ray {
namespace detail {
struct ByTypeData {
  ManangedMemVec<std::optional<detail::BestIntersection>> intersections;
  scene::Shape shape_type;

  ByTypeData(unsigned width, unsigned height, scene::Shape shape_type)
      : intersections(width * height), shape_type(shape_type) {}
};

}; // namespace detail

class Renderer {
public:
  void render(const scene::Scene &scene, BGRA *pixels,
              const scene::Transform &m_film_to_world);

  Renderer(unsigned width, unsigned height)
      : width_(width), height_(height), by_type_data_(std::invoke([&] {
          auto get_by_type = [&](scene::Shape shape_type) {
            return detail::ByTypeData(width, height, shape_type);
          };

          return std::array{get_by_type(scene::Shape::Cube),
                            get_by_type(scene::Shape::Sphere),
                            get_by_type(scene::Shape::Cylinder)};
        })) {}

protected:
  unsigned width_;
  unsigned height_;

  std::array<detail::ByTypeData, scene::shapes_size> by_type_data_;
};

} // namespace ray
