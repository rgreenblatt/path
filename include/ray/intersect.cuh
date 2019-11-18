#pragma once

#include "scene/shape.h"
#include "scene/shape_data.h"
#include <optional>

namespace ray {
namespace detail {

struct BestIntersection {
  float intersection;
  int shape_index;

  BestIntersection(const float intersection, const int shape_index)
      : intersection(intersection), shape_index(shape_index) {}
};

__global__ void
solve_intersections(int width, int height, int num_shapes,
                    const scene::ShapeData *shapes,
                    Eigen::Vector4f *world_space_directions,
                    std::optional<BestIntersection> *best_intersections);
} // namespace detail
} // namespace ray
