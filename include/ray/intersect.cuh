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
solve_intersections(unsigned width, unsigned height, unsigned num_shapes,
                    const scene::ShapeData *shapes,
                    Eigen::Vector4f *world_space_directions,
                    std::optional<BestIntersection> *best_intersections) {
  unsigned x_dim = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y_dim = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned shape_idx = blockIdx.z * blockDim.z + threadIdx.z;

  // TODO
  shapes[shape_idx];
}
} // namespace detail
} // namespace ray
