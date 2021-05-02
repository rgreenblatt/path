#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/triangle.h"
#include "lib/cuda/utils.h"
#include "lib/start_end.h"

namespace intersect {
namespace accel {
namespace detail {
HOST_DEVICE inline AABB chop_triangle_aabb(const Triangle &triangle,
                                           const float left_bound,
                                           const float right_bound,
                                           const unsigned axis) {

  std::array<float, 2> split_points = {left_bound, right_bound};

  constexpr unsigned n_verts = 3;

  std::array<std::array<bool, n_verts>, split_points.size()> outside_split;

  for (unsigned vert_idx = 0; vert_idx < n_verts; ++vert_idx) {
    outside_split[0][vert_idx] =
        triangle.vertices[vert_idx][axis] < split_points[0];
  }
  for (unsigned vert_idx = 0; vert_idx < n_verts; ++vert_idx) {
    outside_split[1][vert_idx] =
        triangle.vertices[vert_idx][axis] > split_points[1];
  }

  AABB out = AABB::empty();

  for (unsigned vert_idx = 0; vert_idx < n_verts; ++vert_idx) {
    if (!outside_split[0][vert_idx] && !outside_split[1][vert_idx]) {
      // vertex is inside
      out = out.union_point(triangle.vertices[vert_idx]);
    }
  }

  std::array<StartEnd<unsigned>, n_verts> edges_start_end = {{
      {.start = 0, .end = 1},
      {.start = 0, .end = 2},
      {.start = 1, .end = 2},
  }};

  std::array<Eigen::Vector3f, n_verts> edge_dirs;
  std::transform(edges_start_end.begin(), edges_start_end.end(),
                 edge_dirs.begin(), [&](StartEnd<unsigned> start_end) {
                   return triangle.vertices[start_end.end] -
                          triangle.vertices[start_end.start];
                 });

  for (unsigned left_right : {0, 1}) {
    for (unsigned vert_idx = 0; vert_idx < n_verts; ++vert_idx) {
      auto [start, end] = edges_start_end[vert_idx];
      if (outside_split[left_right][start] != outside_split[left_right][end]) {
        float dist = split_points[left_right] - triangle.vertices[start][axis];
        float prop = std::abs(dist / edge_dirs[vert_idx][axis]);
        out = out.union_point(prop * edge_dirs[vert_idx] +
                              triangle.vertices[start]);
      }
    }
  }

  return out;
}
} // namespace detail
} // namespace accel
} // namespace intersect
