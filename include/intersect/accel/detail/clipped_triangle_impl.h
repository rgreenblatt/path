#pragma once

#include "intersect/accel/detail/clipped_triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/start_end.h"

namespace intersect {
namespace accel {
namespace detail {
HOST_DEVICE inline ClippedTriangle::ClippedTriangle(const Triangle &triangle)
    : triangle(triangle), bounds(triangle.bounds()) {}

HOST_DEVICE inline AABB ClippedTriangle::new_bounds(const float left_bound,
                                                    const float right_bound,
                                                    const unsigned axis) const {
  // this is probably not the fastest way to do this...
  // optimizations:
  //  - special casing common patterns
  //  - algebric simplification?
  debug_assert(bounds.min_bound[axis] <= right_bound &&
               bounds.max_bound[axis] >= left_bound);

  AABB new_bounds = bounds;
  new_bounds.min_bound[axis] = std::max(new_bounds.min_bound[axis], left_bound);
  new_bounds.max_bound[axis] =
      std::min(new_bounds.max_bound[axis], right_bound);
#if 0
  return new_bounds;
#else
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (new_bounds.min_bound[axis] == new_bounds.max_bound[axis]) {
      debug_assert(triangle.vertices[0][axis] == new_bounds.max_bound[axis] &&
                   triangle.vertices[1][axis] == new_bounds.max_bound[axis] &&
                   triangle.vertices[2][axis] == new_bounds.max_bound[axis]);

      // this is a hack to ensure we actually get triangle -> aabb collisions
      // when the triangle is axis aligned...
      new_bounds.min_bound[axis] -= 1.;
      new_bounds.max_bound[axis] += 1.;
    }
  }

  constexpr unsigned n_verts = 3;

  AABB out = AABB::empty();

  // check if triangle points lie inside AABB
  for (const auto &vertex : triangle.vertices) {
    if (new_bounds.contains(vertex)) {
      out = out.union_point(vertex);
    }
  }

  std::array<StartEnd<unsigned>, n_verts> edges_start_end = {{
      {.start = 0, .end = 1},
      {.start = 0, .end = 2},
      {.start = 1, .end = 2},
  }};

  // intersect triangle with AABB
  for (unsigned vert_idx = 0; vert_idx < n_verts; ++vert_idx) {
    auto [start, end] = edges_start_end[vert_idx];
    const auto dir = triangle.vertices[end] - triangle.vertices[start];
    for (float sign : {-1.f, 1.f}) {
      const auto dir_sign = dir * sign;

      auto inv_direction = get_inv_direction(dir_sign);
      auto intersection = new_bounds.solve_bounding_intersection(
          triangle.vertices[start], inv_direction);
      if (intersection.has_value()) {
        out = out.union_point(dir_sign * (*intersection) +
                              triangle.vertices[start]);
      }
    }
  }

  // intersect AABB with triangle
  for (unsigned projection_dir = 0; projection_dir < 3; ++projection_dir) {
    for (bool is_min_next : {false, true}) {
      for (bool is_min_next_next : {false, true}) {
        unsigned next_axis = (projection_dir + 1) % 3;
        unsigned next_next_axis = (projection_dir + 2) % 3;
        Eigen::Vector2f p = {
            (is_min_next ? new_bounds.min_bound
                         : new_bounds.max_bound)[next_axis],
            (is_min_next_next ? new_bounds.min_bound
                              : new_bounds.max_bound)[next_next_axis],
        };
        auto get_point = [&](const Eigen::Vector3f &vec) -> Eigen::Vector2f {
          return {vec[next_axis], vec[next_next_axis]};
        };

        Eigen::Vector3f dir_0_to_1 =
            triangle.vertices[1] - triangle.vertices[0];
        Eigen::Vector3f dir_0_to_2 =
            triangle.vertices[2] - triangle.vertices[0];

        auto v_0 = get_point(triangle.vertices[0]);
        auto v_1 = get_point(triangle.vertices[1]);
        auto v_2 = get_point(triangle.vertices[2]);

        // source:
        // https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
        float area =
            0.5 * (-v_1.y() * v_2.x() + v_0.y() * (-v_1.x() + v_2.x()) +
                   v_0.x() * (v_1.y() - v_2.y()) + v_1.x() * v_2.y());
        float s = 1 / (2 * area) *
                  (v_0.y() * v_2.x() - v_0.x() * v_2.y() +
                   (v_2.y() - v_0.y()) * p.x() + (v_0.x() - v_2.x()) * p.y());
        float t = 1 / (2 * area) *
                  (v_0.x() * v_1.y() - v_0.y() * v_1.x() +
                   (v_0.y() - v_1.y()) * p.x() + (v_1.x() - v_0.x()) * p.y());

        if (s >= 0 && t >= 0 && (1 - s - t) > 0) {
          out = out.union_point(s * dir_0_to_1 + t * dir_0_to_2 +
                                triangle.vertices[0]);
        }
      }
    }
  }

  return out;
#endif
}
} // namespace detail
} // namespace accel
} // namespace intersect
