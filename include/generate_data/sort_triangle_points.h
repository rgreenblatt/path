#pragma once

#include "generate_data/triangle.h"

#include <algorithm>

namespace generate_data {
void sort_triangle_points(intersect::TriangleGen<double> &tri) {
  std::sort(tri.vertices.begin(), tri.vertices.end(),
            [&](const Eigen::Vector3d &l, const Eigen::Vector3d &r) {
              if (l.z() != r.z()) {
                return l.z() < r.z();
              } else if (l.y() != r.y()) {
                return l.y() < r.y();
              } else {
                return l.x() < r.x();
              }
            });
}
} // namespace generate_data
