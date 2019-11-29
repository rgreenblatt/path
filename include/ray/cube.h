#pragma once

#include "ray/ray_utils.h"

namespace ray {
namespace detail {
template <bool normal_and_uv>
HOST_DEVICE IntersectionOp<normal_and_uv>
solve_cube(const Eigen::Vector3f &point, const Eigen::Vector3f &direction,
           bool texture_map) {
  const float width = 0.5f;
  // p_axis + t d_axis = (+/-)1/2
  // t = ((+/-)1/2 - p_axis) / d_axis
  const auto get_axis = [&](int axis, const std::array<int, 2> &other_indexes,
                            bool flip, int flip_at, bool negate_first) {
    const auto get_side = [&](bool first) {
      const float v =
          ((first ? width : -width) - point[axis]) / direction[axis];
      if (v < 0) {
        return IntersectionOp<normal_and_uv>(thrust::nullopt);
      }
      const auto get_intersection = [&](const size_t number) {
        const auto &index = other_indexes[number];
        return v * direction[index] + point[index];
      };
      const Eigen::Array2f intersection = Eigen::Array2f(
          get_intersection(0),
          negate_first ? -get_intersection(1) : get_intersection(1));
      const auto is_valid = [&](int index) {
        return std::abs(intersection[index]) < width + epsilon;
      };
      return make_optional(
          is_valid(0) && is_valid(1), invoke([&] {
            if constexpr (normal_and_uv) {
              Eigen::Vector3f normal(0, 0, 0);
              normal[axis] = first ? 1.0f : -1.0f;

              return IntersectionNormalUV(
                  v, normal,
                  texture_map
                      ? uv_square_face(intersection,
                                       make_optional(flip != first, flip_at))
                      : UVPosition(0));
            } else {
              return v;
            }
          }));
    };
    const auto a = get_side(true);
    const auto b = get_side(false);

    return optional_min(a, b);
  };

  return optional_min(get_axis(0, {{2, 1}}, false, 0, true),
                      get_axis(1, {{0, 2}}, true, 1, false),
                      get_axis(2, {{0, 1}}, true, 0, true));
}
} // namespace detail
} // namespace ray
