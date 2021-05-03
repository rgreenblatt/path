#include "intersect/accel/detail/clipped_triangle.h"
#include "intersect/accel/detail/clipped_triangle_impl.h"

#include <gtest/gtest.h>

using namespace intersect;
using namespace intersect::accel;
using namespace intersect::accel::detail;

constexpr std::array<std::array<unsigned, 3>, 6> perms{
    {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};

void expect_aabb_eq(const AABB &expected, const AABB &actual) {
  for (unsigned axis = 0; axis < 3; ++axis) {
    EXPECT_FLOAT_EQ(actual.min_bound[axis], expected.min_bound[axis]);
    EXPECT_FLOAT_EQ(actual.max_bound[axis], expected.max_bound[axis]);
  }
}

static AABB chop_triangle_aabb(const Triangle &triangle, float left_bound,
                               float right_bound, unsigned axis) {
  ClippedTriangle clipped(triangle);
  return clipped.new_bounds(left_bound, right_bound, axis);
}

TEST(copy_triangle_aabb, basic_2d) {
  const Triangle basic = {{
      Eigen::Vector3f{0.f, 0.f, 0.f},
      Eigen::Vector3f{1.f, 2.f, 0.f},
      Eigen::Vector3f{0.f, 2.f, 0.f},
  }};

  for (std::array<unsigned, 3> perm : perms) {
    const Triangle permutated = {{
        basic.vertices[perm[0]],
        basic.vertices[perm[1]],
        basic.vertices[perm[2]],
    }};

    const AABB uncut = {
        .min_bound = {0.f, 0.f, 0.f},
        .max_bound = {1.f, 2.f, 0.f},
    };

    for (unsigned axis = 0; axis < 3; ++axis) {
      {
        const AABB actual = chop_triangle_aabb(
            permutated, -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(), axis);

        expect_aabb_eq(uncut, actual);
      }
      {
        const AABB actual = chop_triangle_aabb(permutated, 0.f, 2.f, axis);

        expect_aabb_eq(uncut, actual);
      }
    }

    {
      const AABB actual = chop_triangle_aabb(permutated, 0.25f, 1.f, 0);

      expect_aabb_eq(
          {
              .min_bound = {0.25f, 0.5f, 0.f},
              .max_bound = {1.f, 2.f, 0.f},
          },
          actual);
    }
    {
      const AABB actual = chop_triangle_aabb(permutated, 0.f, 0.75f, 0);

      expect_aabb_eq(
          {
              .min_bound = {0.f, 0.f, 0.f},
              .max_bound = {0.75f, 2.f, 0.f},
          },
          actual);
    }
    {
      const AABB actual = chop_triangle_aabb(permutated, 0.25f, 0.75f, 0);

      expect_aabb_eq(
          {
              .min_bound = {0.25f, 0.5f, 0.f},
              .max_bound = {0.75f, 2.f, 0.f},
          },
          actual);
    }

    {
      ClippedTriangle clipped(permutated);
      clipped.bounds = clipped.new_bounds(0.25f, 0.5f, 0);
      clipped.bounds = clipped.new_bounds(1.25f, 1.5f, 1);
      auto actual = clipped.bounds;

      expect_aabb_eq(
          {
              .min_bound = {0.25f, 1.25f, 0.f},
              .max_bound = {0.5f, 1.5f, 0.f},
          },
          actual);
    }
    {
      ClippedTriangle clipped(permutated);
      clipped.bounds = clipped.new_bounds(1.25f, 1.5f, 1);
      clipped.bounds = clipped.new_bounds(0.25f, 0.5f, 0);
      auto actual = clipped.bounds;

      expect_aabb_eq(
          {
              .min_bound = {0.25f, 1.25f, 0.f},
              .max_bound = {0.5f, 1.5f, 0.f},
          },
          actual);
    }
  }
}
