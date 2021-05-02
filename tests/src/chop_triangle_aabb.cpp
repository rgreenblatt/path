#include "intersect/accel/detail/chop_triangle_aabb.h"

#include <gtest/gtest.h>

using namespace intersect;
using namespace intersect::accel;
using namespace intersect::accel::detail;

constexpr std::array<std::array<unsigned, 3>, 6> perms{
    {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};

void expect_aabb_eq(const AABB &actual, const AABB &expected) {
  for (unsigned axis = 0; axis < 3; ++axis) {
    EXPECT_FLOAT_EQ(actual.min_bound[axis], expected.min_bound[axis]);
    EXPECT_FLOAT_EQ(actual.max_bound[axis], expected.max_bound[axis]);
  }
}

TEST(copy_triangle_aabb, basic) {
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
  }
}
