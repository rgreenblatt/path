#include "lib/projection.h"

#include <gtest/gtest.h>

TEST(Projection, find_rotate_vector_to_vector) {
  auto check = [](const Eigen::Vector3f &inp, const Eigen::Vector3f &target) {
    auto transform = find_rotate_vector_to_vector(inp, target);

    auto transformed = (transform * inp.normalized()).eval();

    auto normalized_target = target.normalized().eval();

    EXPECT_NEAR(transformed.x(), normalized_target.x(), 1e-6f);
    EXPECT_NEAR(transformed.y(), normalized_target.y(), 1e-6f);
    EXPECT_NEAR(transformed.z(), normalized_target.z(), 1e-6f);
  };

  check({0, 0, 1}, {0, 0, 1});
  check({0, 0, 1}, {0, 0, -1});
  check({0, 1, 1}, {0, 0, 1});
  check({1, 0, 1}, {0, 0, 1});
  check({1, 1, 1}, {0, 0, 1});
  check({0.382, 0.38, 1}, {0, 0, 1});
  check({0.7, 0.8, 1}, {0, 0, -1});
  check({0.7, 0.8, 0.1}, {0, 0, -1});
  check({0.7, 0.8, -0.1}, {0, 0, -1});
}
