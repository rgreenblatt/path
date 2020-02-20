#include "material/utils.h"

#include <gtest/gtest.h>

using namespace material;

TEST(Material, refract_by_normal) {
  auto check = [](float ior, Eigen::Vector3f vec,
                  Eigen::Vector3f normal,
                  Eigen::Vector3f expected) {
    vec.normalize();
    normal.normalize();
    expected.normalize();
    auto actual = refract_by_normal(ior, vec, normal);
    float epsilon = 1e-5f;
    EXPECT_NEAR(actual.x(), expected.x(), epsilon);
    EXPECT_NEAR(actual.y(), expected.y(), epsilon);
    EXPECT_NEAR(actual.z(), expected.z(), epsilon);
  };

  check(1.5, {0, 0.5, 0.5}, {0, 0, 1}, {0, 0.534522, -1});
  check(1.5, {0, 0.5, 0.5}, {0, 0, -1},
        reflect_over_normal(Eigen::Vector3f(0, 0.5, 0.5).normalized(),
                            {0, 0, 1}));
  check(1.1, {0, 0.5, 0.5}, {0, 0, -1}, {0, 1.2376, -1});
}
