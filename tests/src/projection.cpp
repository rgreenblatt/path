#include "lib/projection.h"

#include <gtest/gtest.h>

TEST(Projection, find_rotate_vector_to_vector) {
  auto check = [](const UnitVector &inp, const UnitVector &target) {
    auto transform = find_rotate_vector_to_vector(inp, target);

    auto transformed = (transform * *inp).eval();

    EXPECT_NEAR(transformed.x(), target->x(), 1e-6f);
    EXPECT_NEAR(transformed.y(), target->y(), 1e-6f);
    EXPECT_NEAR(transformed.z(), target->z(), 1e-6f);
  };

  check(UnitVector::new_normalize({0, 0, 1}),
        UnitVector::new_normalize({0, 0, 1}));
  check(UnitVector::new_normalize({0, 0, 1}),
        UnitVector::new_normalize({0, 0, -1}));
  check(UnitVector::new_normalize({0, 1, 1}),
        UnitVector::new_normalize({0, 0, 1}));
  check(UnitVector::new_normalize({1, 0, 1}),
        UnitVector::new_normalize({0, 0, 1}));
  check(UnitVector::new_normalize({1, 1, 1}),
        UnitVector::new_normalize({0, 0, 1}));
  check(UnitVector::new_normalize({0.382, 0.38, 1}),
        UnitVector::new_normalize({0, 0, 1}));
  check(UnitVector::new_normalize({0.7, 0.8, 1}),
        UnitVector::new_normalize({0, 0, -1}));
  check(UnitVector::new_normalize({0.7, 0.8, 0.1}),
        UnitVector::new_normalize({0, 0, -1}));
  check(UnitVector::new_normalize({0.7, 0.8, -0.1}),
        UnitVector::new_normalize({0, 0, -1}));
}
