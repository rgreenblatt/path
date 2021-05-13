#include "integrate/sample_triangle.h"
#include "intersect/is_triangle_between.h"
#include "intersect/is_triangle_blocking.h"
#include "intersect/ray.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/info/print_triangle.h"
#include "rng/uniform/uniform.h"

#include <gtest/gtest.h>

#include <iostream>
#include <random>

using namespace intersect;

TEST(is_triangle, random) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

  rng::uniform::Uniform<ExecutionModel::CPU>::Ref::State rng{
      unsigned(testing::UnitTest::GetInstance()->random_seed()),
  };

  auto random_vec = [&]() {
    return Eigen::Vector3f{rng.next(), rng.next(), rng.next()};
  };

  auto gen_tri = [&](float scale, Eigen::Vector3f offset) -> Triangle {
    return {{
        scale * random_vec() + offset,
        scale * random_vec() + offset,
        scale * random_vec() + offset,
    }};
  };

  unsigned num_tests = 1024;
  unsigned base_num_random_checks = 128;
  unsigned max_num_random_checks = 1024;

  unsigned total_possible_between_fails = 0;
  unsigned total_possible_blocking_fails = 0;

  for (unsigned i = 0; i < num_tests; ++i) {
    std::array<Triangle, 3> tris;
    if (rng.next() < 0.6f) {
      auto gen = [&]() { return gen_tri(1.f, Eigen::Vector3f::Zero()); };
      tris = {gen(), gen(), gen()};
    } else {
      // try to create blocking conditions
      float edge_scale = 0.2;
      float middle_scale = 3.0;
      tris = {gen_tri(edge_scale, -Eigen::Vector3f::Ones()),
              gen_tri(middle_scale, Eigen::Vector3f::Zero()),
              gen_tri(edge_scale, Eigen::Vector3f::Ones())};
    }

    for (unsigned inner = 0; inner < tris.size(); ++inner) {
      unsigned outer_0 = (inner + 1) % tris.size();
      unsigned outer_1 = (inner + 2) % tris.size();
      std::array<Triangle, 2> outer_tries{tris[outer_0], tris[outer_1]};
      const auto &inner_tri = tris[inner];

      bool first_order_is_between = is_triangle_between(outer_tries, inner_tri);
      bool first_order_is_blocking =
          is_triangle_blocking(outer_tries, inner_tri);

      // first_order_is_blocking implies first_order_is_between
      EXPECT_TRUE(!first_order_is_blocking || first_order_is_between);

      unsigned num_random_checks = first_order_is_between
                                       ? max_num_random_checks
                                       : base_num_random_checks;

      bool actually_found_blocked_ray_between = false;
      bool actually_found_unblocked_ray_between = false;
      for (unsigned j = 0;
           j < num_random_checks && (!actually_found_blocked_ray_between ||
                                     !actually_found_unblocked_ray_between);
           ++j) {
        auto p0 = integrate::sample_triangle(outer_tries[0], rng);
        auto p1 = integrate::sample_triangle(outer_tries[1], rng);

        Eigen::Vector3f dir = p1 - p0;
        float dist_limit = dir.norm();

        intersect::Ray ray{
            .origin = p0,
            .direction = UnitVector::new_normalize(dir),
        };

        auto intersection = inner_tri.intersect(ray);
        if (intersection.has_value() &&
            intersection->intersection_dist >= 0.f &&
            intersection->intersection_dist <= dist_limit) {
          actually_found_blocked_ray_between = true;
        } else {
          actually_found_unblocked_ray_between = true;
        }
      }

      if (!actually_found_blocked_ray_between && first_order_is_between) {
        ++total_possible_between_fails;
      } else {
        EXPECT_EQ(actually_found_blocked_ray_between, first_order_is_between);
      }

      if (!actually_found_unblocked_ray_between && !first_order_is_blocking) {
        ++total_possible_blocking_fails;
      } else {
        EXPECT_EQ(!actually_found_unblocked_ray_between,
                  first_order_is_blocking);
      }

      std::swap(outer_tries[0], outer_tries[1]);
      bool second_order_is_between =
          is_triangle_between(outer_tries, inner_tri);
      EXPECT_EQ(first_order_is_between, second_order_is_between);
      bool second_order_order_is_blocking =
          is_triangle_blocking(outer_tries, inner_tri);
      EXPECT_EQ(first_order_is_blocking, second_order_order_is_blocking);
    }
  }
  // 3 possible inner tris
  std::cout << "total possible between fails: " << total_possible_between_fails
            << " / " << num_tests * 3 << std::endl;
  std::cout << "total possible blocking fails: "
            << total_possible_blocking_fails << " / " << num_tests * 3
            << std::endl;
}
