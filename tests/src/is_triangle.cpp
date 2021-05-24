#include "generate_data/shadowed.h"
#include "generate_data/triangle.h"
#include "integrate/sample_triangle.h"
#include "intersect/ray.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/info/print_triangle.h"
#include "rng/uniform/uniform.h"

#include <gtest/gtest.h>

#include <iostream>
#include <random>

using namespace generate_data;

TEST(shadowed, random) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

  std::uniform_real_distribution dist(0., 1.);

  auto random_vec = [&]() {
    return Eigen::Vector3d{dist(gen), dist(gen), dist(gen)};
  };

  auto gen_tri = [&](double scale, Eigen::Vector3d offset) -> Triangle {
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

  std::bernoulli_distribution use_uniform_tris(0.6);

  for (unsigned i = 0; i < num_tests; ++i) {
    std::array<Triangle, 3> tris;
    if (use_uniform_tris(gen)) {
      auto gen = [&]() { return gen_tri(1., Eigen::Vector3d::Zero()); };
      tris = {gen(), gen(), gen()};
    } else {
      // try to create blocking conditions
      double edge_scale = 0.2;
      double middle_scale = 3.0;
      tris = {gen_tri(edge_scale, -Eigen::Vector3d::Ones()),
              gen_tri(middle_scale, Eigen::Vector3d::Zero()),
              gen_tri(edge_scale, Eigen::Vector3d::Ones())};
    }

    for (unsigned inner = 0; inner < tris.size(); ++inner) {
      unsigned outer_0 = (inner + 1) % tris.size();
      unsigned outer_1 = (inner + 2) % tris.size();
      std::array<Triangle, 2> outer_tries{tris[outer_0], tris[outer_1]};
      const auto &inner_tri = tris[inner];

      // TODO
    }
  }
  // 3 possible inner tris
  std::cout << "total possible between fails: " << total_possible_between_fails
            << " / " << num_tests * 3 << std::endl;
  std::cout << "total possible blocking fails: "
            << total_possible_blocking_fails << " / " << num_tests * 3
            << std::endl;
}
