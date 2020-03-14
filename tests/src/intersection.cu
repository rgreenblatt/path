#include "boost/hana/for_each.hpp"
#include "intersect/accel/accel.h"
#include "intersect/accel/dir_tree.h"
#include "intersect/accel/impl/kdtree_impl.h"
#include "intersect/accel/impl/loop_all_impl.h"
#include "intersect/accel/kdtree.h"
#include "intersect/accel/loop_all.h"
#include "intersect/impl/triangle_impl.h"
#include "intersect/triangle.h"
#include "lib/span.h"

#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <random>
#include <tuple>

using namespace intersect;
using namespace intersect::accel;

template <AccelType type>
static void test_accelerator(std::mt19937 &gen,
                             const AccelSettings<type> &settings, bool is_gpu) {

  using Test = std::tuple<Ray, thrust::optional<unsigned>>;

  auto run_tests = [&]<ExecutionModel execution_model>(
                       const HostDeviceVector<Triangle> &triangles,
                       const HostDeviceVector<Test> &test_expected) {
    AccelT<type, execution_model, Triangle> inst;

    // Perhaps test partial
    auto ref = inst.gen(settings, triangles, 0, triangles.size(), AABB());

    HostDeviceVector<thrust::optional<unsigned>> results(test_expected.size());

    ThrustData<execution_model> data;

    thrust::transform(
        data.execution_policy(), test_expected.data(),
        test_expected.data() + test_expected.size(), results.data(),
        [=] __host__ __device__(const auto &test) {
          auto [ray, _] = test;
          auto a = IntersectableT<decltype(ref)>::intersect(ray, ref);
          return optional_map(a, [](const auto &v) { return v.info[0]; });
        });

    for (unsigned i = 0; i < test_expected.size(); i++) {
      auto [ray, expected] = test_expected[i];
      auto result = results[i];
      EXPECT_EQ(result.has_value(), expected.has_value());
      if (result.has_value() && expected.has_value()) {
        EXPECT_EQ(*result, *expected);
      }
    }
  };

  {
    HostDeviceVector<Triangle> triangles = {
        Triangle({{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}})};
    HostDeviceVector<Test> tests = {
        {Ray{{0.1, 0.1, -1}, {0, 0, 1}}, 0},
        {Ray{{0.8, 0.7, -1}, {0, 0, 1}}, thrust::nullopt},
        {Ray{{0.3, 0.1, -1}, {0, 0, 1}}, 0},
        {Ray{{0.1, 0.8, -1}, {0, -.7, 1}}, 0},
        {Ray{{0.1, 0.1, -1}, {0, 0, 1}}, 0},
    };

    run_tests.template operator()<ExecutionModel::CPU>(triangles, tests);
  }
    
  const unsigned num_trials = 10;
  const unsigned num_tests = 10;

  {
    std::uniform_int_distribution<unsigned> num_triangles_gen(2, 100);
    std::uniform_real_distribution<float> float_gen(-1, 1);
    for (unsigned trial_idx = 0; trial_idx < num_trials; ++trial_idx) {
      unsigned num_triangles = num_triangles_gen(gen);

      HostDeviceVector<Triangle> triangles(num_triangles);

      auto random_vec = [&] {
        return Eigen::Vector3f{float_gen(gen), float_gen(gen), float_gen(gen)};
      };

      for (unsigned i = 0; i < num_triangles; i++) {
        triangles[i] =
            Triangle(std::array{random_vec(), random_vec(), random_vec()});
      }

      HostDeviceVector<Test> tests(num_tests);

      AccelT<AccelType::LoopAll, ExecutionModel::CPU, Triangle> loop_all_inst;

      auto loop_all_ref =
          loop_all_inst.gen(AccelSettings<AccelType::LoopAll>(), triangles, 0,
                            triangles.size(), AABB());

      auto get_ground_truth =
          [&](const Ray &ray) -> thrust::optional<unsigned> {
        auto a = IntersectableT<decltype(loop_all_ref)>::intersect(
            ray, loop_all_ref);
        if (a.has_value()) {
          return a->info[0];
        } else {
          return thrust::nullopt;
        }
      };

      for (unsigned i = 0; i < num_tests; i++) {
        auto eye = random_vec();
        auto direction = random_vec().normalized();
        Ray ray = {eye, direction};
        tests[i] = Test{ray, get_ground_truth(ray)};
      }

      if (is_gpu) {
        run_tests.template operator()<ExecutionModel::GPU>(triangles, tests);
      } else {
        run_tests.template operator()<ExecutionModel::CPU>(triangles, tests);
      }
    }
  }
}

TEST(Intersection, loop_all) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {
    test_accelerator(gen, AccelSettings<AccelType::LoopAll>(), is_gpu);
  }
}

TEST(Intersection, kdtree) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  HostDeviceVector<kdtree::KDTreeNode<AABB>> copied_nodes;
  for (bool is_gpu : {false, true}) {
    test_accelerator(gen, AccelSettings<AccelType::KDTree>(), is_gpu);
  }
}
