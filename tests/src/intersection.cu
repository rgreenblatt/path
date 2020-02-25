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

    for (const auto &test : test_expected) {
      auto [ray, expected] = test;
      auto a = IntersectableT<decltype(ref)>::intersect(ray, ref);
      EXPECT_EQ(a.has_value(), expected.has_value());
      if (a.has_value() && expected.has_value()) {
        EXPECT_EQ(a->info[0], *expected);
      }
    }
  };

#if 0
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
#endif

#if 0
  {
    HostDeviceVector<Triangle> triangles = {
        Triangle({{{-1, 0, 0}, {1, 1, 0}, {0, 1, 0}}}),
        Triangle({{{3, 3, 0}, {8, 9, 0}, {10, 3, 0}}}),
    };

    const HostDeviceVector<Test> tests{
        {Ray{{0, 0.7, 10}, {0, 0, -1}}, 0},
        {Ray{{3, 0, 10}, {0, 0, -1}}, thrust::nullopt},
        {Ray{{0.25, 0, 10}, {0, 0, -1}}, 0},
        {Ray{{5, 5, 10}, {0, 0, -1}}, 1},
        {Ray{{11, 5, 10}, {0, 0, -1}}, thrust::nullopt},
        {Ray{{4, 18, 10}, {0, -1, -1}}, 1},
    };

    run_tests.template operator()<ExecutionModel::CPU>(triangles, tests);
  }
#endif

#if 0
  {
    HostDeviceVector<scene::ShapeData> shapes = {
        make_id_cube(Eigen::Affine3f::Identity(), 0),
        make_id_cube(
            Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f{-3, 0, 0.01})),
            1)};

    auto accel = accel_gen(shapes);

    const HostDeviceVector<Test> tests{
        {{0, 0, 10}, {0, 0, -1}, 0},
        {{3, 0, 10}, {0, 0, -1}, thrust::nullopt},
        {{0.25, 0, 10}, {0, 0, -1}, 0},
        {{0.0, 0, 10}, {0.03, -0.04, -1}, 0},
        {{0.0, 0, 10}, {0.06, -0.04, -1}, thrust::nullopt},
        {{0.3, 0, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{0.0, -0.2, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{0, 0, -2}, {0, 0, -1}, thrust::nullopt},
        {{0, 0, -2}, {0, 0, 1}, 0},
        {{5, 0, 0.2}, {-1, 0, 0}, 0},
        {{5, 0, 0.2}, {-1, 0, -0.03}, 0},
        {{5, 0, 0.2}, {-1, 0, 0.2}, thrust::nullopt},
        {{5, 0, -0.7}, {-1, 0, 0.2}, 0},
        {{-7, 0, -0.7}, {1, 0, 0.2}, 1},
    };
    test(accel, tests, shapes);
  }
#endif

#if 0
  {
    HostDeviceVector<scene::ShapeData> shapes = {
        make_id_cube(Eigen::Affine3f::Identity(), 0),
        make_id_cube(
            Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f{-3, 0, 0})),
            1)};

    auto accel = accel_gen(shapes);

    const HostDeviceVector<Test> tests{
        {{0, 0, 10}, {0, 0, -1}, 0},
        {{3, 0, 10}, {0, 0, -1}, thrust::nullopt},
        {{0.25, 0, 10}, {0, 0, -1}, 0},
        {{0.0, 0, 10}, {0.03, -0.04, -1}, 0},
        {{0.0, 0, 10}, {0.06, -0.04, -1}, thrust::nullopt},
        {{0.3, 0, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{0.0, -0.2, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{0, 0, -2}, {0, 0, -1}, thrust::nullopt},
        {{0, 0, -2}, {0, 0, 1}, 0},
        {{5, 0, 0.2}, {-1, 0, 0}, 0},
        {{5, 0, 0.2}, {-1, 0, -0.03}, 0},
        {{5, 0, 0.2}, {-1, 0, 0.2}, thrust::nullopt},
        {{5, 0, -0.7}, {-1, 0, 0.2}, 0},
        {{-7, 0, -0.7}, {1, 0, 0.2}, 1},
        {{-3, 0, 5}, {0, 0, -1}, 1},
        {{-3.0, 0, 10}, {0.03, -0.04, -1}, 1},
        {{-3.0, 0, 10}, {0.06, -0.04, -1}, thrust::nullopt},
        {{-2.7, 0, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{-4, 0, 10}, {0.07, -0.04, -1}, 1},
    };
    test(accel, tests, shapes);
  }
#endif

#if 0
  {
    HostDeviceVector<scene::ShapeData> shapes = {
        make_id_cube(Eigen::Affine3f::Identity(), 0),
        make_id_cube(
            Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f{-3, 0, 0.01})),
            1),
        make_id_cube(
            Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f{-3, 3, 0.01})),
            2)};

    auto accel = accel_gen(shapes);

    const HostDeviceVector<Test> tests{
        {{0, 0, 10}, {0, 0, -1}, 0},
        {{3, 0, 10}, {0, 0, -1}, thrust::nullopt},
        {{0.25, 0, 10}, {0, 0, -1}, 0},
        {{0.0, 0, 10}, {0.03, -0.04, -1}, 0},
        {{0.0, 0, 10}, {0.06, -0.04, -1}, thrust::nullopt},
        {{0.3, 0, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{0.0, -0.2, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{0, 0, -2}, {0, 0, -1}, thrust::nullopt},
        {{0, 0, -2}, {0, 0, 1}, 0},
        {{5, 0, 0.2}, {-1, 0, 0}, 0},
        {{5, 0, 0.2}, {-1, 0, -0.03}, 0},
        {{5, 0, 0.2}, {-1, 0, 0.2}, thrust::nullopt},
        {{5, 0, -0.7}, {-1, 0, 0.2}, 0},
        {{-3.0, 0, 10}, {0.03, -0.04, -1}, 1},
        {{-3.0, 0, 10}, {0.06, -0.04, -1}, thrust::nullopt},
        {{-4, 0, 10}, {0.07, -0.04, -1}, 1},
        {{-3, 3, 10}, {0.0, 0.0, -1}, 2},
        {{10, 10, 10}, {-1, -1, -1}, 0},
        {{10, 10, 10}, {-1.3, -1, -1}, 1},
        {{10, 10, 10}, {-1.3, -0.7, -1}, 2},
    };
    test(accel, tests, shapes);
  }
#endif

#if 0
  {
    HostDeviceVector<scene::ShapeData> shapes = {
        make_id_cube(Eigen::Affine3f::Identity(), 0),
        make_id_cube(
            Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f{-3, 0, 0.01})),
            1),
        make_id_cube(
            Eigen::Affine3f(Eigen::Translation3f(Eigen::Vector3f{-3, 3, 0.01})),
            2),
        make_id_cube(Eigen::Affine3f(
                         Eigen::Translation3f(Eigen::Vector3f{-1.5, 0, 0.01})),
                     3)};

    auto accel = accel_gen(shapes);

    const HostDeviceVector<Test> tests{
        {{0, 0, 10}, {0, 0, -1}, 0},
        {{3, 0, 10}, {0, 0, -1}, thrust::nullopt},
        {{0.25, 0, 10}, {0, 0, -1}, 0},
        {{0.0, 0, 10}, {0.03, -0.04, -1}, 0},
        {{0.0, 0, 10}, {0.06, -0.04, -1}, thrust::nullopt},
        {{0.3, 0, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{0.0, -0.2, 10}, {0.03, -0.04, -1}, thrust::nullopt},
        {{0, 0, -2}, {0, 0, -1}, thrust::nullopt},
        {{0, 0, -2}, {0, 0, 1}, 0},
        {{5, 0, 0.2}, {-1, 0, 0}, 0},
        {{5, 0, 0.2}, {-1, 0, -0.03}, 0},
        {{5, 0, 0.2}, {-1, 0, 0.2}, thrust::nullopt},
        {{5, 0, -0.7}, {-1, 0, 0.2}, 0},
        {{-3.0, 0, 10}, {0.03, -0.04, -1}, 1},
        {{-3.0, 0, 10}, {0.06, -0.04, -1}, thrust::nullopt},
        {{-2.7, 0, 10}, {0.08, -0.04, -1}, 3},
        {{-4, 0, 10}, {0.07, -0.04, -1}, 1},
        {{-3, 3, 10}, {0.0, 0.0, -1}, 2},
        {{10, 10, 10}, {-1, -1, -1}, 0},
        {{10, 10, 10}, {-1.3, -1, -1}, 1},
        {{10, 10, 10}, {-1.3, -0.7, -1}, 2},
    };
    test(accel, tests, shapes);
  }
#endif

#if 1
  {
    unsigned num_trials = 100;
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

      unsigned num_tests = 30;

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

      run_tests.template operator()<ExecutionModel::CPU>(triangles, tests);
    }
  }
#endif
}

#if 1
TEST(Intersection, loop_all) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  /* for (bool is_gpu : {false, true}) { */
  for (bool is_gpu : {false}) {
    test_accelerator(gen, AccelSettings<AccelType::LoopAll>(), is_gpu);
  }
}
#endif

#if 1
TEST(Intersection, kdtree) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  HostDeviceVector<kdtree::KDTreeNode<AABB>> copied_nodes;
  for (bool is_gpu : {false /*, true*/}) {
    test_accelerator(gen, AccelSettings<AccelType::KDTree>(), is_gpu);
  }
}
#endif

#if 0
TEST(Intersection, dir_tree) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  dir_tree::DirTreeGenerator<ExecutionModel::CPU> cpu_gen;
  /* dir_tree::DirTreeGenerator<ExecutionModel::GPU> gpu_gen; */
  /* for (bool is_gpu : {false, true}) { */
  for (bool is_gpu : {false}) {
    test_accelerator(
        gen,
        [&](SpanSized<scene::ShapeData> shapes) {
          Eigen::Vector3f min_bound(std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max(),
                                    std::numeric_limits<float>::max());
          Eigen::Vector3f max_bound(std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::lowest(),
                                    std::numeric_limits<float>::lowest());

          for (const auto &shape : shapes) {
            auto [min_bound_new, max_bound_new] =
                ray::detail::accel::get_transformed_bounds(
                    shape.get_transform());
            min_bound = min_bound.cwiseMin(min_bound_new);
            max_bound = max_bound.cwiseMax(max_bound_new);
          }

          auto lookup =
              cpu_gen.generate(shapes, 4, min_bound, max_bound, false);
          return dir_tree::DirTreeLookupRef(lookup);
        },
        is_gpu);
  }
}
#endif
