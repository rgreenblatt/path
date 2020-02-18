#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include "ray/detail/accel/dir_tree/impl/dir_tree_lookup_ref_impl.h"
#include "ray/detail/accel/kdtree/kdtree.h"
#include "ray/detail/accel/kdtree/kdtree_ref_impl.h"
#include "ray/detail/accel/loop_all.h"
#include "ray/detail/intersection/solve.h"

#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <random>

using namespace ray::detail;
using namespace ray::detail::accel;

template <typename AccelGen>
static void test_accelerator(std::mt19937 &gen, const AccelGen &accel_gen,
                             bool is_gpu) {
  using Test =
      std::tuple<Eigen::Vector3f, Eigen::Vector3f, thrust::optional<float>>;

  auto make_id_cube = [](const Eigen::Affine3f &transform, unsigned i) {
    scene::Material material;
    material.ior = i;

    return scene::ShapeData(transform, material, scene::Shape::Cube);
  };

  auto get_test_runner = [](const auto &accel,
                            SpanSized<const scene::ShapeData> shapes) {
    return [=] __host__ __device__(const Test &test) {
      thrust::optional<BestIntersection> best;

      intersection::solve(
          accel, shapes, std::get<0>(test), std::get<1>(test), thrust::nullopt,
          std::numeric_limits<unsigned>::max(),
          [&](const thrust::optional<BestIntersection> &new_best) {
            best = optional_min(best, new_best);

            return false;
          });

      return optional_map(best, [=](const BestIntersection &b) {
        return shapes[b.shape_idx].get_material().ior;
      });
    };
  };

  auto test = [&](const auto &accel, SpanSized<const Test> tests,
                  SpanSized<scene::ShapeData> shapes) {
    auto run_test = get_test_runner(accel, shapes);

    HostDeviceVector<thrust::optional<float>> test_results(tests.size());
    if (is_gpu) {
      thrust::transform(thrust::device, tests.data(),
                        tests.data() + tests.size(), test_results.data(),
                        run_test);
    } else {
      std::transform(tests.begin(), tests.end(), test_results.begin(),
                     run_test);
    }

    for (unsigned i = 0; i < tests.size(); ++i) {
      auto test_val = std::get<2>(tests[i]);
      EXPECT_EQ(test_results[i].has_value(), test_val.has_value());
      if (test_results[i].has_value() && test_val.has_value()) {
        EXPECT_EQ(*test_results[i], *test_val);
      }
    }
  };

#if 1
  {
    HostDeviceVector<scene::ShapeData> shapes = {
        make_id_cube(Eigen::Affine3f::Identity(), 0)};

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
    };
    test(accel, tests, shapes);
  }
#endif

#if 1
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

#if 1
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

#if 1
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

#if 1
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
    unsigned num_trials = 10;
    std::uniform_int_distribution<unsigned> num_shapes_gen(1, 50);
    std::uniform_real_distribution<float> float_gen(-1, 1);
    for (unsigned trial_idx = 0; trial_idx < num_trials; ++trial_idx) {
      unsigned num_shapes = num_shapes_gen(gen);

      HostDeviceVector<scene::ShapeData> shapes(num_shapes);

      auto random_vec = [&] {
        return Eigen::Vector3f{float_gen(gen), float_gen(gen), float_gen(gen)};
      };

      for (unsigned i = 0; i < num_shapes; i++) {
        shapes[i] =
            make_id_cube(Eigen::AngleAxisf(float_gen(gen) * M_PI,
                                           random_vec().normalized()) *
                             Eigen::Translation3f(random_vec()) *
                             Eigen::Affine3f(Eigen::Scaling(random_vec())),
                         i);
      }

      auto accel = accel_gen(shapes);

      unsigned num_tests = 100;

      HostDeviceVector<Test> tests(num_tests);

      auto get_ground_truth =
          get_test_runner(accel::LoopAll(num_shapes), shapes);

      for (unsigned i = 0; i < num_tests; i++) {
        auto eye = random_vec();
        auto direction = random_vec().normalized();
        tests[i] =
            Test{eye, direction, get_ground_truth(Test{eye, direction, 0})};
      }

      test(accel, tests, shapes);
    }
  }
#endif
}

TEST(Intersection, loop_all) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {

    test_accelerator(
        gen,
        [](SpanSized<const scene::ShapeData> shapes) {
          return accel::LoopAll(shapes.size());
        },
        is_gpu);
  }
}

TEST(Intersection, kdtree) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  HostDeviceVector<kdtree::KDTreeNode<AABB>> copied_nodes;
  for (bool is_gpu : {false, true}) {
    test_accelerator(
        gen,
        [&](SpanSized<scene::ShapeData> shapes) {
          auto nodes =
              kdtree::construct_kd_tree(shapes.data(), shapes.size(), 100, 3);
          copied_nodes.clear();
          copied_nodes.insert(copied_nodes.end(), nodes.begin(), nodes.end());
          return kdtree::KDTreeRef(copied_nodes, shapes.size());
        },
        is_gpu);
  }
}

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
