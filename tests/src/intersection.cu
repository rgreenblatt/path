#include "boost/hana/for_each.hpp"
#include "intersect/accel/accel.h"
#include "intersect/accel/enum_accel/enum_accel.h"
#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "intersect/triangle.h"
#include "lib/optional.h"
#include "lib/span.h"

#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <random>
#include <tuple>

using namespace intersect;
using namespace intersect::accel;
using namespace intersect::accel::enum_accel;

template <AccelType type>
static void test_accelerator(std::mt19937 &gen, const Settings<type> &settings,
                             bool is_gpu) {

  using Test = std::tuple<Ray, Optional<unsigned>>;

  auto run_tests = [&]<ExecutionModel execution_model>(
                       const HostDeviceVector<Triangle> &triangles,
                       const HostDeviceVector<Test> &test_expected) {
    EnumAccel<type, execution_model> inst;

    // Perhaps test partial
    auto ref = inst.template gen<Triangle>(settings, triangles, AABB());

    HostDeviceVector<Optional<unsigned>> results(test_expected.size());
    Span<const Triangle> triangles_span = triangles;

    ThrustData<execution_model> data;

    thrust::transform(
        data.execution_policy(), test_expected.data(),
        test_expected.data() + test_expected.size(), results.data(),
        [=] __host__ __device__(const auto &test) {
          auto [ray, _] = test;
          auto a = ref.intersect_objects(ray, triangles_span);
          return optional_map(a, [](const auto &v) { return v.info.idx; });
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
        Triangle{{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}}}};
    HostDeviceVector<Test> tests = {
        {Ray{{0.1, 0.1, -1}, {0, 0, 1}}, 0},
        {Ray{{0.8, 0.7, -1}, {0, 0, 1}}, nullopt_value},
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
            Triangle{{random_vec(), random_vec(), random_vec()}};
      }

      HostDeviceVector<Test> tests(num_tests);

      EnumAccel<AccelType::LoopAll, ExecutionModel::CPU> loop_all_inst;

      auto loop_all_ref = loop_all_inst.template gen<Triangle>(
          Settings<AccelType::LoopAll>(), triangles, AABB());

      auto get_ground_truth =
          [&](const Ray &ray) -> Optional<unsigned> {
        auto a = loop_all_ref.intersect_objects<Triangle>(ray, triangles);
        if (a.has_value()) {
          return a->info.idx;
        } else {
          return nullopt_value;
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
    test_accelerator(gen, Settings<AccelType::LoopAll>(), is_gpu);
  }
}

TEST(Intersection, kdtree) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {
    test_accelerator(gen, Settings<AccelType::KDTree>(), is_gpu);
  }
}
