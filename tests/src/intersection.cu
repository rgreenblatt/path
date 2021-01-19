#include "intersect/accel/accel.h"
#include "intersect/accel/enum_accel/enum_accel.h"
#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/optional.h"
#include "lib/span.h"
#include "meta/tag.h"

#include <gtest/gtest.h>

#include <random>

using namespace intersect;
using namespace intersect::accel;
using namespace intersect::accel::enum_accel;

template <AccelType type>
static void test_accelerator(std::mt19937 &gen, const Settings<type> &settings,
                             bool is_gpu) {

  struct Test {
    Ray ray;
    Optional<unsigned> expected_idx;
  };

  auto run_tests = [&](auto tag, SpanSized<const Triangle> triangles,
                       const HostDeviceVector<Test> &test_expected) {
    constexpr auto exec = decltype(tag)::value;

    EnumAccel<type, exec> inst;

    // Perhaps test partial
    auto ref = inst.gen(settings, triangles, AABB());

    HostDeviceVector<Optional<unsigned>> results(test_expected.size());

    ThrustData<exec> data;

    thrust::transform(
        data.execution_policy(), test_expected.data(),
        test_expected.data() + test_expected.size(), results.data(),
        [=] HOST_DEVICE(const Test &test) {
          auto [ray, _] = test;
          auto a =
              ref.intersect_objects(ray, [&](unsigned idx, const Ray &ray) {
                return triangles[idx].intersect(ray);
              });
          return a.op_map([](const auto &v) { return v.info.idx; });
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
        {Ray{{0.1, 0.1, -1}, UnitVector::new_normalize({0, 0, 1})}, 0},
        {Ray{{0.8, 0.7, -1}, UnitVector::new_normalize({0, 0, 1})},
         nullopt_value},
        {Ray{{0.3, 0.1, -1}, UnitVector::new_normalize({0, 0, 1})}, 0},
        {Ray{{0.1, 0.8, -1}, UnitVector::new_normalize({0, -.7, 1})}, 0},
        {Ray{{0.1, 0.1, -1}, UnitVector::new_normalize({0, 0, 1})}, 0},
    };

    run_tests(TagV<ExecutionModel::CPU>, triangles, tests);
  }

  const unsigned num_trials = 10;
  const unsigned num_tests = 10;

  {
    std::uniform_int_distribution<unsigned> num_triangles_gen(2, 100);
    std::uniform_real_distribution<float> float_gen(-1, 1);
    for (unsigned trial_idx = 0; trial_idx < num_trials; ++trial_idx) {
      unsigned num_triangles = num_triangles_gen(gen);

      HostDeviceVector<Triangle> triangles_vec(num_triangles);
      SpanSized<Triangle> triangles = triangles_vec;

      auto random_vec = [&] {
        return Eigen::Vector3f{float_gen(gen), float_gen(gen), float_gen(gen)};
      };

      for (unsigned i = 0; i < num_triangles; i++) {
        triangles[i] = Triangle{{random_vec(), random_vec(), random_vec()}};
      }

      HostDeviceVector<Test> tests(num_tests);

      EnumAccel<AccelType::LoopAll, ExecutionModel::CPU> loop_all_inst;

      auto loop_all_ref = loop_all_inst.gen(Settings<AccelType::LoopAll>(),
                                            triangles.as_const(), AABB());

      auto get_ground_truth = [&](const Ray &ray) -> Optional<unsigned> {
        auto a = loop_all_ref.intersect_objects(
            ray, [&](unsigned idx, const Ray &ray) {
              return triangles[idx].intersect(ray);
            });
        if (a.has_value()) {
          return a->info.idx;
        } else {
          return nullopt_value;
        }
      };

      for (unsigned i = 0; i < num_tests; i++) {
        auto eye = random_vec();
        auto direction = UnitVector::new_normalize(random_vec());
        Ray ray = {eye, direction};
        tests[i] = Test{ray, get_ground_truth(ray)};
      }

      if (is_gpu) {
#ifndef CPU_ONLY
        run_tests(TagV<ExecutionModel::GPU>, triangles, tests);
#endif
      } else {
        run_tests(TagV<ExecutionModel::CPU>, triangles, tests);
      }
    }
  }
}

TEST(Intersection, loop_all) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {
    test_accelerator<AccelType::LoopAll>(gen, {}, is_gpu);
  }
}

TEST(Intersection, kdtree) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {
    test_accelerator<AccelType::KDTree>(gen, {}, is_gpu);
  }
}
