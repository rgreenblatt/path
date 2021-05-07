#include "intersect/accel/accel.h"
#include "intersect/accel/enum_accel/enum_accel.h"
#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/triangle.h"
#include "intersect/triangle_impl.h"
#include "lib/optional.h"
#include "lib/span.h"
#include "meta/all_values/tag.h"

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
    std::optional<unsigned> expected_idx;
  };

  auto run_tests = [&](auto tag, SpanSized<const Triangle> triangles_in,
                       const HostDeviceVector<Test> &test_expected) {
    constexpr ExecutionModel exec = tag;

    EnumAccel<type, exec> inst;

    // for (unsigned i = 0; i < triangles_in.size(); ++i) {
    //   std::cout << "triangle: " << i << std::endl;
    //   print_triangle(triangles_in[i]);
    //   std::cout << std::endl;
    // }

    // Perhaps test partial
    RefPerm ref_perm = inst.gen(settings, triangles_in);
    ASSERT_EQ(ref_perm.permutation.size(), triangles_in.size());
    HostDeviceVector<Triangle> triangles_vec(triangles_in.size());
    HostVector<unsigned> orig_idxs(triangles_in.size());
    for (unsigned i = 0; i < triangles_in.size(); ++i) {
      triangles_vec[i] = triangles_in[ref_perm.permutation[i]];
      orig_idxs[i] = ref_perm.permutation[i];
    }
    auto ref = ref_perm.ref;
    SpanSized<const Triangle> triangles = triangles_vec;

    HostDeviceVector<std::optional<unsigned>> results(test_expected.size());

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
          return optional_map(a, [](const auto &v) { return v.info.idx; });
        });

    for (unsigned i = 0; i < test_expected.size(); i++) {
      auto [ray, expected] = test_expected[i];
      auto result = results[i];
      EXPECT_EQ(result.has_value(), expected.has_value());
      if (result.has_value() && expected.has_value()) {
        EXPECT_EQ(orig_idxs[*result], *expected);
      }
    }
  };

  auto random_vec = [&](auto float_gen) {
    return Eigen::Vector3f{float_gen(gen), float_gen(gen), float_gen(gen)};
  };

  auto random_tests_for_tris = [&](SpanSized<const Triangle> triangles,
                                   auto float_gen, unsigned num_tests) {
    HostDeviceVector<Test> tests(num_tests);

    EnumAccel<AccelType::LoopAll, ExecutionModel::CPU> loop_all_inst;

    auto loop_all_ref =
        loop_all_inst.gen(Settings<AccelType::LoopAll>(), triangles.as_const())
            .ref;

    auto get_ground_truth = [&](const Ray &ray) -> std::optional<unsigned> {
      auto a = loop_all_ref.intersect_objects(
          ray, [&](unsigned idx, const Ray &ray) {
            return triangles[idx].intersect(ray);
          });
      if (a.has_value()) {
        return a->info.idx;
      } else {
        return std::nullopt;
      }
    };

    for (unsigned i = 0; i < num_tests; i++) {
      auto eye = random_vec(float_gen);
      auto direction = UnitVector::new_normalize(random_vec(float_gen));
      Ray ray = {eye, direction};
      tests[i] = Test{ray, get_ground_truth(ray)};
    }

    if (is_gpu) {
#ifndef CPU_ONLY
      run_tests(tag_v<ExecutionModel::GPU>, triangles, tests);
#endif
    } else {
      run_tests(tag_v<ExecutionModel::CPU>, triangles, tests);
    }
  };

  {
    HostVector<Triangle> triangles = {
        Triangle{{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}}}};
    HostDeviceVector<Test> tests = {
        {Ray{{0.1, 0.1, -1}, UnitVector::new_normalize({0, 0, 1})}, 0},
        {Ray{{0.8, 0.7, -1}, UnitVector::new_normalize({0, 0, 1})},
         std::nullopt},
        {Ray{{0.3, 0.1, -1}, UnitVector::new_normalize({0, 0, 1})}, 0},
        {Ray{{0.1, 0.8, -1}, UnitVector::new_normalize({0, -.7, 1})}, 0},
        {Ray{{0.1, 0.1, -1}, UnitVector::new_normalize({0, 0, 1})}, 0},
    };

    if (is_gpu) {
#ifndef CPU_ONLY
      run_tests(tag_v<ExecutionModel::GPU>, triangles, tests);
#endif
    } else {
      run_tests(tag_v<ExecutionModel::CPU>, triangles, tests);
    }

    std::uniform_real_distribution<float> float_gen(-1, 1);
    random_tests_for_tris(triangles, float_gen, 64);
  }

  {
    HostVector<Triangle> triangles = {
        Triangle{{{{0.13, 0, 0}, {0.13, 0.6, 0}, {0.7, 0.6, 0.17}}}},
        Triangle{{{{0.13, 0, 0}, {0.7, 0.6, 0.17}, {0.7, 0, 0.17}}}},
        Triangle{
            {{{-0.24, 1.98, 0.16}, {-0.24, 1.98, -0.22}, {0.23, 1.98, -0.22}}}},
        Triangle{
            {{{-0.24, 1.98, 0.16}, {0.23, 1.98, -0.22}, {0.23, 1.98, 0.16}}}},
    };

    std::uniform_real_distribution<float> float_gen(-2, 2);
    random_tests_for_tris(triangles, float_gen, 512);
  }

  {
    const unsigned num_trials = 16;
    const unsigned num_tests = 256;

    // TODO: consider switching to an actual property based testing framework...
    std::uniform_real_distribution<float> float_gen(-1, 1);
    std::uniform_int_distribution<unsigned> num_triangles_gen(1, 100);
    for (unsigned trial_idx = 0; trial_idx < num_trials; ++trial_idx) {
      unsigned num_triangles = num_triangles_gen(gen);

      HostVector<Triangle> triangles_vec(num_triangles);
      SpanSized<Triangle> triangles = triangles_vec;

      for (unsigned i = 0; i < num_triangles; i++) {
        triangles[i] = Triangle{{random_vec(float_gen), random_vec(float_gen),
                                 random_vec(float_gen)}};
      }

      random_tests_for_tris(triangles, float_gen, num_tests);
    }
  }
}

TEST(Intersection, loop_all) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {
    test_accelerator<AccelType::LoopAll>(gen, {}, is_gpu);
  }
}

TEST(Intersection, naive_partition_bvh) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {
    test_accelerator<AccelType::NaivePartitionBVH>(gen, {}, is_gpu);
  }
}

TEST(Intersection, sbvh) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {
    test_accelerator<AccelType::SBVH>(gen, {}, is_gpu);
  }
}

TEST(Intersection, direction_grid) {
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());
  for (bool is_gpu : {false, true}) {
    test_accelerator<AccelType::DirectionGrid>(gen, {}, is_gpu);
  }
}
