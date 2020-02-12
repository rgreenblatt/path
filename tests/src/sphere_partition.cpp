#include "ray/detail/accel/dir_tree/sphere_partition.h"
#include "ray/detail/accel/dir_tree/impl/sphere_partition_impl.h"

#include <dbg.h>
#include <gtest/gtest.h>

using namespace ray::detail::accel::dir_tree;

constexpr float epsilon = 1e-6;

TEST(SpherePartition, conversions) {
  auto check = [](Eigen::Vector3f vec, float colatitude, float longitude) {
    vec.normalize();
    {
      auto [actual_colatitude, actual_longitude] =
          HalfSpherePartition::vec_to_colatitude_longitude(vec);
      EXPECT_NEAR(actual_colatitude, colatitude, epsilon);
      EXPECT_NEAR(actual_longitude, longitude, epsilon);
    }
    {
      auto actual_vec = HalfSpherePartition::colatitude_longitude_to_vec(
          colatitude, longitude);

      EXPECT_NEAR(actual_vec.x(), vec.x(), epsilon);
      EXPECT_NEAR(actual_vec.y(), vec.y(), epsilon);
      EXPECT_NEAR(actual_vec.z(), vec.z(), epsilon);
    }
  };

  check({0, 1, 0}, float(0), 0);
  check({0, -1, 0}, float(M_PI), 0);
  check({1, 0, 0}, float(M_PI) / 2.0f, 0);
  check({0, 0, 1}, float(M_PI) / 2.0f, float(M_PI) / 2);
  check({0, 0, -1}, float(M_PI) / 2.0f, -float(M_PI) / 2);
  check({0.5, 0, 0.5}, float(M_PI) / 2.0f, float(M_PI) / 4.0f);
}

TEST(SpherePartition, construct) {
  auto check_properties = [](const HalfSpherePartition &partition) {
    {
      auto [colatitude, longitude] =
          partition.get_center_colatitude_longitude(0, 0);
      EXPECT_NEAR(colatitude, float(0), epsilon);
      EXPECT_NEAR(longitude, float(0), epsilon);
    }

    {
      auto vec = partition.get_center_vec(0, 0);
      EXPECT_NEAR(vec.x(), 0, epsilon);
      EXPECT_NEAR(vec.y(), 1, epsilon);
      EXPECT_NEAR(vec.z(), 0, epsilon);
    }

    {
      EXPECT_GE(partition.size(), partition.colatitude_divs().size());
      unsigned last_idx = 0;
      for (const auto &region : partition.colatitude_divs()) {
        EXPECT_EQ(region.start_index, last_idx);
        last_idx = region.end_index;
      }
      EXPECT_EQ(partition.size(), last_idx);
    }
  };

  for (unsigned size : {1, 2, 3, 4, 6, 11, 37, 60}) {
    HostDeviceVector<HalfSpherePartition::ColatitudeDiv> regions;
    HalfSpherePartition partition(size, regions);
    dbg(size);
    auto divs = partition.colatitude_divs();
    unsigned num_divs = divs.size();
    dbg(num_divs);
    auto last_div = divs[num_divs - 1];
    dbg(last_div.end_index - last_div.start_index);
    check_properties(partition);
  }
}

TEST(SpherePartition, get_closest) {
    HostDeviceVector<HalfSpherePartition::ColatitudeDiv> regions;
    HalfSpherePartition partition(5, regions); // 1 collar (and cap)

    EXPECT_EQ(partition.size(), 5); // tests below assume exactly 5 regions
    EXPECT_EQ(partition.get_closest({0, 1, 0}), (std::tuple{0u, false}));
    EXPECT_EQ(partition.get_closest({0, -1, 0}), (std::tuple{0u, true}));
    EXPECT_EQ(partition.get_closest({0.001, 0.983, -0.03}),
              (std::tuple{0u, false}));
    EXPECT_EQ(partition.get_closest({0.001, -0.983, -0.03}),
              (std::tuple{0u, true}));
    EXPECT_EQ(partition.get_closest({-0.5, 0.5, -0.03}),
              (std::tuple{1u, false}));
    EXPECT_EQ(partition.get_closest({0.5, -0.5, 0.03}),
              (std::tuple{1u, true}));
    EXPECT_EQ(partition.get_closest({-0.5, 0.5, 0.03}),
              (std::tuple{4u, false}));
    EXPECT_EQ(partition.get_closest({0.5, -0.5, -0.03}),
              (std::tuple{4u, true}));
}

// TODO: consider adding more tests to verify regions are spread correctly...
