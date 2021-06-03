#include "generate_data/possibly_shadowed.h"
#include "generate_data/clip_by_plane.h"
#include "generate_data/shadowed.h"
#include "generate_data/triangle.h"
#include "generate_data/triangle_subset.h"
#include "generate_data/triangle_subset_intersection.h"

#include <gtest/gtest.h>

#include <random>

using namespace generate_data;

TEST(possibly_shadowed, compare_to_shadowed) {
  unsigned n_tris = 1024;
  std::mt19937 gen(testing::UnitTest::GetInstance()->random_seed());

  double total_diff = 0;
  double total_count = 0;
  for (unsigned i = 0; i < n_tris; ++i) {
    auto random_vec = [&]() {
      std::uniform_real_distribution<double> dist(0., 1.);
      return Eigen::Vector3d{
          dist(gen),
          dist(gen),
          dist(gen),
      };
    };

    auto gen_tri = [&](float z_offset) -> Triangle {
      Eigen::Vector3d addr = Eigen::Vector3d::UnitZ() * z_offset;
      return {{
          random_vec() + addr,
          random_vec() + addr,
          random_vec() + addr,
      }};
    };

    std::array<Triangle, 3> triangles;
    if (std::bernoulli_distribution(0.8)(gen)) {
      triangles = {
          gen_tri(0.),
          gen_tri(0.),
          gen_tri(0.),
      };
    } else {
      triangles = {
          gen_tri(0.),
          gen_tri(0.5),
          gen_tri(1.),
      };
    }

    for (unsigned blocker_idx = 0; blocker_idx < triangles.size();
         ++blocker_idx) {
      unsigned from = (blocker_idx + 1) % triangles.size();
      unsigned onto = (blocker_idx + 2) % triangles.size();
      std::array<bool, 2> flip_normal = {
          triangles[from].normal_raw().dot(triangles[onto].centroid() -
                                           triangles[from].vertices[0]) < 0.,
          triangles[onto].normal_raw().dot(triangles[from].centroid() -
                                           triangles[onto].vertices[0]) < 0.,
      };
      std::array<Eigen::Vector3d, 2> flipped_normals = {
          triangles[from].normal_raw() * (flip_normal[0] ? -1. : 1.),
          triangles[onto].normal_raw() * (flip_normal[1] ? -1. : 1.),
      };

      // clip by triangle_onto plane
      auto from_region = clip_by_plane_point(
          flipped_normals[1], triangles[onto].vertices[0], triangles[from]);
      always_assert(from_region.type() != TriangleSubsetType::None);
      auto blocker_region_initial =
          clip_by_plane_point(flipped_normals[1], triangles[onto].vertices[0],
                              triangles[blocker_idx]);
      auto blocker_region = triangle_subset_intersection(
          blocker_region_initial,
          clip_by_plane_point(flipped_normals[0], triangles[from].vertices[0],
                              triangles[blocker_idx]));

      if (blocker_region.type() == TriangleSubsetType::None) {
        continue;
      }

      bool is_possibly_shadowed =
          possibly_shadowed({&triangles[from], &triangles[onto]},
                            triangles[blocker_idx], flip_normal);

      bool is_actually_shadowed =
          partially_shadowed(triangles[from], from_region,
                             triangles[blocker_idx], blocker_region,
                             triangles[onto], flip_normal[1])
              .partially_shadowed.type() != TriangleSubsetType::None;
      ++total_count;
      if (is_actually_shadowed != is_possibly_shadowed) {
        ++total_diff;
      }

      // is_actually_shadowed => is_possibly_shadowed
      ASSERT_TRUE(!is_actually_shadowed || is_possibly_shadowed);
    }
  }
  std::cout << "error rate: " << total_diff / total_count << std::endl;
}
