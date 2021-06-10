#include "generate_data/full_scene/generate_data.h"
#include "generate_data/full_scene/scene_generator.h"
#include "generate_data/possibly_shadowed.h"
#include "generate_data/shadowed.h"
#include "generate_data/subset_to_multi.h"
#include "generate_data/triangle.h"
#include "generate_data/triangle_subset_intersection.h"
#include "generate_data/triangle_subset_union.h"
#include "lib/vector_type.h"

#include "dbg.h"
#include "generate_data/print_region.h"

namespace generate_data {
namespace full_scene {
// template <bool is_image>
void generate_data(int max_tris, int n_samples_per_scene_or_dim, int n_samples,
                   unsigned seed) {
  VectorT<scene::Scene> scenes;
  VectorT<unsigned> mesh_threshs;
  SceneGenerator generator;
  always_assert(max_tris > 3);
  int tri_count = 0;
  std::mt19937 rng(seed);
  while (tri_count < max_tris) {
    auto vals = generator.generate(rng);
    auto scene = std::get<0>(vals);
    int new_tri_count = tri_count + scene.triangles().size();
    // if (new_tri_count > max_tris) {
    //   if (tri_count == 0) {
    //     continue; // try again
    //   } else {
    //     tri_count = new_tri_count;
    //     break; // we have enough
    //   }
    // }
    tri_count = new_tri_count;
    scenes.push_back(scene);
    mesh_threshs.push_back(std::get<1>(vals));
  }
  dbg(tri_count);

  VectorT<VectorT<Triangle>> all_triangles(scenes.size());
  std::transform(scenes.begin(), scenes.end(), all_triangles.begin(),
                 [&](const scene::Scene &scene) {
                   VectorT<Triangle> tris(scene.triangles().size());
                   std::transform(scene.triangles().begin(),
                                  scene.triangles().end(), tris.begin(),
                                  [&](const auto &tri) {
                                    return tri.template cast<double>();
                                  });
                   return tris;
                 });

  struct BlockerInfo {
    unsigned idx;
    // l onto and then r onto
    std::array<PartiallyShadowedInfo, 2> partially_shadowed_infos;
    std::array<TotallyShadowedInfo, 2> totally_shadowed_infos;
  };

  struct LightingPair {
    std::array<unsigned, 2> idxs;
    std::array<bool, 2> is_neg_faces;
    std::array<TriangleSubset, 2> regions;

    VectorT<BlockerInfo> infos;

    std::array<TriangleMultiSubset, 2> overall_partially_shadowed;
    std::array<TriangleMultiSubset, 2> overall_totally_shadowed;
    std::array<VectorT<TriangleMultiSubset>, 2> overall_from_each_point;
  };

  VectorT<VectorT<LightingPair>> lighting_pairs(scenes.size());

  dbg(scenes.size());

  // TODO:
  // find all blocking triangles
  // consider if triangles could be merged somehow (mesh)
  // consider if blocker is actually also blocked by another tri
  // compute feature for all blocking triangles
  for (unsigned i = 0; i < scenes.size(); ++i) {
    const auto &tris = all_triangles[i];
    // we only need to go one way
    for (unsigned j = 0; j < tris.size(); ++j) {
      const auto &tri_onto = tris[j];
      auto onto_normal_base = *tri_onto.normal();
      for (bool onto_is_neg_face : {false, true}) {
        auto onto_normal =
            onto_is_neg_face ? -onto_normal_base : onto_normal_base;
        for (unsigned k = j + 1; k < tris.size(); ++k) {
          const auto &tri_from = tris[k];

          auto from_region =
              clip_by_plane_point(onto_normal, tri_onto.vertices[0], tri_from);

          if (from_region.type() == TriangleSubsetType::None) {
            continue;
          }

          auto from_normal_base = *tri_from.normal();

          for (bool from_is_neg_face : {false, true}) {
            auto from_normal =
                from_is_neg_face ? -from_normal_base : from_normal_base;

            auto onto_region = clip_by_plane_point(
                from_normal, tri_from.vertices[0], tri_onto);

            if (onto_region.type() == TriangleSubsetType::None) {
              continue;
            }

            LightingPair lighting_pair{
                .idxs = {j, k},
                .is_neg_faces = {onto_is_neg_face, from_is_neg_face},
                .regions = {onto_region, from_region},

                // will be set later
                .infos = {},
                .overall_partially_shadowed = {},
                .overall_totally_shadowed = {},
                .overall_from_each_point = {},
            };

            bool is_first = true;
            for (unsigned l = 0; l < tris.size(); ++l) {
              if (l == j || l == k) {
                continue;
              }

              const auto &tri_blocker = tris[l];

              const auto blocker_normal = *tri_blocker.normal();

              bool next_blocker = false;
              for (const auto &[normal, vert] : {
                       std::tuple{onto_normal, tri_onto.vertices[0]},
                       std::tuple{from_normal, tri_from.vertices[0]},
                   }) {
                for (double sign : {1., -1.}) {
                  const auto normal_mul = normal * sign;
                  if ((blocker_normal - normal_mul).norm() < 1e-10 &&
                      std::abs(blocker_normal.dot(tri_blocker.vertices[0] -
                                                  vert)) < 1e-10) {
                    // coplanar!
                    next_blocker = true;
                    goto next_blocker;
                  }
                }
              }
            next_blocker:
              if (next_blocker) {
                continue;
              }

              const auto blocker_region = triangle_subset_intersection(
                  clip_by_plane_point(onto_normal, tri_onto.vertices[0],
                                      tri_blocker),
                  clip_by_plane_point(from_normal, tri_from.vertices[0],
                                      tri_blocker));

              if (blocker_region.type() == TriangleSubsetType::None) {
                continue;
              }

              if (!possibly_shadowed({&tri_onto, &tri_from}, tri_blocker,
                                     {onto_is_neg_face, from_is_neg_face})) {
                continue;
              }

              BlockerInfo info{
                  .idx = l,
                  .partially_shadowed_infos{
                      partially_shadowed(tri_from, from_region, tri_blocker,
                                         blocker_region, tri_onto, onto_region,
                                         onto_is_neg_face),
                      partially_shadowed(tri_onto, onto_region, tri_blocker,
                                         blocker_region, tri_from, from_region,
                                         from_is_neg_face),
                  },
                  .totally_shadowed_infos{
                      totally_shadowed(tri_from, from_region, tri_blocker,
                                       blocker_region, tri_onto, onto_region),
                      totally_shadowed(tri_onto, onto_region, tri_blocker,
                                       blocker_region, tri_from, from_region),
                  },
              };

              // other than floating point issues, the first should imply the
              // second...
              if (info.partially_shadowed_infos[0].partially_shadowed.type() ==
                      TriangleSubsetType::None ||
                  info.partially_shadowed_infos[1].partially_shadowed.type() ==
                      TriangleSubsetType::None) {
                continue;
              }

              auto to_multi = [](const VectorT<TriangleSubset> &subsets) {
                VectorT<TriangleMultiSubset> out(subsets.size());
                std::transform(subsets.begin(), subsets.end(), out.begin(),
                               [&](const TriangleSubset &sub) {
                                 return subset_to_multi(sub);
                               });
                return out;
              };

              std::array as_multi{
                  to_multi(info.totally_shadowed_infos[0].from_each_point),
                  to_multi(info.totally_shadowed_infos[1].from_each_point),
              };

              std::array as_multi_partially_shadowed{
                  subset_to_multi(
                      info.partially_shadowed_infos[0].partially_shadowed),
                  subset_to_multi(
                      info.partially_shadowed_infos[1].partially_shadowed),
              };

              if (is_first) {
                lighting_pair.overall_from_each_point = as_multi;
                lighting_pair.overall_partially_shadowed =
                    as_multi_partially_shadowed;
              } else {
                // TODO: this doesn't make sense!
                for (unsigned item = 0;
                     item < info.totally_shadowed_infos.size(); ++item) {
                  debug_assert(
                      as_multi[item].size() ==
                      lighting_pair.overall_from_each_point[item].size());
                  for (unsigned point_idx = 0;
                       point_idx <
                       info.totally_shadowed_infos[item].from_each_point.size();
                       ++point_idx) {
                    lighting_pair.overall_from_each_point[item][point_idx] =
                        triangle_subset_union(
                            lighting_pair
                                .overall_from_each_point[item][point_idx],
                            as_multi[item][point_idx]);
                  }
                }
                for (unsigned item = 0;
                     item < info.totally_shadowed_infos.size(); ++item) {

                  lighting_pair.overall_partially_shadowed[item] =
                      triangle_subset_union(
                          lighting_pair.overall_partially_shadowed[item],
                          as_multi_partially_shadowed[item]);
                }
              }
              is_first = false;

              lighting_pair.infos.push_back(info);
            }

            for (unsigned item = 0;
                 item < lighting_pair.overall_totally_shadowed.size(); ++item) {
              auto &shadowed = lighting_pair.overall_totally_shadowed[item];
              if (lighting_pair.overall_from_each_point[item].empty()) {
                shadowed = {tag_v<TriangleSubsetType::None>, {}};
              } else {
                shadowed = {tag_v<TriangleSubsetType::All>, {}};
                for (const auto &point_subset :
                     lighting_pair.overall_from_each_point[item]) {
                  shadowed = triangle_multi_subset_intersection(shadowed,
                                                                point_subset);
                }
              }

              if (shadowed.type() == TriangleSubsetType::All) {
                goto next_face;
              }

              const auto &region = item == 0 ? onto_region : from_region;

              // check if clipped region is totally shadowed also
              // (if end_region[1]->type() is All, then this case isn't
              // relavent, the clipped is the whole thing...)
              // TODO: dedup with single_triangle
              if (shadowed.type() == TriangleSubsetType::Some &&
                  region.type() == TriangleSubsetType::Some) {

                const auto &region_poly =
                    region.get(tag_v<TriangleSubsetType::Some>);
                const auto &intersected_poly =
                    shadowed.get(tag_v<TriangleSubsetType::Some>);
                double region_area = boost::geometry::area(region_poly);
                double intersected_area =
                    boost::geometry::area(intersected_poly);
                if (std::abs(region_area - intersected_area) < 1e-10) {
                  goto next_face;
                }
              }
            }

            lighting_pairs[i].push_back(lighting_pair);

          next_face : {}
          }
        }
      }
    }
  }

  dbg(lighting_pairs.size());
  dbg(lighting_pairs[0].size());

  for (unsigned i = 0; i < lighting_pairs[0].size(); ++i) {
    const auto &p = lighting_pairs[0][i];
    if (p.infos.size() < 5) {
      continue;
    }
    const auto &tris = all_triangles[0];
    dbg(p.idxs);
    std::cout << "l_triangle =";
    print_triangle(tris[p.idxs[0]].template cast<float>());
    std::cout << "r_triangle =";
    print_triangle(tris[p.idxs[1]].template cast<float>());
    Eigen::Vector3d normal_0 =
        (*tris[p.idxs[0]].normal()) * (p.is_neg_faces[0] ? -1. : 1.);
    Eigen::Vector3d normal_1 =
        (*tris[p.idxs[1]].normal()) * (p.is_neg_faces[1] ? -1. : 1.);

    dbg(normal_0);
    dbg(normal_1);
    dbg(tris[p.idxs[0]].centroid());
    dbg(tris[p.idxs[1]].centroid());

    std::cout << "partially_shadowed_on_l =";
    print_multi_region(p.overall_partially_shadowed[0]);
    std::cout << "partially_shadowed_on_r =";
    print_multi_region(p.overall_partially_shadowed[1]);

    std::cout << "totally_shadowed_on_l =";
    print_multi_region(p.overall_totally_shadowed[0]);
    std::cout << "totally_shadowed_on_r =";
    print_multi_region(p.overall_totally_shadowed[1]);

    for (unsigned i = 0; i < p.infos.size(); ++i) {
      const auto &info = p.infos[i];
      std::cout << "blocker_triangle_" << i << " =";
      print_triangle(tris[info.idx].template cast<float>());
    }

    std::cout << "\n\n\n";
  }
  // TODO: compute all triangle features
}
} // namespace full_scene
} // namespace generate_data
