#include "generate_data/gen_data.h"

#include "generate_data/amend_config.h"
#include "generate_data/baryocentric_coords.h"
#include "generate_data/baryocentric_to_ray.h"
#include "generate_data/clip_by_plane.h"
#include "generate_data/constants.h"
#include "generate_data/generate_scene.h"
#include "generate_data/generate_scene_triangles.h"
#include "generate_data/get_points_from_subset.h"
#include "generate_data/normalize_scene_triangles.h"
#include "generate_data/region_setter.h"
#include "generate_data/remap_large.h"
#include "generate_data/shadowed.h"
#include "generate_data/to_tensor.h"
#include "generate_data/torch_utils.h"
#include "generate_data/triangle.h"
#include "generate_data/triangle_subset_intersection.h"
#include "generate_data/value_adder.h"
#include "integrate/sample_triangle.h"
#include "lib/projection.h"
#include "render/renderer.h"
#include "rng/uniform/uniform.h"

#include <ATen/ATen.h>
#include <boost/multi_array.hpp>

#include <iostream>
#include <omp.h>
#include <tuple>
#include <vector>

// TODO: consider breaking up more of this
namespace generate_data {
static VectorT<render::Renderer> renderers;

template <bool is_image>
using Out = std::conditional_t<is_image, ImageData, StandardData>;

// TODO: consider fixing extra copies (if needed).
// Could really return gpu tensor and output directly to tensor.
template <bool is_image>
Out<is_image> gen_data_impl(int n_scenes, int n_samples_per_scene_or_dim,
                            int n_samples, unsigned base_seed) {
  using namespace generate_data;

  debug_assert(boost::geometry::is_valid(full_triangle));
  debug_assert(std::abs(boost::geometry::area(full_triangle) - 0.5) < 1e-12);

  auto vals = [&]() {
    if constexpr (is_image) {
      unsigned dim = n_samples_per_scene_or_dim;
      return baryocentric_coords(dim, dim);
    } else {
      return std::tuple{0, 0};
    }
  }();

  auto baryocentric_indexes = std::get<0>(vals);
  auto baryocentric_grid_values = std::get<1>(vals);

  int n_samples_per_scene = [&]() {
    if constexpr (is_image) {
      return baryocentric_grid_values.size();
    } else {
      return n_samples_per_scene_or_dim;
    }
  }();

  boost::multi_array<float, 2> overall_scene_features{
      std::array{n_scenes, constants.n_scene_values}};
  boost::multi_array<float, 3> triangle_features{
      std::array{n_scenes, constants.n_tris, constants.n_tri_values}};
  boost::multi_array<float, 3> baryocentric_coords{std::array{
      n_scenes, n_samples_per_scene, constants.n_coords_feature_values}};

  constexpr unsigned n_prior_dims = 1;
  std::array<TorchIdxT, n_prior_dims> prior_dims{n_scenes};

  using SinglePolyRegionSetter = RegionSetter<n_prior_dims>;

  auto get_vec = [&](unsigned n) {
    VectorT<SinglePolyRegionSetter> out;
    for (unsigned i = 0; i < n; ++i) {
      out.push_back({prior_dims});
    }
    return out;
  };

  auto clipped_setters = get_vec(constants.n_tris);
  // TODO: do we also want non intersected with clipped variations?
  auto totally_shadowed_setters = get_vec(constants.n_shadowable_tris);
  auto partially_shadowed_setters = get_vec(constants.n_shadowable_tris);
  auto centroid_shadowed_setters = get_vec(constants.n_shadowable_tris);
  VectorT<VectorT<VectorT<PartiallyShadowedInfo::RayItem>>> ray_items(
      constants.n_ray_items);
  for (auto &r : ray_items) {
    r.resize(n_scenes);
  }
  VectorT<boost::multi_array<TorchIdxT, 1>> ray_item_counts(
      constants.n_ray_items);
  for (auto &r : ray_item_counts) {
    r.resize(prior_dims);
  }

  renderers.resize(omp_get_max_threads());

  render::Settings settings;
  amend_config(settings);

  std::array values_dim{n_scenes, n_samples_per_scene, constants.n_rgb_dims};
  boost::multi_array<float, 3> values{values_dim};

  boost::multi_array<TorchIdxT, 2> image_indexes;
  if constexpr (is_image) {
    image_indexes.resize(std::array{unsigned(baryocentric_indexes.size()), 2u});
    for (int i = 0; i < int(baryocentric_indexes.size()); ++i) {
      auto [x, y] = baryocentric_indexes[i];
      image_indexes[i][0] = TorchIdxT(x);
      image_indexes[i][1] = TorchIdxT(y);
    }
  }

  // could use more than cpu cores really - goal is async...
#pragma omp parallel for schedule(dynamic, 8) if (!debug_build)
  for (int i = 0; i < n_scenes; ++i) {
    UniformState rng_state(base_seed + i);
  restart:
    auto tris = generate_scene_triangles(rng_state);
    auto new_tris = normalize_scene_triangles(tris);

    // clip by triangle_onto plane
    auto light_region =
        clip_by_plane(Eigen::Vector3d::UnitZ(), 0.,
                      new_tris.triangle_light.template cast<double>());
    auto blocking_region_initial =
        clip_by_plane(Eigen::Vector3d::UnitZ(), 0.,
                      new_tris.triangle_blocking.template cast<double>());

    // clip by triangle_light plane
    auto light_normal =
        new_tris.triangle_light.template cast<double>().normal_raw();
    auto light_point =
        new_tris.triangle_light.template cast<double>().vertices[0];
    auto onto_region =
        clip_by_plane_point(light_normal, light_point,
                            new_tris.triangle_onto.template cast<double>());

    auto blocking_region = triangle_subset_intersection(
        blocking_region_initial,
        clip_by_plane_point(
            light_normal, light_point,
            new_tris.triangle_blocking.template cast<double>()));

    if (light_region.type() == TriangleSubsetType::None ||
        blocking_region.type() == TriangleSubsetType::None ||
        onto_region.type() == TriangleSubsetType::None) {
      goto restart;
    }

    std::array tri_pointers{
        &new_tris.triangle_onto,
        &new_tris.triangle_blocking,
        &new_tris.triangle_light,
    };
    std::array region_pointers{
        &onto_region,
        &blocking_region,
        &light_region,
    };
    std::array<Eigen::Vector3d, 3> region_centroids;
    std::array<Eigen::Vector3d, 3> centroids;
    for (int tri_idx = 0; tri_idx < int(tri_pointers.size()); ++tri_idx) {
      const auto &tri = *tri_pointers[tri_idx];
      auto from_points = get_points_from_subset(tri, *region_pointers[tri_idx]);
      region_centroids[tri_idx] = Eigen::Vector3d::Zero();
      for (const auto &p : from_points) {
        region_centroids[tri_idx] += p;
      }
      region_centroids[tri_idx] /= from_points.size();
      centroids[tri_idx] = tri.centroid();
    }

    for (unsigned onto_idx : {0, 2}) {
      std::array<const intersect::TriangleGen<double> *, 2> end_tris{
          &new_tris.triangle_light, &new_tris.triangle_onto};
      std::array<const TriangleSubset *, 2> end_region{&light_region,
                                                       &onto_region};
      unsigned feature_idx = 0;
      bool is_light = onto_idx == 2;
      if (is_light) {
        feature_idx = 1;
        std::swap(end_tris[0], end_tris[1]);
        std::swap(end_region[0], end_region[1]);
      }
      auto shadow_info = totally_shadowed(*end_tris[0], *end_region[0],
                                          new_tris.triangle_blocking,
                                          blocking_region, *end_tris[1]);
      if (shadow_info.totally_shadowed.type() == TriangleSubsetType::All) {
        // assert that we aren't on the second iter (should have already
        // restarted)
        // always_assert(onto_idx == 0);
        // numerical issues cause this to occur (very rarely...)

        // totally blocked
        goto restart;
      }

      auto totally_shadowed_intersected = triangle_subset_intersection(
          *end_region[1], shadow_info.totally_shadowed);

      // check if clipped region is totally shadowed also
      if (totally_shadowed_intersected.type() == TriangleSubsetType::Some &&
          end_region[1]->type() == TriangleSubsetType::Some) {
        const auto &end_region_poly =
            end_region[1]->get(tag_v<TriangleSubsetType::Some>);
        const auto &intersected_poly =
            totally_shadowed_intersected.get(tag_v<TriangleSubsetType::Some>);
        double region_area = boost::geometry::area(end_region_poly);
        double intersected_area = boost::geometry::area(intersected_poly);
        if (std::abs(region_area - intersected_area) < 1e-10) {
          // assert that we aren't on the second iter (should have already
          // restarted)
          // always_assert(onto_idx == 0);
          // numerical issues cause this to occur (very rarely...)

          // totally blocked
          goto restart;
        }
      }

      auto partially_shadowed_info = partially_shadowed(
          *end_tris[0], *end_region[0], new_tris.triangle_blocking,
          blocking_region, *end_tris[1]);
      auto partially_shadowed_intersected = triangle_subset_intersection(
          partially_shadowed_info.partially_shadowed, *end_region[1]);

      if (partially_shadowed_intersected.type() == TriangleSubsetType::None) {
        // assert that we aren't on the second iter (should have already
        // restarted)
        // always_assert(onto_idx == 0);
        // numerical issues cause this to occur (very rarely...)

        // not blocked at all
        goto restart;
      }

      std::array<TorchIdxT, n_prior_dims> prior_idxs{i};

      totally_shadowed_setters[feature_idx].set_region(
          prior_idxs, totally_shadowed_intersected, *end_tris[1]);

      partially_shadowed_setters[feature_idx].set_region(
          prior_idxs, partially_shadowed_intersected, *end_tris[1]);
      ray_items[feature_idx][i] = partially_shadowed_info.ray_items;
      ray_item_counts[feature_idx][i] =
          partially_shadowed_info.ray_items.size();

      auto centroid_shadow = shadowed_from_point(
          region_centroids[onto_idx],
          get_points_from_subset(new_tris.triangle_blocking, blocking_region),
          *end_tris[1]);
      centroid_shadowed_setters[feature_idx].set_region(
          prior_idxs,
          triangle_subset_intersection(centroid_shadow, *end_region[1]),
          *end_tris[1]);
    }

    auto scene_adder = make_value_adder(
        [&](float v, int idx) { overall_scene_features[i][idx] = v; });

    // TODO: precompute where possible
    for (int tri_idx = 0; tri_idx < int(tri_pointers.size()); ++tri_idx) {
      const auto &tri = *tri_pointers[tri_idx];

      // add tri scene values

      // TODO: logs or other functions of these values?
      auto tri_adder = make_value_adder([&](float v, int value_idx) {
        triangle_features[i][tri_idx][value_idx] = v;
      });

      for (const auto &point : tri.vertices) {
        tri_adder.add_values(point);
      }
      const auto normal_scaled = tri.normal_scaled_by_area();
      tri_adder.add_values(normal_scaled);
      // no need for 'centroids', can be trivially computed by net
      tri_adder.add_values(region_centroids[tri_idx]);
      const auto normal = normal_scaled.normalized().eval();
      tri_adder.add_values(normal);
      tri_adder.add_value(tri.area());
      double region_area = clipped_setters[tri_idx].set_region(
          {i}, *region_pointers[tri_idx], tri);
      tri_adder.add_value(region_area);
      debug_assert(tri_adder.idx == constants.n_tri_values);

      // add tri interaction general scene values
      for (int other_tri_idx = tri_idx + 1;
           other_tri_idx < int(tri_pointers.size()); ++other_tri_idx) {
        const auto &other_tri = *tri_pointers[other_tri_idx];
        const auto other_normal_scaled = other_tri.normal_scaled_by_area();
        scene_adder.add_value(other_normal_scaled.dot(normal_scaled));
        const auto other_normal = other_normal_scaled.normalized().eval();
        const double normal_dot = other_normal.dot(normal);
        scene_adder.add_value(normal_dot);
        scene_adder.add_value(std::acos(normal_dot));

        for (const auto &centroid_arr : {centroids, region_centroids}) {
          Eigen::Vector3d centroid_vec_raw =
              centroid_arr[other_tri_idx] - centroid_arr[tri_idx];
          double centroid_dist = centroid_vec_raw.norm();
          scene_adder.add_value(centroid_dist);
          Eigen::Vector3d centroid_vec = centroid_vec_raw.normalized();
          scene_adder.add_values(centroid_vec);
          double normal_centroid_dot = normal.dot(centroid_vec);
          scene_adder.add_value(normal_centroid_dot);
          scene_adder.add_value(std::acos(normal_centroid_dot));
          double other_normal_centroid_dot = other_normal.dot(-centroid_vec);
          scene_adder.add_value(other_normal_centroid_dot);
          scene_adder.add_value(std::acos(other_normal_centroid_dot));
        }
      }
    }

    debug_assert(scene_adder.idx == constants.n_scene_values);

    VectorT<intersect::Ray> rays(n_samples_per_scene);

    auto dir_towards = -tris.triangle_onto.template cast<float>().normal();
    for (int j = 0; j < n_samples_per_scene; ++j) {
      auto [s, t] = [&]() {
        if constexpr (is_image) {
          return baryocentric_grid_values[j];
        } else {
          return integrate::uniform_baryocentric(rng_state);
        }
      }();
      rays[j] = baryocentric_to_ray(
          s, t, tris.triangle_onto.template cast<float>(), dir_towards);
      auto baryo_adder = make_value_adder(
          [&](float v, int idx) { baryocentric_coords[i][j][idx] = v; });

      baryo_adder.add_value(s);
      baryo_adder.add_value(t);
      const auto point = tris.triangle_onto.baryo_to_point({s, t});
      baryo_adder.add_values(point);
    }

    VectorT<FloatRGB> values_vec(n_samples_per_scene);

    if (!rays.empty()) {
      auto scene = generate_scene(tris);
      always_assert(size_t(omp_get_thread_num()) < renderers.size());
      renderers[omp_get_thread_num()].render(
          ExecutionModel::GPU,
          {tag_v<render::SampleSpecType::InitialRays>, rays},
          {tag_v<render::OutputType::FloatRGB>, values_vec}, scene, n_samples,
          settings, false);
      for (int j = 0; j < n_samples_per_scene; ++j) {
        for (int k = 0; k < 3; ++k) {
          values[i][j][k] = values_vec[j][k];
        }
      }
    }
  }

  std::vector<PolygonInputForTri> polygon_inputs;
  for (int tri_idx = 0; tri_idx < constants.n_tris; ++tri_idx) {
    polygon_inputs.push_back(
        {.polygon_feature = clipped_setters[tri_idx].as_poly_input(),
         .tri_idx = tri_idx});
  }
  for (int onto_idx : {0, 2}) {
    unsigned feature_idx = onto_idx == 0 ? 0 : 1;

    auto add = [&](VectorT<SinglePolyRegionSetter> &setters) {
      polygon_inputs.push_back(
          {.polygon_feature = setters[feature_idx].as_poly_input(),
           .tri_idx = onto_idx});
    };

    add(totally_shadowed_setters);
    add(partially_shadowed_setters);
    add(centroid_shadowed_setters);
  }
  debug_assert(int(polygon_inputs.size()) == constants.n_polys);

  std::vector<RayInput> ray_inputs(ray_items.size());
  for (unsigned i = 0; i < ray_items.size(); ++i) {
    auto counts = to_tensor(ray_item_counts[i]);
    TorchIdxT total = counts.sum().item().template to<TorchIdxT>();
    boost::multi_array<float, 2> values{
        std::array{total, TorchIdxT(constants.n_ray_item_values)}};
    boost::multi_array<bool, 1> is_ray{std::array{total}};
    unsigned running_idx = 0;
    for (const auto &vec : ray_items[i]) {
      for (const auto &item : vec) {
        auto adder = make_value_adder(
            [&](float v, int idx) { values[running_idx][idx] = v; });
        adder.add_values(baryo_to_eigen(item.baryo_origin));
        adder.add_values(baryo_to_eigen(item.baryo_endpoint));
        adder.add_values(item.origin);
        adder.add_values(item.endpoint);
        item.result.visit_tagged([&](auto tag, const auto &v) {
          if constexpr (tag == RayItemResultType::Ray) {
            debug_assert(std::abs(v.norm() - 1.f) < 1e-12);
            adder.add_values(v);
            adder.add_value(0.);
            adder.add_remap_value(0., 1e4);
            adder.add_remap_value(0., 1e4);
            adder.add_remap_value(0., 1e4);
          } else {
            static_assert(tag == RayItemResultType::Intersection);
            adder.add_values(v.normalized().eval());
            adder.add_value(std::atan2(v.y(), v.x()));
            // values can get VERY large
            const double norm = v.norm();
            adder.add_remap_value(norm);
            adder.add_remap_value(v.x());
            adder.add_remap_value(v.y());
          }
        });
        is_ray[running_idx] = item.result.type() == RayItemResultType::Ray;
        debug_assert(adder.idx == constants.n_ray_item_values);
        ++running_idx;
      }
    }
    always_assert(running_idx == total);

    TorchIdxT min_count = counts.min().item().template to<TorchIdxT>();
    always_assert(min_count > 0);

    ray_inputs[i] = {
        .values = to_tensor(values),
        .counts = counts,
        .prefix_sum_counts =
            at::cat({at::tensor({TorchIdxT(0)}), counts.flatten().cumsum(0)}),
        .is_ray = to_tensor(is_ray),
    };
  }

  StandardData out{
      .inputs =
          {
              .overall_scene_features = to_tensor(overall_scene_features),
              .triangle_features = to_tensor(triangle_features),
              .polygon_inputs = polygon_inputs,
              .ray_inputs = ray_inputs,
              .baryocentric_coords = to_tensor(baryocentric_coords),
          },
      .values = to_tensor(values),

  };

  if constexpr (is_image) {
    return {.standard = out, .image_indexes = to_tensor(image_indexes)};
  } else {
    return out;
  }
}

StandardData gen_data(int n_scenes, int n_samples_per_scene, int n_samples,
                      unsigned base_seed) {
  return gen_data_impl<false>(n_scenes, n_samples_per_scene, n_samples,
                              base_seed);
}

ImageData gen_data_for_image(int n_scenes, int dim, int n_samples,
                             unsigned base_seed) {
  return gen_data_impl<true>(n_scenes, dim, n_samples, base_seed);
}

void deinit_renderers() { renderers.resize(0); }
} // namespace generate_data
