#include "generate_data/gen_data.h"
#include "generate_data/amend_config.h"
#include "generate_data/baryocentric_coords.h"
#include "generate_data/baryocentric_to_ray.h"
#include "generate_data/clip_by_plane.h"
#include "generate_data/generate_scene.h"
#include "generate_data/generate_scene_triangles.h"
#include "generate_data/get_points_from_subset.h"
#include "generate_data/normalize_scene_triangles.h"
#include "generate_data/shadowed.h"
#include "generate_data/triangle.h"
#include "generate_data/triangle_subset_intersection.h"
#include "integrate/sample_triangle.h"
#include "kernel/atomic.h"
#include "lib/array_vec.h"
#include "lib/async_for.h"
#include "lib/projection.h"
#include "render/renderer.h"
#include "rng/uniform/uniform.h"

#include <boost/hana/ext/std/array.hpp>
#include <omp.h>
#include <torch/extension.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "dbg.h"
#include "lib/info/print_triangle.h"

namespace generate_data {
static VectorT<render::Renderer> renderers;

template <bool is_image>
using Out = std::conditional_t<
    is_image,
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>,
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>;

template <typename F> struct ValueAdder {
  F base_add;
  int idx = 0;

  void add_value(float v) { base_add(v, idx++); }

  template <typename T> void add_values(const T &vals) {
    for (const auto v : vals) {
      add_value(v);
    }
  }
};

struct PolyPoint {
  BaryoPoint baryo;
  double angle;
  Eigen::Vector3d point;
};

constexpr int n_tris = 3;
constexpr int n_scene_values = (3 + 2 * 11) * n_tris;
constexpr int n_dims = 3;
constexpr int n_tri_values = n_tris * n_dims + 4 * n_dims + 2;
constexpr int n_baryo_dims = 2;
constexpr int n_poly_point_values = 3 * n_baryo_dims + n_dims + 4;
constexpr int n_rgb_dims = 3;
constexpr int n_shadowable_tris = 2;   // onto and light
constexpr int n_poly_feature_dims = 2; // area and properly scaled area
constexpr double dotted_thresh = 1. - 1e-7;

using TorchIdxT = int64_t;
const static auto long_tensor_type =
    torch::TensorOptions(caffe2::TypeMeta::Make<TorchIdxT>());

using kernel::atomic::detail::CopyableAtomic;

template <typename T>
void update_maximum(CopyableAtomic<T> &maximum_value, const T &value) noexcept {
  T prev_value = maximum_value;
  while (prev_value < value &&
         !maximum_value.compare_exchange_weak(prev_value, value)) {
  }
}

// TODO: consider more efficient representation later
template <unsigned n_prior_dims> struct RegionSetter {
  static constexpr TorchIdxT magic_value_none = 2l << 30;

  RegionSetter() = default;

  RegionSetter(std::array<TorchIdxT, n_prior_dims> prior_dims, int max_n_points)
      : max_n_points(max_n_points) {
    std::array<TorchIdxT, n_prior_dims + 2> region_dims;
    std::copy(prior_dims.begin(), prior_dims.end(), region_dims.begin());
    region_dims[n_prior_dims] = max_n_points;
    region_dims[n_prior_dims + 1] = n_poly_point_values;
    regions = torch::empty(region_dims);
    std::array<TorchIdxT, n_prior_dims + 1> feature_dims;
    std::copy(prior_dims.begin(), prior_dims.end(), feature_dims.begin());
    feature_dims[n_prior_dims] = n_poly_feature_dims;
    overall_features = torch::empty(feature_dims);
    counts = torch::empty(prior_dims, long_tensor_type);
  }

  double set_region(std::array<TorchIdxT, n_prior_dims> prior_idxs_in,
                    const TriangleSubset &region, const Triangle &tri) {
    return region.visit_tagged([&](auto tag, const auto &value) {
      auto prior_idxs = boost::hana::unpack(
          prior_idxs_in,
          [&](auto... idxs)
              -> std::array<torch::indexing::TensorIndex, n_prior_dims> {
            return {idxs...};
          });
      auto feature_idxs = boost::hana::unpack(
          prior_idxs_in,
          [&](auto... idxs)
              -> std::array<torch::indexing::TensorIndex, n_prior_dims + 1> {
            return {idxs..., 0};
          });

      auto add_feature = [&](int idx, float v) {
        feature_idxs[n_prior_dims] = idx;
        overall_features.index_put_(feature_idxs, v);
      };

      double area_3d = 0.;
      if constexpr (tag == TriangleSubsetType::None) {
        counts.index_put_(prior_idxs, magic_value_none);
        add_feature(0, 0.);
        add_feature(1, 0.);
      } else {
        const auto pb = get_points_from_subset_with_baryo(tri, region);
        always_assert(pb.points.size() == pb.baryo.size());
        always_assert(pb.points.size() <= max_n_points);
        update_maximum(actual_max_n_points, unsigned(pb.points.size()));

        counts.index_put_(prior_idxs, TorchIdxT(pb.baryo.size()));

        // get areas in baryo and in 3d
        const TriPolygon *baryo_poly;
        if constexpr (tag == TriangleSubsetType::All) {
          baryo_poly = &full_triangle;
        } else {
          static_assert(tag == TriangleSubsetType::Some);

          baryo_poly = &value;
        }
        auto rot = find_rotate_vector_to_vector(
            tri.normal(), UnitVectorGen<double>::new_normalize({0., 0., 1.}));
        Eigen::Vector2d vec0 =
            (rot * (tri.vertices[1] - tri.vertices[0])).head(2);
        Eigen::Vector2d vec1 =
            (rot * (tri.vertices[2] - tri.vertices[0])).head(2);
        TriPolygon poly_distorted;
        auto add_point_to_distorted = [&](const BaryoPoint &p) {
          Eigen::Vector2d new_p = p.x() * vec0 + p.y() * vec1;
          boost::geometry::append(poly_distorted,
                                  BaryoPoint{new_p.x(), new_p.y()});
        };
        for (const auto &p : pb.baryo) {
          add_point_to_distorted(p);
        }
        add_point_to_distorted(pb.baryo[0]);

        if (!boost::geometry::is_valid(poly_distorted)) {
          std::reverse(poly_distorted.outer().begin(),
                       poly_distorted.outer().end());
        }

        debug_assert(boost::geometry::is_valid(*baryo_poly));
        debug_assert(boost::geometry::is_valid(poly_distorted));

        add_feature(0, boost::geometry::area(*baryo_poly));
        area_3d = boost::geometry::area(poly_distorted);
        add_feature(1, area_3d);

        auto idxs = boost::hana::unpack(
            prior_idxs_in,
            [&](auto... idxs)
                -> std::array<torch::indexing::TensorIndex, n_prior_dims + 2> {
              return {idxs..., 0, 0};
            });
        std::copy(prior_idxs.begin(), prior_idxs.end(), idxs.begin());

        for (unsigned i = 0; i < pb.baryo.size(); ++i) {
          idxs[n_prior_dims] = int(i);

          auto value_adder = make_value_adder([&](float v, int idx) {
            idxs[n_prior_dims + 1] = idx;
            regions.index_put_(idxs, v);
          });

          unsigned i_prev = (i == 0) ? pb.baryo.size() - 1 : i - 1;
          unsigned i_next = (i + 1) % pb.baryo.size();

          const auto baryo_prev = baryo_to_eigen(pb.baryo[i_prev]);
          const auto baryo = baryo_to_eigen(pb.baryo[i]);
          const auto baryo_next = baryo_to_eigen(pb.baryo[i_next]);

          Eigen::Vector2d edge_l = baryo_prev - baryo;
          double norm_l = edge_l.norm();
          debug_assert(norm_l >= 1e-10);
          value_adder.add_value(norm_l);
          Eigen::Vector2d normalized_l = edge_l / norm_l;
          value_adder.add_values(normalized_l);

          Eigen::Vector2d edge_r = baryo_next - baryo;
          double norm_r = edge_r.norm();
          debug_assert(norm_r >= 1e-10);
          value_adder.add_value(norm_r);
          Eigen::Vector2d normalized_r = edge_r / norm_r;
          value_adder.add_values(normalized_r);

          double dotted = normalized_l.dot(normalized_r);
          debug_assert(dotted < dotted_thresh);
          value_adder.add_value(dotted);
          value_adder.add_value(std::acos(dotted));
          value_adder.add_values(baryo);
          value_adder.add_values(pb.points[i]);

          debug_assert(value_adder.idx == n_poly_point_values);
        }
      }

      return area_3d;
    });
  }

  unsigned max_n_points;
  CopyableAtomic<unsigned> actual_max_n_points = 0;
  torch::Tensor regions;
  torch::Tensor overall_features;
  torch::Tensor counts;
};

template <unsigned n_prior_dims> struct MultiRegionSetter {
  MultiRegionSetter(std::array<TorchIdxT, n_prior_dims> prior_dims,
                    int max_n_points, int max_n_polys)
      : max_n_polys(max_n_polys) {
    std::array<TorchIdxT, n_prior_dims + 1> all_prior_dims;
    std::copy(prior_dims.begin(), prior_dims.end(), all_prior_dims.begin());
    all_prior_dims[n_prior_dims] = max_n_polys;
    setter = {all_prior_dims, max_n_points};
    poly_counts = torch::empty(prior_dims, long_tensor_type);
  }

  void add_regions(std::array<TorchIdxT, n_prior_dims> prior_idxs_in,
                   VectorT<TriangleSubset> regions, const Triangle &tri) {
    if (regions.empty()) {
      regions.push_back({tag_v<TriangleSubsetType::None>, {}});
    }

    always_assert(regions.size() <= max_n_polys);

    update_maximum(actual_max_n_polys, unsigned(regions.size()));

    auto prior_idxs = boost::hana::unpack(
        prior_idxs_in,
        [&](auto... idxs)
            -> std::array<torch::indexing::TensorIndex, n_prior_dims> {
          return {idxs...};
        });

    poly_counts.index_put_(prior_idxs, int(regions.size()));
    std::array<TorchIdxT, n_prior_dims + 1> all_prior_idxs;
    std::copy(prior_idxs_in.begin(), prior_idxs_in.end(),
              all_prior_idxs.begin());
    for (int i = 0; i < int(regions.size()); ++i) {
      all_prior_idxs[n_prior_dims] = i;
      setter.set_region(all_prior_idxs, regions[i], tri);
    }
  }

  unsigned max_n_polys;
  CopyableAtomic<unsigned> actual_max_n_polys = 0;
  torch::Tensor poly_counts;
  RegionSetter<n_prior_dims + 1> setter;
};

template <typename F> auto make_value_adder(F v) { return ValueAdder<F>{v}; }

// scenes, coords, values (and potentially indexes for image)
// TODO: consider fixing extra copies (if needed).
// Could really return gpu tensor and output directly to tensor.
template <bool is_image>
Out<is_image> gen_data_impl(int n_scenes, int n_samples_per_scene_or_dim,
                            int n_samples, unsigned base_seed) {
  using namespace generate_data;
  using namespace torch::indexing;

  debug_assert(boost::geometry::is_valid(full_triangle));
  debug_assert(std::abs(boost::geometry::area(full_triangle) - 0.5) < 1e-12);

  renderers.resize(omp_get_max_threads());

  render::Settings settings;
  amend_config(settings);

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

  constexpr unsigned n_prior_dims = 2;
  std::array<TorchIdxT, n_prior_dims> basic_prior_dims{n_scenes, n_tris};

  using SinglePolyRegionSetter = RegionSetter<n_prior_dims>;
  using MultiPolyRegionSetter = MultiRegionSetter<n_prior_dims>;

  // each clip adds at most 1 point and we clip at most twice
  // (actually we only clip once for light and onto, so the max for
  // those is actually only 4)
  int max_n_clipped_points = 5;
  SinglePolyRegionSetter clipped_setter{basic_prior_dims, max_n_clipped_points};

  // TODO: check these point counts (here and below)

  // could intersect leading to large number of points
  int max_n_clip_removed_points = 6;
  int max_n_clip_removed_polys = 2;
  MultiPolyRegionSetter clip_removed_setter{
      basic_prior_dims, max_n_clip_removed_points, max_n_clip_removed_polys};

  // first is onto and second is light
  std::array<TorchIdxT, n_prior_dims> shadowed_prior_dims{n_scenes,
                                                          n_shadowable_tris};

  // TODO: do we also want non intersected with clipped variations?
  int max_n_totally_shadowed_points = 9;
  SinglePolyRegionSetter totally_shadowed_setter{shadowed_prior_dims,
                                                 max_n_totally_shadowed_points};

  // clipped - totally shadowed
  int max_n_partially_visible_points = 15;
  int max_n_partially_visible_polys = 4;
  MultiPolyRegionSetter partially_visible_setter{shadowed_prior_dims,
                                                 max_n_partially_visible_points,
                                                 max_n_partially_visible_polys};

  int max_n_partially_shadowed_points = 15;
  SinglePolyRegionSetter partially_shadowed_setter{
      shadowed_prior_dims, max_n_partially_shadowed_points};

  // clipped - partially shadowed
  int max_n_totally_visible_points = 15;
  int max_n_totally_visible_polys = 4;
  MultiPolyRegionSetter totally_visible_setter{shadowed_prior_dims,
                                               max_n_totally_visible_points,
                                               max_n_totally_visible_polys};

  // TODO: do we also want non intersected with clipped variations?
  int max_n_centroid_shadowed_points = 9;
  SinglePolyRegionSetter centroid_shadowed_setter{
      shadowed_prior_dims, max_n_centroid_shadowed_points};

  torch::Tensor triangles = torch::empty({n_scenes, n_tris, n_tri_values});
  torch::Tensor scenes = torch::empty({n_scenes, n_scene_values});
  torch::Tensor baryocentric_coords =
      torch::empty({n_scenes, n_samples_per_scene, n_baryo_dims});
  torch::Tensor values =
      torch::empty({n_scenes, n_samples_per_scene, n_rgb_dims});
  torch::Tensor indexes;
  if constexpr (is_image) {
    indexes =
        torch::empty({int(baryocentric_indexes.size()), 2}, long_tensor_type);

    for (int i = 0; i < int(baryocentric_indexes.size()); ++i) {
      auto [x, y] = baryocentric_indexes[i];
      indexes.index_put_({i, 0}, int64_t(x));
      indexes.index_put_({i, 1}, int64_t(y));
    }
  }

  // could use more than cpu cores really - goal is async...
#pragma omp parallel for schedule(dynamic, 8) if (!debug_build)
  for (int i = 0; i < n_scenes; ++i) {
    UniformState rng_state(base_seed + i);
  restart:
    auto tris = generate_scene_triangles(rng_state);
    auto new_tris = normalize_scene_triangles(tris);

    auto scene_adder = make_value_adder([&](float v, int idx) {
      scenes.index_put_({i, idx}, v);
    });

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

    // l - r
    // TODO: fix this eventually!
    // Fix pushing points away.
    // Fix removal
    // Maybe both of these are actually fine?
    auto region_difference =
        [](const TriangleSubset &l,
           const TriangleSubset &r) -> VectorT<TriangleSubset> {
      if (l.type() == TriangleSubsetType::None ||
          r.type() == TriangleSubsetType::All) {
        return {{tag_v<TriangleSubsetType::None>, {}}};
      }
      if (r.type() == TriangleSubsetType::None) {
        return {l};
      }
      const TriPolygon *l_poly;
      TriPolygon r_poly = r.get(tag_v<TriangleSubsetType::Some>);
      Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
      for (unsigned i = 0; i < r_poly.outer().size() - 1; ++i) {
        centroid += baryo_to_eigen(r_poly.outer()[i]);
      }
      centroid /= r_poly.outer().size() - 1;
      for (auto &p : r_poly.outer()) {
        // go slightly further from centroid
        Eigen::Vector2d new_point =
            (1. + 1e-6) * (baryo_to_eigen(p) - centroid) + centroid;
        p = {new_point.x(), new_point.y()};
      }

      l.visit_tagged([&](auto tag, const auto &val) {
        if constexpr (tag == TriangleSubsetType::Some) {
          l_poly = &val;
        } else if constexpr (tag == TriangleSubsetType::All) {
          l_poly = &full_triangle;
        } else {
          static_assert(tag == TriangleSubsetType::None);
          unreachable_unchecked();
        }
      });

      VectorT<TriPolygon> out_polys;
      boost::geometry::difference(*l_poly, r_poly, out_polys);
      VectorT<TriangleSubset> out;
      out.reserve(out_polys.size());
      for (TriPolygon &p : out_polys) {
        debug_assert(p.outer().size() >= 4);

        // NOTE: multiple stages of removal might be needed -
        // we don't do this right now...
        VectorT<unsigned> to_retain;
        for (unsigned i = 0; i < p.outer().size() - 1; ++i) {
          unsigned i_prev = i == 0 ? p.outer().size() - 2 : i - 1;
          unsigned i_next = i + 1;

          const auto baryo_prev = baryo_to_eigen(p.outer()[i_prev]);
          const auto baryo = baryo_to_eigen(p.outer()[i]);
          const auto baryo_next = baryo_to_eigen(p.outer()[i_next]);
          const auto vec0 = (baryo_prev - baryo).normalized();
          const auto vec1 = (baryo_next - baryo).normalized();
          double dotted = vec0.dot(vec1);
          if (dotted < dotted_thresh) {
            // not a spike, so we will retain
            to_retain.push_back(i);
          } else {
            dbg("REMOVED POINT!");
          }
        }

        if (to_retain.size() < 3) {
          // nothing left - just continue
          continue;
        }
        if (to_retain.size() == p.outer().size() - 1) {
          // retain everything
          out.push_back({tag_v<TriangleSubsetType::Some>, p});
        } else {

          TriPolygon new_poly;
          for (unsigned retained : to_retain) {
            boost::geometry::append(new_poly, p.outer()[retained]);
          }
          boost::geometry::append(new_poly, p.outer()[to_retain[0]]);
          debug_assert(boost::geometry::is_valid(new_poly));

          out.push_back({tag_v<TriangleSubsetType::Some>, new_poly});
        }
      };
      return out;
    };

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
        always_assert(onto_idx == 0);

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
        always_assert(onto_idx == 0);

        // not blocked at all
        goto restart;
      }

      std::array<TorchIdxT, n_prior_dims> prior_idxs{i, feature_idx};

      totally_shadowed_setter.set_region(
          prior_idxs, totally_shadowed_intersected, *end_tris[1]);

      partially_visible_setter.add_regions(
          prior_idxs,
          region_difference(*end_region[1], shadow_info.totally_shadowed),
          *end_tris[1]);

      partially_shadowed_setter.set_region(
          prior_idxs, partially_shadowed_intersected, *end_tris[1]);

      totally_visible_setter.add_regions(
          prior_idxs,
          region_difference(*end_region[1],
                            partially_shadowed_info.partially_shadowed),
          *end_tris[1]);

      auto centroid_shadow = shadowed_from_point(
          region_centroids[onto_idx],
          get_points_from_subset(new_tris.triangle_blocking, blocking_region),
          *end_tris[1]);
      centroid_shadowed_setter.set_region(prior_idxs, centroid_shadow,
                                          *end_tris[1]);
    }

    // TODO: precompute and use normals
    for (int tri_idx = 0; tri_idx < int(tri_pointers.size()); ++tri_idx) {
      const auto &tri = *tri_pointers[tri_idx];

      // add tri scene values

      // TODO: logs or other functions of these values?
      auto tri_adder = make_value_adder([&](float v, int value_idx) {
        triangles.index_put_({i, tri_idx, value_idx}, v);
      });

      for (const auto &point : tri.vertices) {
        tri_adder.add_values(point);
      }
      const auto normal_scaled = tri.normal_scaled_by_area();
      tri_adder.add_values(normal_scaled);
      tri_adder.add_values(centroids[tri_idx]);
      tri_adder.add_values(region_centroids[tri_idx]);
      const auto normal = normal_scaled.normalized().eval();
      tri_adder.add_values(normal);
      tri_adder.add_value(tri.area());
      double region_area = clipped_setter.set_region(
          {i, tri_idx}, *region_pointers[tri_idx], tri);
      clip_removed_setter.add_regions(
          {i, tri_idx},
          region_difference({tag_v<TriangleSubsetType::All>, {}},
                            *region_pointers[tri_idx]),
          tri);
      tri_adder.add_value(region_area);
      debug_assert(tri_adder.idx == n_tri_values);

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
          scene_adder.add_values(centroid_vec_raw);
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

    always_assert(scene_adder.idx == n_scene_values);

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
      baryocentric_coords.index_put_({i, j, 0}, s);
      baryocentric_coords.index_put_({i, j, 1}, t);
    }

    VectorT<FloatRGB> values_vec(n_samples_per_scene);

    auto scene = generate_scene(tris);
    always_assert(size_t(omp_get_thread_num()) < renderers.size());
    renderers[omp_get_thread_num()].render(
        ExecutionModel::GPU, {tag_v<render::SampleSpecType::InitialRays>, rays},
        {tag_v<render::OutputType::FloatRGB>, values_vec}, scene, n_samples,
        settings, false);
    for (int j = 0; j < n_samples_per_scene; ++j) {
      for (int k = 0; k < 3; ++k) {
        values.index_put_({i, j, k}, values_vec[j][k]);
      }
    }
  }

  dbg(clipped_setter.actual_max_n_points);
  dbg(clip_removed_setter.setter.actual_max_n_points);
  dbg(clip_removed_setter.actual_max_n_polys);
  dbg(totally_shadowed_setter.actual_max_n_points);
  dbg(partially_visible_setter.setter.actual_max_n_points);
  dbg(partially_visible_setter.actual_max_n_polys);
  dbg(partially_shadowed_setter.actual_max_n_points);
  dbg(totally_visible_setter.actual_max_n_polys);
  dbg(totally_visible_setter.setter.actual_max_n_points);
  dbg(centroid_shadowed_setter.actual_max_n_points);

  if constexpr (is_image) {
    return {scenes, baryocentric_coords, values, indexes};
  } else {
    return {scenes, baryocentric_coords, values};
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
gen_data(int n_scenes, int n_samples_per_scene, int n_samples,
         unsigned base_seed) {
  return gen_data_impl<false>(n_scenes, n_samples_per_scene, n_samples,
                              base_seed);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
gen_data_for_image(int n_scenes, int dim, int n_samples, unsigned base_seed) {
  return gen_data_impl<true>(n_scenes, dim, n_samples, base_seed);
}

void deinit_renderers() { renderers.resize(0); }
} // namespace generate_data
