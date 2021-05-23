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
#include "lib/array_vec.h"
#include "lib/async_for.h"
#include "lib/projection.h"
#include "render/renderer.h"
#include "rng/uniform/uniform.h"

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
constexpr int n_scene_values = 14 * n_tris;
constexpr int n_dims = 3;
constexpr int n_tri_values = n_tris * n_dims + n_dims + n_dims + 1;
constexpr int n_baryo_dims = 2;
constexpr int n_poly_point_values = n_baryo_dims + n_dims + 1;
constexpr int n_rgb_dims = 3;
constexpr int n_shadowable_tris = 2;   // onto and light
constexpr int n_poly_feature_dims = 2; // area and properly scaled area

using TorchIdxT = int64_t;
const static auto long_tensor_type =
    torch::TensorOptions(caffe2::TypeMeta::Make<TorchIdxT>());

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
    removed_regions = torch::empty(region_dims);
    std::array<TorchIdxT, n_prior_dims + 1> feature_dims;
    std::copy(prior_dims.begin(), prior_dims.end(), feature_dims.begin());
    feature_dims[n_prior_dims] = n_poly_feature_dims;
    overall_features = torch::empty(feature_dims);
    counts = torch::empty(prior_dims, long_tensor_type);
  }

  void set_region(std::array<TorchIdxT, n_prior_dims> prior_idxs,
                  const TriangleSubset &region, const Triangle &tri) {
    region.visit_tagged([&](auto tag, const auto &value) {
      std::array<TorchIdxT, n_prior_dims + 1> feature_idxs;
      std::copy(prior_idxs.begin(), prior_idxs.end(), feature_idxs.begin());

      auto add_feature = [&](int idx, float v) {
        feature_idxs[n_prior_dims] = idx;
        overall_features.index_put_(feature_idxs, v);
      };

      if constexpr (tag == TriangleSubsetType::None) {
        counts.index_put_(prior_idxs, magic_value_none);
        add_feature(0, 0.);
        add_feature(1, 0.);
      } else {
        const auto pb = get_points_from_subset_with_baryo(tri, region);
        always_assert(pb.points.size() == pb.baryo.size());
        always_assert(pb.points.size() <= max_n_points);

        counts.index_put_(prior_idxs, pb.baryo.size());

        // get areas in baryo and in 3d
        const TriPolygon *baryo_poly;
        TriPolygon poly_all;
        if constexpr (tag == TriangleSubsetType::All) {
          for (const auto &p : pb.baryo) {
            boost::geometry::append(poly_all, p);
          }
          boost::geometry::append(poly_all, pb.baryo[0]);
          baryo_poly = &poly_all;
        } else {
          static_assert(tag == TriangleSubsetType::Some);

          baryo_poly = &value;
        }
        using BoostPoint3d = boost::geometry::model::d3::point_xyz<double>;
        boost::geometry::model::polygon<BoostPoint3d> poly_3d;
        auto add_point_to_3d = [&](const Eigen::Vector3d &p) {
          boost::geometry::append(poly_3d, BoostPoint3d{p.x(), p.y(), p.z()});
        };
        for (const auto &p : pb.points) {
          add_point_to_3d(p);
        }
        add_point_to_3d(pb.points[0]);

        debug_assert(boost::geometry::is_valid(baryo_poly));
        debug_assert(boost::geometry::is_valid(poly_3d));

        add_feature(0, boost::geometry::area(baryo_poly));
        add_feature(1, boost::geometry::area(poly_3d));

        std::array<TorchIdxT, n_prior_dims + 2> idxs;
        std::copy(prior_idxs.begin(), prior_idxs.end(), idxs.begin());

        for (unsigned i = 0; i < pb.baryo.size(); ++i) {
          idxs[n_prior_dims] = i;

          auto value_adder = make_value_adder([&](float v, int idx) {
            idxs[n_prior_dims + 1] = idx;
            removed_regions.index_put_(idxs, v);
          });

          unsigned i_prev = (i == 0) ? pb.baryo.size() - 1 : i - 1;
          unsigned i_next = (i + 1) % pb.baryo.size();

          auto as_eigen = [](const BaryoPoint &p) -> Eigen::Vector2d {
            return {p.x(), p.y()};
          };

          const auto baryo_prev = as_eigen(pb.baryo[i_prev]);
          const auto baryo = as_eigen(pb.baryo[i]);
          const auto baryo_next = as_eigen(pb.baryo[i_next]);

          double angle =
              std::acos((baryo_prev - baryo).dot(baryo_next - baryo));

          value_adder.add_value(angle);
          value_adder.add_values(baryo);
          value_adder.add_values(pb.points[i]);
        }
      }
    });
  }

  unsigned max_n_points;
  torch::Tensor removed_regions;
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

  void add_regions(std::array<TorchIdxT, n_prior_dims> prior_idxs,
                   const VectorT<TriangleSubset> &regions,
                   const Triangle &tri) {
    always_assert(int(regions.size()) <= max_n_polys);
    poly_counts.index_put_(prior_idxs, int(regions.size()));
    std::array<TorchIdxT, n_prior_dims + 1> all_prior_idxs;
    std::copy(prior_idxs.begin(), prior_idxs.end(), all_prior_idxs.begin());
    for (int i = 0; i < int(regions.size()); ++i) {
      all_prior_idxs[n_prior_dims] = i;
      setter.set_region(all_prior_idxs, regions[i], tri);
    }
  }

  int max_n_polys;
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

  std::array<TorchIdxT, n_prior_dims> shadowed_prior_dims{n_scenes,
                                                          n_shadowable_tris};

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

    auto print_subset = [](const TriangleSubset &subset) {
      subset.visit_tagged([&](auto tag, const auto &value) {
        if constexpr (tag == TriangleSubsetType::None) {
          std::cout << "None";
        } else if constexpr (tag == TriangleSubsetType::All) {
          std::cout << "'all'";
        } else {
          static_assert(tag == TriangleSubsetType::Some);
          std::cout << "[";
          debug_assert(value.inners().empty());
          for (auto point : value.outer()) {
            std::cout << "[" << point.x() << ", " << point.y() << "], ";
          }
          std::cout << "]";
        }
        std::cout << "\n";
      });
    };

    for (bool onto_light : {false, true}) {
      // TODO: use regions + feature etc.. (including centroid)
      std::array<const intersect::TriangleGen<double> *, 2> end_tris{
          &new_tris.triangle_light, &new_tris.triangle_onto};
      std::array<const TriangleSubset *, 2> end_region{&light_region,
                                                       &onto_region};
      if (onto_light) {
        std::swap(end_tris[0], end_tris[1]);
        std::swap(end_region[0], end_region[1]);
      }
      auto shadow_info = totally_shadowed(*end_tris[0], *end_region[0],
                                          new_tris.triangle_blocking,
                                          blocking_region, *end_tris[1]);
      if (shadow_info.totally_shadowed.type() == TriangleSubsetType::All) {
        // assert that we aren't on the second iter (should have already
        // restarted)
        always_assert(!onto_light);

        // totally blocked
        goto restart;
      }

      // check if clipped region is totally shadowed also
      if (shadow_info.totally_shadowed.type() == TriangleSubsetType::Some &&
          end_region[1]->type() == TriangleSubsetType::Some) {
        const auto &end_region_poly =
            end_region[1]->get(tag_v<TriangleSubsetType::Some>);
        VectorT<TriPolygon> intersection;
        boost::geometry::intersection(
            end_region_poly,
            shadow_info.totally_shadowed.get(tag_v<TriangleSubsetType::Some>),
            intersection);
        double region_area = boost::geometry::area(end_region_poly);
        double total_area = 0.;
        for (const auto &poly : intersection) {
          total_area += boost::geometry::area(poly);
        }
        if (std::abs(region_area - total_area) < 1e-10) {
          // totally blocked
          goto restart;
        }
      }

      auto some_shadowed_info = some_shadowed(*end_tris[0], *end_region[0],
                                              new_tris.triangle_blocking,
                                              blocking_region, *end_tris[1]);

      if (some_shadowed_info.some_shadowed.type() == TriangleSubsetType::None) {
        // assert that we aren't on the second iter (should have already
        // restarted)
        always_assert(!onto_light);

        // not blocked at all
        goto restart;
      }

      const auto base = onto_light ? "onto_light_" : "onto_";

      std::cout << base << "some_blocking = ";
      print_subset(some_shadowed_info.some_shadowed);
      std::cout << base << "totally_blocked = ";
      print_subset(shadow_info.totally_shadowed);
      for (unsigned i = 0; i < shadow_info.from_each_point.size(); ++i) {
        std::cout << base << "from_each_point_" << i << " = ";
        print_subset(shadow_info.from_each_point[i]);
      }

      auto from_points = get_points_from_subset(*end_tris[0], *end_region[0]);
      Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
      for (const auto &p : from_points) {
        centroid += p;
      }
      centroid /= from_points.size();

      auto centroid_shadow = shadowed_from_point(
          centroid,
          get_points_from_subset(new_tris.triangle_blocking, blocking_region),
          *end_tris[1]);
      std::cout << base << "centroid_shadow = ";
      print_subset(centroid_shadow);
    }

    std::cout << "triangle_onto = ";
    print_triangle(new_tris.triangle_onto.template cast<float>());
    std::cout << "triangle_blocking = ";
    print_triangle(new_tris.triangle_blocking.template cast<float>());
    std::cout << "triangle_light = ";
    print_triangle(new_tris.triangle_light.template cast<float>());

    std::cout << "onto_region = ";
    print_subset(onto_region);
    std::cout << "blocking_region = ";
    print_subset(blocking_region);
    std::cout << "light_region = ";
    print_subset(light_region);

    std::array tri_pointers{
        &new_tris.triangle_onto,
        &new_tris.triangle_blocking,
        &new_tris.triangle_light,
    };
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
      const auto normal = normal_scaled.normalized().eval();
      tri_adder.add_values(normal);
      tri_adder.add_value(tri.area());
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
        Eigen::Vector3d centroid_vec_raw =
            other_tri.centroid() - tri.centroid();
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

    std::cout << "\n\n\n\n";
  }

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
