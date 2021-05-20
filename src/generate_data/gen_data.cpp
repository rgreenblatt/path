#include "generate_data/gen_data.h"
#include "generate_data/amend_config.h"
#include "generate_data/baryocentric_coords.h"
#include "generate_data/baryocentric_to_ray.h"
#include "generate_data/clip_by_plane.h"
#include "generate_data/generate_scene.h"
#include "generate_data/generate_scene_triangles.h"
#include "generate_data/get_dir_towards.h"
#include "generate_data/get_points_from_subset.h"
#include "generate_data/normalize_scene_triangles.h"
#include "generate_data/shadowed.h"
#include "generate_data/torch_rng_state.h"
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

// scenes, coords, values
// TODO: consider fixing extra copies (if needed).
// Could really return gpu tensor and output directly to tensor.
template <bool is_image>
Out<is_image> gen_data_impl(int n_scenes, int n_samples_per_scene_or_dim,
                            int n_samples, unsigned base_seed) {
  using namespace generate_data;
  using namespace torch::indexing;

  renderers.resize(omp_get_max_threads());

  // TODO: is sobel fine???
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

  int n_scene_values = 37;
  torch::Tensor scenes = torch::empty({n_scenes, n_scene_values});
  torch::Tensor baryocentric_coords =
      torch::empty({n_scenes, n_samples_per_scene, 2});
  torch::Tensor values = torch::empty({n_scenes, n_samples_per_scene, 3});
  torch::Tensor indexes;
  if constexpr (is_image) {
    indexes =
        torch::empty({int(baryocentric_indexes.size()), 2},
                     torch::TensorOptions(caffe2::TypeMeta::Make<int64_t>()));

    for (int i = 0; i < int(baryocentric_indexes.size()); ++i) {
      auto [x, y] = baryocentric_indexes[i];
      indexes.index_put_({i, 0}, int64_t(x));
      indexes.index_put_({i, 1}, int64_t(y));
    }
  }

  // could use more than cpu cores really - goal is async...
#pragma omp parallel for schedule(dynamic, 8) if (!debug_build)
  for (int i = 0; i < n_scenes; ++i) {
    rng::uniform::Uniform<ExecutionModel::CPU>::Ref::State rng_state(base_seed +
                                                                     i);

    auto tris = generate_scene_triangles(rng_state);
    auto scene = generate_scene(tris);

    auto dir_towards = get_dir_towards(tris);

    auto new_tris = normalize_scene_triangles(tris);
    int scene_idx = 0;
    for (const auto &tri : {
             new_tris.triangle_onto,
             new_tris.triangle_blocking,
             new_tris.triangle_light,
         }) {
      for (const auto &point : tri.vertices) {
        for (const auto v : point) {
          scenes.index_put_({i, scene_idx++}, v);
        }
      }
    }
    auto onto_normal = new_tris.triangle_onto.normal();
    for (const auto &other_tri :
         {new_tris.triangle_blocking, new_tris.triangle_light}) {
      scenes.index_put_({i, scene_idx++},
                        other_tri.normal()->dot(*onto_normal));
      scenes.index_put_({i, scene_idx++}, other_tri.area());
      for (int j = 0; j < 3; ++j) {
        scenes.index_put_({i, scene_idx++}, other_tri.centroid()[j]);
      }
    }

    std::cout << "triangle_onto = ";
    print_triangle(new_tris.triangle_onto);
    std::cout << "triangle_blocking = ";
    print_triangle(new_tris.triangle_blocking);
    std::cout << "triangle_light = ";
    print_triangle(new_tris.triangle_light);

    // TODO: use regions + feature etc..

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

    // clip by triangle_onto plane
    auto light_region =
        clip_by_plane(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(),
                      new_tris.triangle_light.template cast<double>());
    std::cout << "light_region = ";
    print_subset(light_region);
    auto blocking_region_initial =
        clip_by_plane(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(),
                      new_tris.triangle_blocking.template cast<double>());
    std::cout << "blocking_region_initial = ";
    print_subset(blocking_region_initial);

    // clip by triangle_light plane
    auto light_normal =
        new_tris.triangle_light.template cast<double>().normal_raw();
    auto light_point =
        new_tris.triangle_light.template cast<double>().vertices[0];
    auto onto_region =
        clip_by_plane(light_normal, light_point,
                      new_tris.triangle_onto.template cast<double>());
    std::cout << "onto_region = ";
    print_subset(onto_region);
    auto blocking_region = triangle_subset_intersection(
        blocking_region_initial,
        clip_by_plane(light_normal, light_point,
                      new_tris.triangle_blocking.template cast<double>()));
    std::cout << "blocking_region = ";
    print_subset(blocking_region);

    auto tris_as_double = new_tris.template cast<double>();
    for (bool onto_light : {false, true}) {
      std::array<const intersect::TriangleGen<double> *, 2> end_tris{
          &tris_as_double.triangle_light, &tris_as_double.triangle_onto};
      std::array<const TriangleSubset *, 2> end_region{&light_region,
                                                       &onto_region};
      if (onto_light) {
        std::swap(end_tris[0], end_tris[1]);
        std::swap(end_region[0], end_region[1]);
      }

      auto from_points = get_points_from_subset(*end_tris[0], *end_region[0]);
      Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
      for (const auto &p : from_points) {
        centroid += p;
      }
      centroid /= from_points.size();

      // TODO: use this stuff (including centroid!)

      const auto base = onto_light ? "onto_light_" : "onto_";

      auto centroid_shadow = shadowed_from_point(
          centroid, tris_as_double.triangle_blocking, *end_tris[1]);
      std::cout << base << "centroid_shadow = ";
      print_subset(centroid_shadow);

      auto shadow_info =
          shadowed(from_points, tris_as_double.triangle_blocking, *end_tris[1]);

      std::cout << base << "some_blocking = ";
      print_subset(shadow_info.some_blocking);
      std::cout << base << "totally_blocked = ";
      print_subset(shadow_info.totally_blocked);
      for (unsigned i = 0; i < shadow_info.from_each_point.size(); ++i) {
        std::cout << base << "from_each_point_" << i << " = ";
        print_subset(shadow_info.from_each_point[i]);
      }
    }

    always_assert(scene_idx == n_scene_values);

    VectorT<intersect::Ray> rays(n_samples_per_scene);

    for (int j = 0; j < n_samples_per_scene; ++j) {
      auto [s, t] = [&]() {
        if constexpr (is_image) {
          return baryocentric_grid_values[j];
        } else {
          return integrate::uniform_baryocentric(rng_state);
        }
      }();
      rays[j] = baryocentric_to_ray(s, t, tris.triangle_onto, dir_towards);
      baryocentric_coords.index_put_({i, j, 0}, s);
      baryocentric_coords.index_put_({i, j, 1}, t);
    }

    VectorT<FloatRGB> values_vec(n_samples_per_scene);

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
