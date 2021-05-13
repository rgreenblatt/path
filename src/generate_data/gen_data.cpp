#include "generate_data/gen_data.h"
#include "generate_data/amend_config.h"
#include "generate_data/baryocentric_coords.h"
#include "generate_data/baryocentric_to_ray.h"
#include "generate_data/generate_scene.h"
#include "generate_data/generate_scene_triangles.h"
#include "generate_data/get_dir_towards.h"
#include "generate_data/normalize_scene_triangles.h"
#include "generate_data/torch_rng_state.h"
#include "integrate/sample_triangle.h"
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

  // for (unsigned i = 0; i < baryocentric_grid_values.size(); ++i) {
  //   auto [x_v, y_v] = baryocentric_grid_values[i];
  //   baryocentric_grid.push_back(
  //       baryocentric_to_ray(x_v, y_v, tris.triangle_onto, dir_towards));
  // }

  int n_scene_values = 24;
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
    for (const auto &point : new_tris.triangle_onto.vertices) {
      for (float v : {point.x(), point.y()}) {
        scenes.index_put_({i, scene_idx++}, v);
      }
    }
    for (const auto &tri : {
             new_tris.triangle_blocking,
             new_tris.triangle_light,
         }) {
      for (const auto &point : tri.vertices) {
        for (const auto v : point) {
          scenes.index_put_({i, scene_idx++}, v);
        }
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
