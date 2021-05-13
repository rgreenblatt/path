#include "generate_data/gen_data.h"
#include "generate_data/amend_config.h"
#include "generate_data/baryocentric_to_ray.h"
#include "generate_data/generate_scene.h"
#include "generate_data/generate_scene_triangles.h"
#include "generate_data/get_dir_towards.h"
#include "generate_data/normalize_scene_triangles.h"
#include "generate_data/torch_rng_state.h"
#include "integrate/sample_triangle.h"
#include "lib/projection.h"
#include "render/renderer.h"

#include <torch/extension.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "dbg.h"

namespace generate_data {
// scenes, coords, values
// TODO: consider fixing extra copies (if needed).
// Could really return gpu tensor and output directly to tensor.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
gen_data(int n_scenes, int n_samples_per_scene, int n_samples) {
  using namespace generate_data;
  using namespace torch::indexing;

  // GlObAL vArIAbLes aRe FiNe...
  // also consider fixing "CUDA free failed: cudaErrorCudartUnloading: driver
  // shutting down"
  static render::Renderer renderer;

  // TODO: is sobel fine???
  render::Settings settings;
  amend_config(settings);

  TorchRng rng{};

  int n_scene_values = 24;
  torch::Tensor scenes = torch::empty({n_scenes, n_scene_values});
  torch::Tensor baryocentric_coords =
      torch::empty({n_scenes, n_samples_per_scene, 2});
  torch::Tensor values = torch::empty({n_scenes, n_samples_per_scene, 3});

  for (int i = 0; i < n_scenes; ++i) {
    auto tris = generate_scene_triangles(rng);
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
      auto [s, t] = integrate::uniform_baryocentric(rng);
      rays[j] = baryocentric_to_ray(s, t, tris.triangle_onto, dir_towards);
      baryocentric_coords.index_put_({i, j, 0}, s);
      baryocentric_coords.index_put_({i, j, 1}, t);
    }

    VectorT<FloatRGB> values_vec(n_samples_per_scene);

    renderer.render(ExecutionModel::GPU,
                    {tag_v<render::SampleSpecType::InitialRays>, rays},
                    {tag_v<render::OutputType::FloatRGB>, values_vec}, scene,
                    n_samples, settings, false);
    for (int j = 0; j < n_samples_per_scene; ++j) {
      for (int k = 0; k < 3; ++k) {
        values.index_put_({i, j, k}, values_vec[j][k]);
      }
    }
  }

  return {scenes, baryocentric_coords, values};
}
} // namespace generate_data
