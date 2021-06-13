#include "generate_data/full_scene/generate_data.h"
#include "generate_data/baryocentric_to_ray.h"
#include "generate_data/full_scene/amend_config.h"
#include "generate_data/full_scene/constants.h"
#include "generate_data/full_scene/intersect_for_baryocentric_coords.h"
#include "generate_data/full_scene/scene_generator.h"
#include "generate_data/possibly_shadowed.h"
#include "generate_data/shadowed.h"
#include "generate_data/sort_triangle_points.h"
#include "generate_data/subset_to_multi.h"
#include "generate_data/to_tensor.h"
#include "generate_data/triangle.h"
#include "generate_data/triangle_subset_intersection.h"
#include "generate_data/triangle_subset_union.h"
#include "generate_data/value_adder.h"
#include "integrate/dir_sampler/uniform_direction_sample.h"
#include "lib/vector_type.h"
#include "meta/array_cat.h"
#include "render/renderer.h"
#include "rng/uniform/uniform.h"

#include <boost/multi_array.hpp>

#include "dbg.h"

namespace generate_data {
namespace full_scene {
static VectorT<render::Renderer> renderers;

template <bool is_image>
using Out = std::conditional_t<is_image, ImageData<NetworkInputs>,
                               StandardData<NetworkInputs>>;

template <bool is_image>
Out<is_image> generate_data_impl(int max_tris,
                                 std::optional<int> forced_n_scenes,
                                 int n_samples_per_tri_or_dim, int n_samples,
                                 int n_steps, unsigned seed) {
  VectorT<IntersectedBaryocentricCoords> intersected_coords;
  unsigned max_n_samples_per_scene = 0;
  VectorT<scene::Scene> scenes;
  VectorT<unsigned> mesh_threshs; // not used ATM
  SceneGenerator generator;
  int tri_count = 0;
  unsigned max_tri_count = 0;
  rng::uniform::Uniform<ExecutionModel::CPU>::Ref::State rng_state(seed);
  std::vector<unsigned> n_total_samples_per;
  while (forced_n_scenes.has_value()
             ? scenes.size() < unsigned(*forced_n_scenes)
             : tri_count < max_tris) {
    auto vals = generator.generate(rng_state.state());
    auto scene = std::get<0>(vals);
    unsigned n_tris = scene.triangles().size();
    int new_tri_count = tri_count + n_tris;

    // TODO: restrict tris somehow?

    tri_count = new_tri_count;
    max_tri_count = std::max(max_tri_count, n_tris);

    unsigned n_total_samples;
    if constexpr (is_image) {
      unsigned dim = n_samples_per_tri_or_dim;
      auto values = intersect_for_baryocentric_coords(scene, dim);
      n_total_samples = unsigned(values.image_indexes.size());
      intersected_coords.push_back(values);
    } else {
      unsigned n_samples_per_tri = n_samples_per_tri_or_dim;
      n_total_samples = n_samples_per_tri * n_tris;
    }
    max_n_samples_per_scene =
        std::max(max_n_samples_per_scene, n_total_samples);
    n_total_samples_per.push_back(n_total_samples);

    scenes.push_back(scene);
    mesh_threshs.push_back(std::get<1>(vals));
  }

  render::Settings settings;
  amend_config(settings);

  std::array<unsigned, 2> base_dims{unsigned(scenes.size()), max_tri_count};

  boost::multi_array<float, 3> triangle_features{
      array_append(base_dims, unsigned(constants.n_tri_values))};
  boost::multi_array<float, 3> bsdf_features{
      array_append(base_dims, unsigned(constants.n_bsdf_values))};
  boost::multi_array<float, 3> emissive_values{
      array_append(base_dims, unsigned(constants.n_rgb_dims))};
  boost::multi_array<bool, 2> mask{base_dims};

  std::array<unsigned, 2> base_sample_dims{unsigned(scenes.size()),
                                           max_n_samples_per_scene};

  boost::multi_array<float, 3> baryocentric_coords{array_append(
      base_sample_dims, unsigned(constants.n_coords_feature_values))};
  boost::multi_array<TorchIdxT, 2> triangle_idxs_for_coords{base_sample_dims};
  boost::multi_array<TorchIdxT, 3> image_indexes;
  if constexpr (is_image) {
    image_indexes.resize(array_append(base_sample_dims, 2u));
  }
  boost::multi_array<float, 4> values{
      std::array{unsigned(scenes.size()), unsigned(n_steps),
                 max_n_samples_per_scene, unsigned(constants.n_rgb_dims)}};
  for (unsigned scene_idx = 0; scene_idx < scenes.size(); ++scene_idx) {
    const auto &scene = scenes[scene_idx];
    VectorT<Triangle> tris(scene.triangles().size());

    Eigen::Vector3d avg = Eigen::Vector3d::Zero();
    Eigen::Vector3d min_pos = max_eigen_vec<double>();
    Eigen::Vector3d max_pos = min_eigen_vec<double>();

    for (unsigned i = 0; i < tris.size(); ++i) {
      tris[i] = scene.triangles()[i].template cast<double>();
      for (const auto &vert : tris[i].vertices) {
        avg += vert;
        min_pos = min_pos.cwiseMin(vert);
        max_pos = max_pos.cwiseMax(vert);
      }
    }

    avg /= tris.size() * 3;

    double scale =
        1. / std::max((avg - min_pos).maxCoeff(), (max_pos - avg).maxCoeff());
    auto scaling = Eigen::Scaling(scale);
    auto translate = Eigen::Translation3d(-avg);
    const Eigen::Affine3d transform = scaling * translate;

    for (unsigned i = 0; i < tris.size(); ++i) {
      auto &tri = tris[i];
      for (auto &vert : tri.vertices) {
        vert = transform * vert;
      }

      sort_triangle_points(tri);

      if (tri.normal_raw().z() < 0.f) {
        std::swap(tri.vertices[1], tri.vertices[2]);
      }

      auto tri_adder = make_value_adder([&](float v, int value_idx) {
        triangle_features[scene_idx][i][value_idx] = v;
      });

      for (const auto &point : tri.vertices) {
        tri_adder.add_remap_all_values(point);
      }
      const auto normal_scaled = tri.normal_scaled_by_area();
      tri_adder.add_remap_all_values(normal_scaled);
      const auto normal = normal_scaled.normalized().eval();
      tri_adder.add_values(normal);
      tri_adder.add_remap_all_value(tri.area());

      debug_assert(tri_adder.idx == constants.n_tri_values);

      const auto &material =
          scene.materials()[scene.triangle_data()[i].material_idx()];

      make_value_adder([&](float v, int value_idx) {
        emissive_values[scene_idx][i][value_idx] = v;
      }).add_values(material.emission);

      FloatRGB diffuse_value = FloatRGB::Zero();
      FloatRGB glossy_value = FloatRGB::Zero();
      FloatRGB mirror_value = FloatRGB::Zero();
      FloatRGB dielectric_value = FloatRGB::Zero();
      float shininess = 40.;
      float ior = 1.6;

      material.bsdf.bsdf.visit_tagged([&](auto tag, const auto &value) {
        if constexpr (tag == bsdf::BSDFType::Diffuse) {
          diffuse_value = value.diffuse;
        } else if constexpr (tag == bsdf::BSDFType::Glossy) {
          glossy_value = value.specular();
          shininess = value.shininess();
        } else if constexpr (tag == bsdf::BSDFType::DiffuseGlossy) {
          float diffuse_weight = value.weight_continuous_inclusive()[0];
          float glossy_weight = 1 - diffuse_weight;
          diffuse_value =
              diffuse_weight * value.items()[boost::hana::int_c<0>].diffuse;
          glossy_value =
              glossy_weight * value.items()[boost::hana::int_c<1>].specular();
          shininess = value.items()[boost::hana::int_c<1>].shininess();
        } else if constexpr (tag == bsdf::BSDFType::DiffuseMirror) {
          unreachable_unchecked();
        } else if constexpr (tag == bsdf::BSDFType::Mirror) {
          mirror_value = value.specular;
        } else {
          static_assert(tag == bsdf::BSDFType::DielectricRefractive);
          dielectric_value = value.specular();
          ior = value.ior();
        }
      });

      auto bsdf_adder = make_value_adder([&](float v, int value_idx) {
        bsdf_features[scene_idx][i][value_idx] = v;
      });

      bsdf_adder.add_values(diffuse_value);
      bsdf_adder.add_values(glossy_value);
      bsdf_adder.add_values(mirror_value);
      bsdf_adder.add_values(dielectric_value);
      bsdf_adder.add_value(shininess);
      bsdf_adder.add_value(ior);

      debug_assert(bsdf_adder.idx == constants.n_bsdf_values);
    }

    for (unsigned i = 0; i < max_tri_count; ++i) {
      mask[scene_idx][i] = i >= tris.size();
    }

    unsigned n_samples_this_scene = [&] {
      if constexpr (is_image) {
        return intersected_coords[scene_idx].coords.size();
      } else {
        unsigned n_samples_per_tri = n_samples_per_tri_or_dim;
        return tris.size() * n_samples_per_tri;
      }
    }();

    VectorT<render::InitialIdxAndDirSpec> idxs_and_dirs(n_samples_this_scene);

    auto add_value = [&](unsigned i, float s, float t, unsigned tri_idx,
                         const UnitVector &dir) {
      const auto &tri = tris[tri_idx];
      idxs_and_dirs[i] = {
          .idx = tri_idx,
          .ray = baryocentric_to_ray(s, t, tri.template cast<float>(), dir),
      };
      triangle_idxs_for_coords[scene_idx][i] = tri_idx;

      auto baryo_addr = make_value_adder([&](float v, int value_idx) {
        baryocentric_coords[scene_idx][i][value_idx] = v;
      });

      baryo_addr.add_value(s);
      baryo_addr.add_value(t);
      const Eigen::Vector3d vec0 = tri.vertices[1] - tri.vertices[0];
      const Eigen::Vector3d vec1 = tri.vertices[2] - tri.vertices[0];

      const Eigen::Vector3d double_dir = dir->template cast<double>();
      Eigen::Vector3d dir_tri_space{tri.normal()->dot(double_dir),
                                    vec0.dot(double_dir) / vec0.norm(),
                                    vec1.dot(double_dir) / vec1.norm()};
      dir_tri_space.normalize();
      baryo_addr.add_values(dir_tri_space);
    };
    if constexpr (is_image) {
      const auto &item = intersected_coords[scene_idx];
      for (unsigned i = 0; i < item.coords.size(); ++i) {
        auto [s, t] = item.coords[i];
        add_value(i, s, t, item.tri_idxs[i], item.directions[i]);
      }
      for (unsigned i = 0; i < item.image_indexes.size(); ++i) {
        auto [x, y] = item.image_indexes[i];
        image_indexes[scene_idx][i][0] = TorchIdxT(x);
        image_indexes[scene_idx][i][1] = TorchIdxT(y);
      }
    } else {
      unsigned n_samples_per_tri = n_samples_per_tri_or_dim;
      unsigned i = 0;
      for (unsigned tri_idx = 0; tri_idx < tris.size(); ++tri_idx) {
        for (unsigned sample_idx = 0; sample_idx < n_samples_per_tri;
             ++sample_idx, ++i) {
          auto [s, t] = integrate::uniform_baryocentric(rng_state);
          auto direction = integrate::dir_sampler::uniform_direction_sample(
              rng_state, UnitVector::new_unchecked(Eigen::Vector3f::UnitX()),
              true);
          add_value(i, s, t, tri_idx, direction);
        }
      }
    }

    VectorT<VectorT<FloatRGB>> step_outputs(
        n_steps, VectorT<FloatRGB>{n_samples_this_scene});
    VectorT<Span<FloatRGB>> outputs(step_outputs.begin(), step_outputs.end());

    if (n_samples_this_scene > 0) {
      renderers.resize(1);
      renderers[0].render(
          ExecutionModel::GPU,
          {tag_v<render::SampleSpecType::InitialIdxAndDir>, idxs_and_dirs},
          {tag_v<render::OutputType::OutputPerStep>, outputs}, scene, n_samples,
          settings, false);
      for (unsigned j = 0; j < unsigned(n_steps); ++j) {
        for (unsigned k = 0; k < n_samples_this_scene; ++k) {
          for (unsigned l = 0; l < 3; ++l) {
            values[scene_idx][j][k][l] = outputs[j][k][l];
          }
        }
      }
    }
  }

  StandardData<NetworkInputs> out{
      .inputs =
          {
              .triangle_features = to_tensor(triangle_features),
              .mask = to_tensor(mask),
              .bsdf_features = to_tensor(bsdf_features),
              .emissive_values = to_tensor(emissive_values),
              .baryocentric_coords = to_tensor(baryocentric_coords),
              .triangle_idxs_for_coords = to_tensor(triangle_idxs_for_coords),
              .total_tri_count = unsigned(tri_count),
              .n_samples_per = n_total_samples_per,
          },

      .values = to_tensor(values),
  };

  if constexpr (is_image) {
    return {.standard = out, .image_indexes = to_tensor(image_indexes)};
  } else {
    return out;
  }
}

StandardData<NetworkInputs> generate_data(int max_tris,
                                          std::optional<int> forced_n_scenes,
                                          int n_samples_per_scene,
                                          int n_samples, int n_steps,
                                          unsigned base_seed) {
  return generate_data_impl<false>(max_tris, forced_n_scenes,
                                   n_samples_per_scene, n_samples, n_steps,
                                   base_seed);
}

ImageData<NetworkInputs>
generate_data_for_image(int max_tris, std::optional<int> forced_n_scenes,
                        int dim, int n_samples, int n_steps,
                        unsigned base_seed) {
  return generate_data_impl<true>(max_tris, forced_n_scenes, dim, n_samples,
                                  n_steps, base_seed);
}

void deinit_renderers() { renderers.clear(); }
} // namespace full_scene
} // namespace generate_data
