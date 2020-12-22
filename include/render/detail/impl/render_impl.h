#pragma once

#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/impl/ray_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "lib/group.h"
#include "meta/dispatch_value.h"
#include "render/detail/divide_work.h"
#include "render/detail/intensities.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/tone_map.h"

namespace render {
namespace detail {
template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(Span<BGRA> pixels,
                                           const scene::Scene &s,
                                           unsigned &samples_per,
                                           unsigned x_dim, unsigned y_dim,
                                           const Settings &settings,
                                           bool show_progress, bool) {
  auto division = divide_work(samples_per, x_dim, y_dim);

  if (execution_model == ExecutionModel::GPU) {
    samples_per = (division.num_sample_blocks * division.block_size *
                   division.samples_per_thread) /
                  (division.x_block_size * division.y_block_size);
  }

  Span<const scene::TriangleData> triangle_data;
  Span<const material::Material> materials;
  Span<BGRA> output_pixels;

  if constexpr (execution_model == ExecutionModel::GPU) {
    auto inp_t_data = s.triangle_data();
    auto inp_materials = s.materials();

    triangle_data_.resize(inp_t_data.size());
    materials_.resize(inp_materials.size());

    thrust::copy(inp_t_data.begin(), inp_t_data.end(), triangle_data_.begin());
    thrust::copy(inp_materials.begin(), inp_materials.end(),
                 materials_.begin());

    triangle_data = triangle_data_;
    materials = materials_;

    if (division.num_sample_blocks != 1) {
      intensities_.resize(division.num_sample_blocks * x_dim * y_dim);
    }

    bgra_.resize(x_dim * y_dim);
    output_pixels = bgra_;
  } else {
    triangle_data = s.triangle_data();
    materials = s.materials();
    output_pixels = pixels;
  }

  dispatch_value(
      [&](auto &&settings_tup) {
        // TODO: consider dispatching more generically...
        constexpr CompileTimeSettings compile_time_settings =
            std::decay_t<decltype(settings_tup)>::value;

        constexpr auto flat_accel_type =
            compile_time_settings.flat_accel_type();

        auto scene_ref =
            stored_scene_generators_.template get<flat_accel_type>().gen(
                intersectable_scene::flat_triangle::Settings<
                    intersect::accel::enum_accel::Settings<flat_accel_type>>{
                    settings.flat_accel.template get<flat_accel_type>()},
                s);

        constexpr auto light_sampler_type =
            compile_time_settings.light_sampler_type();
        constexpr auto dir_sampler_type =
            compile_time_settings.dir_sampler_type();
        constexpr auto term_prob_type = compile_time_settings.term_prob_type();
        constexpr auto rng_type = compile_time_settings.rng_type();

        auto light_sampler =
            light_samplers_.template get<light_sampler_type>().gen(
                settings.light_sampler.template get<light_sampler_type>(),
                s.emissive_clusters(), s.emissive_cluster_ends_per_mesh(),
                s.materials(), s.transformed_mesh_objects(),
                s.transformed_mesh_idxs(), s.triangles());

        auto dir_sampler = dir_samplers_.template get<dir_sampler_type>().gen(
            settings.dir_sampler.template get<dir_sampler_type>());

        auto term_prob = term_probs_.template get<term_prob_type>().gen(
            settings.term_prob.template get<term_prob_type>());

        // TODO
        const unsigned max_draws_per_sample = 128;

        auto rng = rngs_.template get<rng_type>().gen(
            settings.rng.template get<rng_type>(), samples_per, x_dim, y_dim,
            max_draws_per_sample);

        intensities(settings.general_settings, show_progress, division,
                    samples_per, x_dim, y_dim, scene_ref, light_sampler,
                    dir_sampler, term_prob, rng, output_pixels, intensities_,
                    s.film_to_world());
      },
      settings.compile_time.values());

  if constexpr (execution_model == ExecutionModel::GPU) {
    if (division.num_sample_blocks != 1) {
      tone_map<execution_model>(intensities_, bgra_);
    }

    thrust::copy(bgra_.begin(), bgra_.end(), pixels.begin());
  }
}
} // namespace detail
} // namespace render
