#pragma once

#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/triangle_impl.h"
#include "lib/group.h"
#include "meta/dispatch_value.h"
#include "render/detail/compile_time_settings_impl.h"
#include "render/detail/integrate_image.h"
#include "render/detail/reduce_intensities_gpu.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/work_division.h"

namespace render {
using namespace detail;

template <ExecutionModel exec>
void Renderer::Impl<exec>::general_render(
    bool output_as_bgra, Span<BGRA> pixels, Span<Eigen::Array3f> intensities,
    const scene::Scene &s, unsigned &samples_per, unsigned x_dim,
    unsigned y_dim, const Settings &settings, bool show_progress, bool) {
  // only used for gpu
  WorkDivision division;

  if (exec == ExecutionModel::GPU) {
    division = WorkDivision(
        settings.general_settings.computation_settings.render_work_division,
        samples_per, x_dim, y_dim);
  }

  Span<BGRA> output_pixels;
  Span<Eigen::Array3f> output_intensities;

  if (output_as_bgra) {
    if constexpr (exec == ExecutionModel::GPU) {
      if (division.num_sample_blocks() != 1 || output_as_bgra) {
        intensities_.resize(division.num_sample_blocks() * x_dim * y_dim);
        output_intensities = intensities_;
      }

      bgra_.resize(x_dim * y_dim);
      output_pixels = bgra_;
    } else {
      output_pixels = pixels;
    }
  } else {
    if constexpr (exec == ExecutionModel::GPU) {
      intensities_.resize(division.num_sample_blocks() * x_dim * y_dim);
      output_intensities = intensities_;
    } else {
      output_intensities = intensities;
    }
  }

  dispatch_value(
      [&](auto &&settings_tup) {
        // TODO: consider dispatching more generically...
        constexpr CompileTimeSettings compile_time_settings =
            std::decay_t<decltype(settings_tup)>::value;

        constexpr auto flat_accel_type = compile_time_settings.flat_accel_type;

        auto scene_ref =
            stored_scene_generators_.template get<flat_accel_type>().gen(
                intersectable_scene::flat_triangle::Settings<
                    intersect::accel::enum_accel::Settings<flat_accel_type>>{
                    settings.flat_accel.template get<flat_accel_type>()},
                s);

        constexpr auto light_sampler_type =
            compile_time_settings.light_sampler_type;
        constexpr auto dir_sampler_type =
            compile_time_settings.dir_sampler_type;
        constexpr auto term_prob_type = compile_time_settings.term_prob_type;
        constexpr auto rng_type = compile_time_settings.rng_type;

        auto light_sampler =
            light_samplers_.template get<light_sampler_type>().gen(
                settings.light_sampler.template get<light_sampler_type>(),
                s.emissive_clusters(), s.emissive_cluster_ends_per_mesh(),
                s.materials().as_unsized(), s.transformed_mesh_objects(),
                s.transformed_mesh_idxs(), s.triangles().as_unsized());

        auto dir_sampler = dir_samplers_.template get<dir_sampler_type>().gen(
            settings.dir_sampler.template get<dir_sampler_type>());

        auto term_prob = term_probs_.template get<term_prob_type>().gen(
            settings.term_prob.template get<term_prob_type>());

        unsigned n_locations = x_dim * y_dim;

        auto rng = rngs_.template get<rng_type>().gen(
            settings.rng.template get<rng_type>(), samples_per, n_locations);

        integrate_image(output_as_bgra, settings.general_settings,
                        show_progress, division, samples_per, x_dim, y_dim,
                        scene_ref, light_sampler, dir_sampler, term_prob, rng,
                        output_pixels, output_intensities, s.film_to_world());
      },
      settings.compile_time);

  if constexpr (exec == ExecutionModel::GPU) {
    auto intensities_gpu = reduce_intensities_gpu(
        output_as_bgra, division.num_sample_blocks(), samples_per,
        &intensities_, &reduced_intensities_, bgra_);

    if (output_as_bgra) {
      thrust::copy(bgra_.begin(), bgra_.end(), pixels.begin());
    } else {
      thrust::copy(intensities_gpu->begin(), intensities_gpu->end(),
                   intensities.begin());
    }
  }
}
} // namespace render
