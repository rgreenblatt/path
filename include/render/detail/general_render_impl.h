#pragma once

#include "integrate/rendering_equation_components.h"
#include "intersect/accel/enum_accel/enum_accel_impl.h"
#include "intersect/triangle_impl.h"
#include "intersectable_scene/to_bulk_impl.h"
#include "kernel/work_division.h"
#include "lib/info/timer.h"
#include "meta/all_values/dispatch.h"
#include "render/detail/integrate_image/mega_kernel/run.h"
#include "render/detail/integrate_image/streaming/run.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/settings_compile_time_impl.h"

#include <thrust/device_ptr.h>

namespace render {
using namespace detail;

template <ExecutionModel exec>
double Renderer::Impl<exec>::general_render(
    const SampleSpec &sample_spec, const Output &output, const scene::Scene &s,
    unsigned samples_per, const Settings &settings, bool show_progress,
    bool show_times) {

  Timer run_timer(std::nullopt);

  // We return float_rgb_out because the mega kernel reduce can pick
  // which value to avoid a copy. (might be premature optimization...)
  auto intermediate_output = dispatch(settings.compile_time(), [&](auto tag) {
    constexpr auto compile_time = tag();

    constexpr auto kernel_approach_type = compile_time.kernel_approach_type;
    constexpr auto light_sampler_type = compile_time.light_sampler_type;
    constexpr auto dir_sampler_type = compile_time.dir_sampler_type;
    constexpr auto term_prob_type = compile_time.term_prob_type;
    constexpr auto rng_type = compile_time.rng_type;

    constexpr auto kernel_approach = kernel_approach_type.type();

    auto as_intersectable_scene =
        [&](auto accel_type, IndividuallyIntersectableSettings settings) {
          auto accel_settings = settings.accel.get(accel_type);

          return stored_scene_generators_.get(accel_type)
              .gen({accel_settings}, s);
        };

    constexpr auto kernel_approach_type_value =
        kernel_approach_type.get(tag_v<kernel_approach>);
    const auto &kernel_approach_settings =
        settings.kernel_approach.get(tag_v<kernel_approach>);
    auto intersectable_items = [&]() {
      if constexpr (kernel_approach == KernelApproach::MegaKernel) {
        constexpr auto accel_type = kernel_approach_type_value;
        auto intersectable_scene = as_intersectable_scene(
            tag_v<accel_type>,
            kernel_approach_settings.individually_intersectable_settings);
        return std::tuple{intersectable_scene.scene,
                          intersectable_scene.intersector};
      } else {
        static_assert(kernel_approach == KernelApproach::Streaming);
        auto &accel = kernel_approach_settings.accel;
        return accel.visit_tagged([&](auto tag, const auto &value) {
          // If more cases are added, this will need to be an set of if else
          // statements.
          static_assert(
              tag ==
              StreamingSettings::BulkIntersectionApproaches::IndividualToBulk);

          constexpr auto accel_type = kernel_approach_type_value.get(tag);
          auto intersectable_scene = as_intersectable_scene(
              tag_v<accel_type>, value.individual_settings);

          auto &intersector = to_bulk_.get(tag_v<accel_type>);
          intersector.set_settings_intersectable(
              value.to_bulk_settings, intersectable_scene.intersector);

          // Specify exact type to make sure we get a reference for
          // intersector.
          return std::tuple<decltype(intersectable_scene.scene),
                            decltype(intersector)>{intersectable_scene.scene,
                                                   intersector};
        });
      }
    }();

    auto scene = std::get<0>(intersectable_items);
    auto &intersector = std::get<1>(intersectable_items);

    auto light_sampler =
        light_samplers_.get(tag_v<light_sampler_type>)
            .gen(settings.light_sampler.get(tag_v<light_sampler_type>),
                 s.emissive_clusters(), s.emissive_cluster_ends_per_mesh(),
                 s.materials().as_unsized(), s.transformed_mesh_objects(),
                 s.transformed_mesh_idxs(), s.triangles().as_unsized());

    auto dir_sampler =
        dir_samplers_.get(tag_v<dir_sampler_type>)
            .gen(settings.dir_sampler.get(tag_v<dir_sampler_type>));

    auto term_prob = term_probs_.get(tag_v<term_prob_type>)
                         .gen(settings.term_prob.get(tag_v<term_prob_type>));

    unsigned n_locations =
        sample_spec.visit_tagged([&](auto tag, const auto &sample_spec) {
          if constexpr (tag == SampleSpecType::SquareImage) {
            return sample_spec.x_dim * sample_spec.y_dim;
          } else {
            static_assert(tag == SampleSpecType::InitialRays);
            return sample_spec.size();
          }
        });

    auto rng =
        rngs_.get(tag_v<rng_type>)
            .gen(settings.rng.get(tag_v<rng_type>), samples_per, n_locations);

    if (output.type() == OutputType::BGRA) {
      bgra_32_.resize(n_locations);
    }

    const auto device_sample_spec =
        sample_spec.visit_tagged([&](auto tag, const auto &spec) -> SampleSpec {
          if constexpr (tag == SampleSpecType::SquareImage) {
            return sample_spec;
          } else {
            static_assert(tag == SampleSpecType::InitialRays);

            copy_to_vec(spec, sample_rays_);

            return {tag, sample_rays_};
          }
        });

    // not really any good way to infer these arguments...
    using Components = integrate::RenderingEquationComponents<
        decltype(scene), decltype(light_sampler), decltype(dir_sampler),
        decltype(term_prob)>;
    using Items = integrate_image::Items<Components, decltype(rng)>;
    integrate_image::Inputs<Items> inputs = {
        .items =
            {
                .components =
                    {
                        .scene = scene,
                        .light_sampler = light_sampler,
                        .dir_sampler = dir_sampler,
                        .term_prob = term_prob,
                    },
                .rng = rng,
                .film_to_world = s.film_to_world(),
            },
        .output_type = output.type(),
        .sample_spec = device_sample_spec,
        .samples_per = samples_per,
        .show_progress = show_progress,
        .show_times = show_times,
    };
    return [&]() {
      run_timer.start();
      if constexpr (kernel_approach == KernelApproach::MegaKernel) {
        return integrate_image::mega_kernel::Run<exec>::run(
            inputs, intersector, kernel_approach_settings, bgra_32_, float_rgb_,
            output_per_step_rgb_);
      } else {
        static_assert(kernel_approach == KernelApproach::Streaming);

        auto &streaming_state = streaming_state_.get(
            tag_v<make_meta_tuple(light_sampler_type, rng_type)>);

        return integrate_image::streaming::Run<exec>::run(
            inputs, intersector, streaming_state, kernel_approach_settings,
            bgra_32_, float_rgb_[0], output_per_step_rgb_[0]);
      }
    }();
  });

  always_assert(intermediate_output.type() == output.type());

  // We could save a copy in the cpu case by directly outputing to the input
  // bgra_32/float_rgb when possible.
  // However, the perf gains are small and we don't really
  // case about the performance of the cpu use case.
  intermediate_output.visit_tagged([&](auto tag,
                                       const auto &intermediate_output_v) {
    auto get_ptr = [](auto ptr) {
      if constexpr (exec == ExecutionModel::GPU) {
        return thrust::device_pointer_cast(ptr);
      } else {
        static_assert(exec == ExecutionModel::CPU);
        return ptr;
      }
    };

    auto output_v = output.get(tag);

    // TODO: dedup with integrate_image misc
    unsigned size = sample_spec.visit_tagged([&](auto tag, const auto &spec) {
      if constexpr (tag == SampleSpecType::SquareImage) {
        return spec.x_dim * spec.y_dim;
      } else {
        return spec.size();
      }
    });

    auto copy_to_out = [&](auto span_in, auto span_out) {
      thrust::copy(get_ptr(span_in.begin()), get_ptr(span_in.begin() + size),
                   span_out.begin());
    };

    if constexpr (tag == OutputType::BGRA || tag == OutputType::FloatRGB) {
      copy_to_out(intermediate_output_v, output_v);
    } else {
      static_assert(tag == OutputType::OutputPerStep);

      always_assert(intermediate_output_v.size() == output_v.size());

      for (unsigned i = 0; i < intermediate_output_v.size(); ++i) {
        copy_to_out(intermediate_output_v[i], output_v[i]);
      }
    }
  });

  run_timer.stop();

  return run_timer.elapsed();
}
} // namespace render
