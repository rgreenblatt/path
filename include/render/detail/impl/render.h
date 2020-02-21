#pragma once

#include "intersect/impl/ray_impl.h"
#include "intersect/impl/triangle_impl.h"
#include "lib/compile_time_dispatch/dispatch_value.h"
#include "lib/compile_time_dispatch/tuple.h"
#include "lib/group.h"
#include "render/detail/compute_intensities.h"
#include "render/detail/dir_sampler_generator.h"
#include "render/detail/divide_work.h"
#include "render/detail/light_sampler_generator.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/term_prob_generator.h"
#include "render/detail/tone_map.h"

namespace render {
namespace detail {
template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(Span<BGRA> pixels,
                                           const scene::Scene &s,
                                           unsigned samples_per, unsigned x_dim,
                                           unsigned y_dim,
                                           const Settings &settings, bool) {
  if (samples_per > std::numeric_limits<uint16_t>::max()) {
    std::cerr << "more samples than allowed" << std::endl;
    return;
  }

  unsigned block_size = 512;
  unsigned target_work_per_thread = 4;

  auto division = divide_work(samples_per, x_dim, y_dim, block_size,
                              target_work_per_thread);

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

        constexpr auto triangle_accel_type =
            compile_time_settings.triangle_accel_type();

        auto &triangle_accels =
            stored_triangle_accels_.template get_item<triangle_accel_type>();

        using Triangle = intersect::Triangle;
        using Generator = intersect::accel::Generator<Triangle, execution_model,
                                                      triangle_accel_type>;
        using TriRefType = typename Generator::RefType;

        unsigned num_meshs = s.mesh_paths().size();

        triangle_accels.reset();

        std::vector<TriRefType> cpu_refs(num_meshs);
        std::vector<uint8_t> ref_set(num_meshs, 0);

        for (unsigned i = 0; i < num_meshs; i++) {
          auto ref_op = triangle_accels.query(s.mesh_paths()[i]);
          if (ref_op.has_value()) {
            cpu_refs[i] = *ref_op;
            ref_set[i] = true;
          }
        }

        assert(s.mesh_aabbs().size() == num_meshs);

        for (unsigned i = 0; i < num_meshs; i++) {
          if (!ref_set[i]) {
            const auto &aabb = s.mesh_aabbs()[i];
            cpu_refs[i] = triangle_accels.add(
                s.triangles(), get_previous(i, s.mesh_ends()), s.mesh_ends()[i],
                aabb.get_min_bound(), aabb.get_max_bound(),
                settings.triangle_accel
                    .template get_item<triangle_accel_type>());
          }
        }

        ExecVecT<TriRefType> refs(cpu_refs.begin(), cpu_refs.end());

        constexpr auto mesh_accel_type =
            compile_time_settings.mesh_accel_type();

        using MeshInstanceRef = intersect::MeshInstanceRef<TriRefType>;
        using MeshGenerator =
            intersect::accel::Generator<MeshInstanceRef, execution_model,
                                        mesh_accel_type>;

        MeshGenerator generator;

        unsigned num_mesh_instances = s.mesh_instances().size();

        std::vector<MeshInstanceRef> instance_refs(num_mesh_instances);

        for (unsigned i = 0; i < num_mesh_instances; ++i) {
          instance_refs[i] =
              s.mesh_instances()[i].get_ref(Span<const TriRefType>{refs});
        }

        const auto &aabb = s.overall_aabb();

        auto mesh_instance_accel_ref = generator.gen(
            instance_refs, 0, num_mesh_instances, aabb.get_min_bound(),
            aabb.get_max_bound(),
            settings.mesh_accel.template get_item<mesh_accel_type>());

        constexpr auto light_sampler_type =
            compile_time_settings.light_sampler_type();
        constexpr auto dir_sampler_type =
            compile_time_settings.dir_sampler_type();
        constexpr auto term_prob_type = compile_time_settings.term_prob_type();

        auto light_sampler =
            LightSamplerGenerator<execution_model, light_sampler_type>().gen(
                settings.light_sampler.template get_item<light_sampler_type>());

        auto dir_sampler =
            DirSamplerGenerator<execution_model, dir_sampler_type>().gen(
                settings.dir_sampler.template get_item<dir_sampler_type>());

        auto term_prob =
            TermProbGenerator<execution_model, term_prob_type>().gen(
                settings.term_prob.template get_item<term_prob_type>());

        compute_intensities<execution_model>(
            division, samples_per, x_dim, y_dim, block_size,
            mesh_instance_accel_ref, light_sampler, dir_sampler, term_prob,
            output_pixels, intensities_, triangle_data, materials,
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
