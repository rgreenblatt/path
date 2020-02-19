#include "intersect/accel/loop_all.h"
#include "lib/group.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/timer.h"
#include "render/detail/compute_intensities.h"
#include "render/detail/divide_work.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/tone_map.h"
#include "scene/camera.h"

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/combine.hpp>
#include <thrust/copy.h>
#include <thrust/fill.h>

namespace render {
namespace detail {
template <ExecutionModel execution_model>
RendererImpl<execution_model>::RendererImpl() {}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(
    Span<RGBA> pixels, const scene::Scene &s, unsigned x_dim, unsigned y_dim,
    unsigned samples_per, intersect::accel::AcceleratorType mesh_accel_type,
    intersect::accel::AcceleratorType triangle_accel_type, bool show_times) {
  unsigned target_block_size = 512;
  unsigned target_work_per_thread = 4;

  auto division =
      divide_work(samples_per, target_block_size, target_work_per_thread);
  unsigned required_size = division.blocks_per_pixel * x_dim * y_dim;

  if (division.blocks_per_pixel != 1) {
    intermediate_intensities_.resize(required_size);
  } else {
    intermediate_intensities_.clear();
  }

  final_intensities_.resize(x_dim * y_dim);

  const float dir_tree_triangle_traversal_cost = 1;
  const float dir_tree_triangle_intersection_cost = 1;
  const unsigned num_dir_trees_triangle = 16;

  const float kd_tree_triangle_traversal_cost = 1;
  const float kd_tree_triangle_intersection_cost = 1;

  const float dir_tree_mesh_traversal_cost = 1;
  const float dir_tree_mesh_intersection_cost = 1;
  const unsigned num_dir_trees_mesh = 16;

  const float kd_tree_mesh_traversal_cost = 1;
  const float kd_tree_mesh_intersection_cost = 1;

  intersect::accel::run_over_accelerator_types(
      [&](auto &&i) {
        constexpr auto tri_accel_type = std::decay_t<decltype(i)>::value;
        using AcceleratorType = intersect::accel::AcceleratorType;

        auto &v = stored_mesh_accels_.template get_item<tri_accel_type>();

        using TriRefType = typename std::decay_t<decltype(v)>::RefType;
        using TriSettings = typename std::decay_t<decltype(v)>::Settings;

        TriSettings triangle_accel_settings;
        if constexpr (tri_accel_type == AcceleratorType::DirTree) {
          triangle_accel_settings.s_a_heuristic_settings = {
              dir_tree_triangle_traversal_cost,
              dir_tree_triangle_intersection_cost};
          triangle_accel_settings.num_dir_trees = num_dir_trees_triangle;
        } else if constexpr (tri_accel_type == AcceleratorType::KDTree) {
          triangle_accel_settings.s_a_heuristic_settings = {
              kd_tree_triangle_traversal_cost,
              kd_tree_triangle_intersection_cost};
        }

        unsigned num_meshs = s.mesh_paths().size();

        v.reset();

        std::vector<TriRefType> cpu_refs(num_meshs);
        std::vector<uint8_t> ref_set(num_meshs, 0);

        for (unsigned i = 0; i < num_meshs; i++) {
          auto ref_op = v.query(s.mesh_paths()[i]);
          if (ref_op.has_value()) {
            cpu_refs[i] = *ref_op;
            ref_set[i] = true;
          }
        }

        assert(s.mesh_aabbs().size() == num_meshs);

        for (unsigned i = 0; i < num_meshs; i++) {
          if (!ref_set[i]) {
            const auto &aabb = s.mesh_aabbs()[i];
            cpu_refs[i] = v.add(s.triangles(), get_previous(i, s.mesh_ends()),
                                s.mesh_ends()[i], aabb.get_min_bound(),
                                aabb.get_max_bound());
          }
        }

        ExecVecT<TriRefType> refs(cpu_refs.begin(), cpu_refs.end());

        intersect::accel::run_over_accelerator_types(
            [&](auto &&i) {
              constexpr auto mesh_accel_type = std::decay_t<decltype(i)>::value;

              using MeshInstanceRef =
                  intersect::accel::MeshInstanceRef<TriRefType>;
              using MeshGenerator =
                  intersect::accel::Generator<MeshInstanceRef, execution_model,
                                              mesh_accel_type>;
              using MeshSettings = intersect::accel::Settings<mesh_accel_type>;

              MeshSettings mesh_accel_settings;
              if constexpr (mesh_accel_type == AcceleratorType::DirTree) {
                mesh_accel_settings.s_a_heuristic_settings = {
                    dir_tree_mesh_traversal_cost,
                    dir_tree_mesh_intersection_cost};
                mesh_accel_settings.num_dir_trees = num_dir_trees_mesh;
              } else if constexpr (mesh_accel_type == AcceleratorType::KDTree) {
                mesh_accel_settings.s_a_heuristic_settings = {
                    kd_tree_mesh_traversal_cost,
                    kd_tree_mesh_intersection_cost};
              }

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
                  aabb.get_max_bound(), mesh_accel_settings);

              compute_intensities<execution_model>(
                  division, x_dim, y_dim, mesh_instance_accel_ref,
                  intermediate_intensities_, final_intensities_);
            },
            mesh_accel_type);
      },
      triangle_accel_type);

  Span<RGBA> output_pixels;
  if constexpr (execution_model == ExecutionModel::GPU) {
    bgra_.resize(x_dim * y_dim);
    output_pixels = bgra_;
  } else {
    output_pixels = pixels;
  }

  tone_map<execution_model>(x_dim, y_dim, final_intensities_, output_pixels);

  if constexpr (execution_model == ExecutionModel::GPU) {
    thrust::copy(bgra_.begin(), bgra_.end(), pixels.begin());
  }
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace render
