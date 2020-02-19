#include "intersect/accel/loop_all.h"
#include "lib/group.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/timer.h"
#include "render/detail/divide_work.h"
#include "render/detail/renderer_impl.h"
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
    RGBA *pixels, const scene::Scene &s, unsigned x_dim, unsigned y_dim,
    unsigned samples_per, intersect::accel::AcceleratorType mesh_accel_type,
    intersect::accel::AcceleratorType triangle_accel_type, bool show_times) {
  unsigned target_block_size = 512;
  unsigned target_work_per_thread = 4;

  auto division =
      divide_work(samples_per, target_block_size, target_work_per_thread);

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

        intersect::accel::run_over_accelerator_types([&](auto &&i) {
          constexpr auto mesh_accel_type = std::decay_t<decltype(i)>::value;

          using MeshInstanceRef = intersect::accel::MeshInstanceRef<TriRefType>;
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

          const auto& aabb = s.overall_aabb();

          auto mesh_instance_accel_ref =
              generator.gen(instance_refs, 0, num_mesh_instances,
                            aabb.get_min_bound(), aabb.get_max_bound(), mesh_accel_settings);
        }, mesh_accel_type);
      },
      triangle_accel_type);

#if 0
  const unsigned num_shapes = scene_->getNumShapes();
  ManangedMemVec<scene::ShapeData> moved_shapes_(num_shapes);

  {
    auto start_shape = scene_->getShapes();
    std::copy(start_shape, start_shape + num_shapes, moved_shapes_.begin());
  }

  SpanSized<const scene::Light> lights_span(lights, num_lights);

  accel::dir_tree::DirTreeLookup dir_tree_lookup;

  if (use_dir_tree) {
    unsigned target_num_dir_trees = 16;
    dir_tree_lookup = dir_tree_generator_.generate(
        moved_shapes_, target_num_dir_trees, scene_->getMinBound(),
        scene_->getMaxBound(), show_times_);
  } else if (use_kd_tree) {
    Timer kdtree_timer;

    auto kdtree = accel::kdtree::construct_kd_tree(moved_shapes_.data(),
                                                   num_shapes, 25, 3);
    kdtree_nodes_.resize(kdtree.size());
    std::copy(kdtree.begin(), kdtree.end(), kdtree_nodes_.begin());

    std::fill(group_disables_.begin(), group_disables_.end(), false);

    if (show_times_) {
      kdtree_timer.report("kdtree");
    }
  }

  for (unsigned depth = 0; depth < recursive_iterations_; depth++) {
    bool is_first = depth == 0;

    current_num_blocks = 0;

    for (unsigned i = 0; i < group_disables_.size(); i++) {
      if (!group_disables_[i]) {
        group_indexes_[current_num_blocks] = i;
        current_num_blocks++;
      }
    }

    Timer intersect_timer;

    Span textures_span(textures, num_textures);

    auto raytrace = [&](const auto &data_structure) {
      if (is_first) {
        raytrace_pass<true>(data_structure, current_num_blocks, moved_shapes_,
                            lights_span, textures_span);
      } else {
        raytrace_pass<false>(data_structure, current_num_blocks, moved_shapes_,
                             lights_span, textures_span);
      }
    };

    if (use_dir_tree) {
      raytrace(accel::dir_tree::DirTreeLookupRef(dir_tree_lookup));
    } else if (use_kd_tree) {
      raytrace(accel::kdtree::KDTreeRef(kdtree_nodes_, moved_shapes_.size()));
    } else {
      raytrace(accel::LoopAll(num_shapes));
    }

    if (show_times) {
      intersect_timer.report("intersect");
    }
  }

  float_to_bgra(pixels, colors_);
#endif
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace render
