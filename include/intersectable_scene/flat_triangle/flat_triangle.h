#pragma once

#include "data_structure/copyable_to_vec.h"
#include "execution_model/execution_model.h"
#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/accel.h"
#include "intersect/triangle.h"
#include "intersectable_scene/flat_triangle/settings.h"
#include "intersectable_scene/intersectable_scene.h"
#include "intersectable_scene/scene_generator.h"
#include "lib/cuda/utils.h"
#include "lib/edges.h"
#include "lib/vector_group.h"
#include "meta/all_values_enum.h"
#include "scene/scene.h"
#include "scene/triangle_data.h"

namespace intersectable_scene {
namespace flat_triangle {
namespace detail {
using InfoType = intersect::accel::IdxHolder<intersect::Triangle::InfoType>;
using Intersection = intersect::Intersection<InfoType>;

enum class TriItem {
  Triangle,
  Data,
};
} // namespace detail

struct SceneRef {
  Span<const intersect::Triangle> triangles;
  Span<const scene::TriangleData> triangle_data;
  Span<const scene::Material> materials;

  using B = scene::Material::BSDFT;
  using InfoType = detail::InfoType;

  ATTR_PURE_NDEBUG HOST_DEVICE inline UnitVector
  get_normal(const detail::Intersection &intersection,
             const intersect::Ray &ray) const {
    unsigned triangle_idx = intersection.info.idx;

    return triangle_data[triangle_idx].get_normal(
        intersection.intersection_point(ray), triangles[triangle_idx]);
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline const scene::Material &
  get_material(const detail::Intersection &intersection) const {
    auto [triangle_idx, triangle_info] = intersection.info;

    return materials[triangle_data[triangle_idx].material_idx()];
  }
};

template <intersect::accel::AccelRef Accel> struct IntersectableRef {
  [[no_unique_address]] Accel accel;
  Span<const intersect::Triangle> triangles;

  static constexpr bool individually_intersectable = true;

  using InfoType = detail::InfoType;

  ATTR_PURE_NDEBUG HOST_DEVICE inline intersect::IntersectionOp<InfoType>
  intersect(const intersect::Ray &ray) const {
    return accel.intersect_objects(
        ray, [&](unsigned idx, const intersect::Ray &ray) {
          return triangles[idx].intersect(ray);
        });
  }
};

template <
    ExecutionModel exec, Setting AccelSettings,
    intersect::accel::ObjectSpecificAccel<AccelSettings, intersect::Triangle>
        Accel>
class Generator {
public:
  Generator(){};

  using Settings = flat_triangle::Settings<AccelSettings>;
  using Intersector = flat_triangle::IntersectableRef<typename Accel::Ref>;
  using SceneRef = flat_triangle::SceneRef;

  IntersectableScene<Intersector, SceneRef> gen(const Settings &settings,
                                                const scene::Scene &scene) {
    host_triangle_values_.clear_all();

    auto objects = scene.transformed_mesh_objects();
    auto scene_triangles = scene.triangles();
    auto scene_triangle_data = scene.triangle_data();
    auto object_mesh_idxs = scene.transformed_mesh_idxs();
    auto mesh_ends = scene.mesh_ends();

    for (unsigned i = 0; i < objects.size(); ++i) {
      unsigned mesh_idx = object_mesh_idxs[i];
      const auto &transform = objects[i].object_to_world();
      for (unsigned j = edges_get_previous(mesh_idx, mesh_ends.as_unsized());
           j < mesh_ends[i]; ++j) {
        host_triangle_values_.push_back_all(
            scene_triangles[j].transform(transform), scene_triangle_data[j]);
      }
    }

    host_triangle_values_.copy_to_other(triangle_values_);
    copy_to_vec(scene.materials(), materials_);

    auto triangles = triangle_values_.get(TagV<TriItem::Triangle>);

    return {
        .intersector =
            {
                .accel = accel_.gen(
                    settings.accel_settings,
                    host_triangle_values_.get(TagV<TriItem::Triangle>)
                        .as_const(),
                    scene.overall_aabb()),
                .triangles = triangles,
            },
        .scene =
            {
                .triangles = triangles,
                .triangle_data = triangle_values_.get(TagV<TriItem::Data>),
                .materials = materials_,
            },
    };
  }

private:
  template <typename T> using ExecVecT = ExecVector<exec, T>;

  using TriItem = detail::TriItem;

  template <template <typename> class VecT>
  using VectorGroup =
      VectorGroup<VecT, TriItem, intersect::Triangle, scene::TriangleData>;

  VectorGroup<HostVector> host_triangle_values_;
  VectorGroup<ExecVecT> triangle_values_;
  ExecVecT<scene::Material> materials_;
  Accel accel_;
};

#if 0
template <ExecutionModel exec>
struct IsSceneGenerator
    : std::bool_constant<SceneGenerator<Generator<exec, MockAccelSettings, MockAccel>,
                                 Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsSceneGenerator>);
#endif
} // namespace flat_triangle
} // namespace intersectable_scene
