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
#include "lib/group.h"
#include "lib/vector_group.h"
#include "scene/scene.h"
#include "scene/triangle_data.h"

namespace intersectable_scene {
namespace flat_triangle {
namespace detail {
template <intersect::accel::AccelRef Accel> struct Ref {
  using B = scene::Material::BSDFT;

  Accel accel;
  Span<const intersect::Triangle> triangles;
  Span<const scene::TriangleData> triangle_data;
  Span<const scene::Material> materials;

  HOST_DEVICE inline intersect::accel::AABB bounds() const {
    return accel.bounds();
  }

  using InfoType = intersect::accel::IdxHolder<intersect::Triangle::InfoType>;

  HOST_DEVICE inline auto intersect(const intersect::Ray &ray) const {
    return accel.intersect_objects(
        ray, [&](unsigned idx, const intersect::Ray &ray) {
          return triangles[idx].intersect(ray);
        });
  }

  using Intersection = intersect::Intersection<InfoType>;

  ATTR_PURE_NDEBUG HOST_DEVICE inline UnitVector
  get_normal(const Intersection &intersection,
             const intersect::Ray &ray) const {
    unsigned triangle_idx = intersection.info.idx;

    return triangle_data[triangle_idx].get_normal(
        intersection.intersection_point(ray), triangles[triangle_idx]);
  }

  ATTR_PURE_NDEBUG HOST_DEVICE inline const scene::Material &
  get_material(const Intersection &intersection) const {
    auto [triangle_idx, triangle_info] = intersection.info;

    return materials[triangle_data[triangle_idx].material_idx()];
  }
};

enum class TriItem {
  Triangle,
  Data,
};
} // namespace detail

template <
    ExecutionModel exec, Setting AccelSettings,
    intersect::accel::ObjectSpecificAccel<AccelSettings, intersect::Triangle>
        Accel>
class Generator {
public:
  Generator(){};

  using Settings = Settings<AccelSettings>;

  auto gen(const Settings &settings, const scene::Scene &scene) {
    host_triangle_values_.clear_all();

    auto objects = scene.transformed_mesh_objects();
    auto scene_triangles = scene.triangles();
    auto scene_triangle_data = scene.triangle_data();
    auto object_mesh_idxs = scene.transformed_mesh_idxs();
    auto mesh_ends = scene.mesh_ends();

    for (unsigned i = 0; i < objects.size(); ++i) {
      unsigned mesh_idx = object_mesh_idxs[i];
      const auto &transform = objects[i].object_to_world();
      for (unsigned j = get_previous<unsigned>(mesh_idx, mesh_ends);
           j < mesh_ends[i]; ++j) {
        host_triangle_values_.push_back_all(
            scene_triangles[j].transform(transform), scene_triangle_data[j]);
      }
    }

    host_triangle_values_.copy_to_other(triangle_values_);
    copy_to_vec(scene.materials(), materials_);

    auto accel_ref = accel_.template gen<intersect::Triangle>(
        settings.accel_settings,
        host_triangle_values_.template get<TriItem::Triangle>(),
        scene.overall_aabb());

    return detail::Ref<std::decay_t<decltype(accel_ref)>>{
        accel_ref, triangle_values_.template get<TriItem::Triangle>(),
        triangle_values_.template get<TriItem::Data>(), materials_};
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

// TODO: consider implementing later...
#if 0
template <ExecutionModel exec>
struct IsSceneGenerator
    : BoolWrapper<SceneGenerator<Generator<exec, MockAccelSettings, MockAccel>,
                                 Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsSceneGenerator>);
#endif
} // namespace flat_triangle
} // namespace intersectable_scene
