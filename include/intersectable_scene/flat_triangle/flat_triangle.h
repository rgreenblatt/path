#pragma once

#include "data_structure/copyable_to_vec.h"
#include "execution_model/execution_model.h"
#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/accel.h"
#include "intersect/triangle.h"
#include "intersectable_scene/flat_triangle/settings.h"
#include "intersectable_scene/intersectable_scene.h"
#include "intersectable_scene/triangle_generator.h"
#include "lib/cuda/utils.h"
#include "lib/group.h"
#include "scene/scene.h"
#include "scene/triangle_data.h"

namespace intersectable_scene {
namespace flat_triangle {
template <intersect::accel::AccelRef<intersect::Triangle> Accel> struct Ref {
  Accel accel;
  Span<const intersect::Triangle> triangles;
  Span<const scene::TriangleData> triangle_data;
  Span<const material::Material> materials;

  HOST_DEVICE inline intersect::accel::AABB bounds() const {
    return accel.bounds();
  }

  using InfoType = intersect::accel::IdxHolder<intersect::Triangle::InfoType>;

  HOST_DEVICE inline auto intersect(const intersect::Ray &ray) const {
    return accel.intersect_objects(ray, triangles);
  }

  using Intersection = intersect::Intersection<InfoType>;

  HOST_DEVICE inline Eigen::Vector3f
  get_normal(const Intersection &intersection,
             const intersect::Ray &ray) const {
    unsigned triangle_idx = intersection.info.idx;

    return triangle_data[triangle_idx].get_normal(
        intersection.intersection_point(ray), triangles[triangle_idx]);
  }

  HOST_DEVICE inline const material::Material &
  get_material(const Intersection &intersection) const {
    auto [triangle_idx, triangle_info] = intersection.info;

    return materials[triangle_data[triangle_idx].material_idx()];
  }
};

template <
    ExecutionModel exec, Setting AccelSettings,
    intersect::accel::ObjectSpecificAccel<AccelSettings, intersect::Triangle>
        Accel>
class Generator {
public:
  Generator(){};

  using Settings = Settings<AccelSettings>;

  auto gen(const Settings &settings, const scene::Scene &scene) {
    host_triangles_.clear();
    host_triangle_data_.clear();

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
        host_triangles_.push_back(scene_triangles[j].transform(transform));
        host_triangle_data_.push_back(scene_triangle_data[j]);
      }
    }

    copy_to_vec(host_triangles_, triangles_);
    copy_to_vec(host_triangle_data_, triangle_data_);
    copy_to_vec(scene.materials(), materials_);

    auto accel_ref = accel_.template gen<intersect::Triangle>(
        settings.accel_settings, host_triangles_, scene.overall_aabb());

    return Ref<std::decay_t<decltype(accel_ref)>>{accel_ref, triangles_,
                                                  triangle_data_, materials_};
  }

private:
  template <typename T> using ExecVecT = ExecVector<exec, T>;
  std::vector<intersect::Triangle> host_triangles_;
  std::vector<scene::TriangleData> host_triangle_data_;
  ExecVecT<intersect::Triangle> triangles_;
  ExecVecT<scene::TriangleData> triangle_data_;
  ExecVecT<material::Material> materials_;
  Accel accel_;
};
} // namespace flat_triangle
} // namespace intersectable_scene
