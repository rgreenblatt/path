#pragma once

#include "intersect/accel/accelerator_type.h"
#include "intersect/accel/accelerator_type_settings.h"
#include "lib/compile_time_dispatch/enum.h"
#include "lib/compile_time_dispatch/one_per_instance.h"

#include <tuple>

namespace render {
class CompileTimePerfSettings {
public:
  using AcceleratorType = intersect::accel::AcceleratorType;
  using T = std::tuple<AcceleratorType, AcceleratorType>;

  constexpr CompileTimePerfSettings(const T &v) : values_(v) {}

  template <typename... Vals>
  constexpr CompileTimePerfSettings(Vals &&... vals) : values_({vals...}) {}

  constexpr AcceleratorType triangle_accel_type() const {
    return std::get<0>(values_);
  }

  constexpr AcceleratorType mesh_accel_type() const {
    return std::get<0>(values_);
  }

  constexpr const T &values() const { return values_; }

private:
  T values_;
};

struct PerfSettings {
private:
  // default should be pretty reasonable...
  using AccelType = intersect::accel::AcceleratorType;

  template <AccelType t> using AccelSettings = intersect::accel::Settings<t>;

  static constexpr AccelSettings<AccelType::LoopAll> default_loop_all_triangle;

  static constexpr float default_kd_tree_triangle_traversal_cost = 1;
  static constexpr float default_kd_tree_triangle_intersection_cost = 1;

  static constexpr AccelSettings<AccelType::KDTree> default_kd_tree_triangle = {
      {default_kd_tree_triangle_traversal_cost,
       default_kd_tree_triangle_intersection_cost}};

  static constexpr float default_dir_tree_triangle_traversal_cost = 1;
  static constexpr float default_dir_tree_triangle_intersection_cost = 1;
  static constexpr unsigned default_num_dir_trees_triangle = 16;

  static constexpr AccelSettings<AccelType::DirTree> default_dir_tree_triangle =
      {{default_dir_tree_triangle_traversal_cost,
        default_dir_tree_triangle_intersection_cost},
       default_num_dir_trees_triangle};

  static constexpr AccelSettings<AccelType::LoopAll> default_loop_all_mesh;

  static constexpr float default_kd_tree_mesh_traversal_cost = 1;
  static constexpr float default_kd_tree_mesh_intersection_cost = 1;

  static constexpr AccelSettings<AccelType::KDTree> default_kd_tree_mesh = {
      {default_kd_tree_mesh_traversal_cost,
       default_kd_tree_mesh_intersection_cost}};

  static constexpr float default_dir_tree_mesh_traversal_cost = 1;
  static constexpr float default_dir_tree_mesh_intersection_cost = 1;
  static constexpr unsigned default_num_dir_trees_mesh = 16;

  static constexpr AccelSettings<AccelType::DirTree> default_dir_tree_mesh = {
      {default_dir_tree_mesh_traversal_cost,
       default_dir_tree_mesh_intersection_cost},
      default_num_dir_trees_mesh};

public:
  OnePerInstance<AccelType, intersect::accel::Settings> mesh_accel_settings = {
      default_loop_all_mesh, default_kd_tree_mesh, default_dir_tree_mesh};
  OnePerInstance<AccelType, intersect::accel::Settings>
      triangle_accel_settings = {default_loop_all_triangle,
                                 default_kd_tree_triangle,
                                 default_dir_tree_triangle};

  CompileTimePerfSettings compile_time;
};
} // namespace render
