#pragma once

#include "compile_time_dispatch/one_per_instance.h"
#include "intersect/accel/accel.h"
#include "render/dir_sampler.h"
#include "render/light_sampler.h"
#include "render/term_prob.h"
#include "rng/rng.h"

#include <tuple>

namespace render {
class CompileTimeSettings {
public:
  using AccelType = intersect::accel::AccelType;
  using RngType = rng::RngType;
  using T = std::tuple<AccelType, AccelType, LightSamplerType, DirSamplerType,
                       TermProbType, RngType>;

  constexpr CompileTimeSettings(const T &v) : values_(v) {}

  template <typename... Vals>
  constexpr CompileTimeSettings(Vals &&... vals) : values_({vals...}) {}

  constexpr AccelType triangle_accel_type() const {
    return std::get<0>(values_);
  }

  AccelType &triangle_accel_type() { return std::get<0>(values_); }

  constexpr AccelType mesh_accel_type() const { return std::get<1>(values_); }

  AccelType &mesh_accel_type() { return std::get<1>(values_); }

  constexpr LightSamplerType light_sampler_type() const {
    return std::get<2>(values_);
  }

  LightSamplerType &light_sampler_type() { return std::get<2>(values_); }

  constexpr DirSamplerType dir_sampler_type() const {
    return std::get<3>(values_);
  }

  DirSamplerType &dir_sampler_type() { return std::get<3>(values_); }

  constexpr TermProbType term_prob_type() const { return std::get<4>(values_); }

  TermProbType &term_prob_type() { return std::get<4>(values_); }

  constexpr RngType rng_type() const { return std::get<5>(values_); }

  RngType &rng_type() { return std::get<5>(values_); }

  constexpr const T &values() const { return values_; }

private:
  T values_;
};

// TODO: serialization???
struct Settings {
private:
  // default should be pretty reasonable...
  using AccelType = intersect::accel::AccelType;
  using RngType = rng::RngType;

  template <AccelType type>
  using AccelSettings = typename intersect::accel::AccelSettings<type>;

  using AllAccelSettings = OnePerInstance<AccelType, AccelSettings>;

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

  AllAccelSettings default_triangle_accel = {default_loop_all_triangle,
                                             default_kd_tree_triangle,
                                             default_dir_tree_triangle};

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

  AllAccelSettings default_mesh_accel = {
      default_loop_all_mesh, default_kd_tree_mesh, default_dir_tree_mesh};

  using AllLightSamplerSettings =
      OnePerInstance<LightSamplerType, LightSamplerSettings>;

  static constexpr LightSamplerSettings<LightSamplerType::NoDirectLighting>
      default_no_direct_light;

  static constexpr LightSamplerSettings<LightSamplerType::WeightedAABB>
      default_weighted_aabb;

  AllLightSamplerSettings default_light_sampler = {default_no_direct_light,
                                                   default_weighted_aabb};

  using AllDirSamplerSettings =
      OnePerInstance<DirSamplerType, DirSamplerSettings>;

  static constexpr DirSamplerSettings<DirSamplerType::Uniform>
      default_uniform_sampler;

  static constexpr DirSamplerSettings<DirSamplerType::BRDF>
      default_brdf_sampler;

  AllDirSamplerSettings default_dir_sampler = {default_uniform_sampler,
                                               default_brdf_sampler};

  using AllTermProbSettings = OnePerInstance<TermProbType, TermProbSettings>;

  static constexpr TermProbSettings<TermProbType::Uniform>
      default_uniform_term_prob = {0.1f};

  static constexpr TermProbSettings<TermProbType::MultiplierNorm>
      default_multiplier_norm_term_prob = {0.0f};

  AllTermProbSettings default_term_prob = {default_uniform_term_prob,
                                           default_multiplier_norm_term_prob};

  using AllRngSettings = OnePerInstance<RngType, rng::RngSettings>;

  static constexpr rng::RngSettings<RngType::Uniform> default_rng_uniform = {};

  AllRngSettings default_rng = {default_rng_uniform};

public:
  AllAccelSettings triangle_accel = default_triangle_accel;

  AllAccelSettings mesh_accel = default_mesh_accel;

  AllLightSamplerSettings light_sampler = default_light_sampler;

  AllDirSamplerSettings dir_sampler = default_dir_sampler;

  AllTermProbSettings term_prob = default_term_prob;

  AllRngSettings rng = default_rng;

  CompileTimeSettings compile_time;
};
} // namespace render
