#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/transformed_object.h"
#include "lib/binary_search.h"
#include "lib/group.h"
#include "lib/projection.h"
#include "lib/span.h"
#include "material/material.h"
#include "render/light_sampler.h"
#include "rng/rng.h"
#include "rng/test_rng_state_type.h"
#include "scene/emissive_group.h"

#include <Eigen/Core>

#include <array>

namespace render {
namespace detail {
template <LightSamplerType type, ExecutionModel execution_model>
struct LightSamplerImpl;

template <unsigned n> struct LightSamples {
  std::array<DirSample, n> samples;
  unsigned num_samples;
};

template <typename V>
concept LightSamplerRef = requires(const V &light_sampler,
                                   const Eigen::Vector3f &position,
                                   const material::Material &material,
                                   const Eigen::Vector3f &incoming_dir,
                                   const Eigen::Vector3f &normal,
                                   rng::TestRngStateT &rng) {
  V::max_sample_size;
  V::performs_samples;

  { light_sampler(position, material, incoming_dir, normal, rng) }
  ->std::common_with<LightSamples<V::max_sample_size>>;
};

template <LightSamplerType type, ExecutionModel execution_model>
concept LightSampler = requires {
  typename LightSamplerSettings<type>;
  typename LightSamplerImpl<type, execution_model>;

  requires requires(
      LightSamplerImpl<type, execution_model> & light_sampler,
      const LightSamplerSettings<type> &settings,
      Span<const scene::EmissiveGroup> emissive_groups,
      Span<const unsigned> emissive_group_ends_per_mesh,
      Span<const material::Material> materials,
      SpanSized<const intersect::TransformedObject> transformed_objects,
      Span<const intersect::Triangle> triangles) {
    {
      light_sampler.gen(settings, emissive_groups, emissive_group_ends_per_mesh,
                        materials, transformed_objects, triangles)
    }
    ->LightSamplerRef;
  };
};

template <LightSamplerType type, ExecutionModel execution_model>
requires LightSampler<type, execution_model> struct LightSamplerT
    : LightSamplerImpl<type, execution_model> {
  using LightSamplerImpl<type, execution_model>::LightSamplerImpl;
};

template <ExecutionModel execution_model>
struct LightSamplerImpl<LightSamplerType::NoLightSampling, execution_model> {
public:
  using Settings = LightSamplerSettings<LightSamplerType::NoLightSampling>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    static const unsigned max_sample_size = 0;
    static const bool performs_samples = false;

    template <rng::RngState R>
    HOST_DEVICE LightSamples<max_sample_size>
    operator()(const Eigen::Vector3f &, const material::Material &,
               const Eigen::Vector3f &, const Eigen::Vector3f &, R &) const {
      return {{}, 0};
    }
  };

  auto gen(const Settings &settings, Span<const scene::EmissiveGroup>,
           Span<const unsigned>, Span<const material::Material>,
           SpanSized<const intersect::TransformedObject>,
           Span<const intersect::Triangle>) {
    return Ref(settings);
  }
};

template <ExecutionModel execution_model>
struct LightSamplerImpl<LightSamplerType::WeightedAABB, execution_model> {
public:
  using Settings = LightSamplerSettings<LightSamplerType::WeightedAABB>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &, Span<const intersect::accel::AABB> aabbs,
                    SpanSized<const float> cumulative_weights)
        : aabbs_(aabbs), cumulative_weights_(cumulative_weights) {}

    static const unsigned max_sample_size = 1;
    static const bool performs_samples = true;

    template <rng::RngState R>
    HOST_DEVICE LightSamples<max_sample_size>
    operator()(const Eigen::Vector3f &position,
               const material::Material &material, const Eigen::Vector3f &,
               const Eigen::Vector3f &normal, R &rng) const {
      if (cumulative_weights_.size() == 0) {
        return LightSamples<max_sample_size>{{}, 0};
      }

      // TODO: SPEED, complexity...
      unsigned sample_idx = binary_search<float>(
          0, cumulative_weights_.size(), rng.next(), cumulative_weights_);
      assert(sample_idx < cumulative_weights_.size());

      const auto &aabb = aabbs_[sample_idx];
      const auto &min_bound = aabb.get_min_bound();
      const auto &max_bound = aabb.get_max_bound();

      const auto transform = find_rotate_vector_to_vector(normal, {0, 0, 1});

      float min_theta = std::numeric_limits<float>::max();
      float max_theta = std::numeric_limits<float>::lowest();
      float start_phi = std::numeric_limits<float>::max();
      float end_phi = std::numeric_limits<float>::lowest();

      auto angle_of_world_space_direction =
          [&](const auto &world_space_direction) {
            auto normal_is_up_space_direction =
                transform * world_space_direction;
            auto dir = normal_is_up_space_direction.normalized().eval();

            float theta = std::acos(dir.z());
            float phi = std::atan2(dir.y(), dir.x());

            return std::make_tuple(theta, phi);
          };

      bool is_first = true;

      for (bool x_is_min : {false, true}) {
        for (bool y_is_min : {false, true}) {
          for (bool z_is_min : {false, true}) {
            auto get_axis = [&](bool is_min, uint8_t axis) {
              return is_min ? min_bound[axis] : max_bound[axis];
            };
            // TODO: check
            auto point_on_aabb =
                Eigen::Vector3f(get_axis(x_is_min, 0), get_axis(y_is_min, 1),
                                get_axis(z_is_min, 2));
            // TODO: is normalization needed here?
            auto world_space_direction =
                (point_on_aabb - position).normalized();

            auto [theta, phi] =
                angle_of_world_space_direction(world_space_direction);

            min_theta = std::min(min_theta, theta);
            max_theta = std::max(max_theta, theta);

            if (is_first) {
              start_phi = end_phi = phi;
              is_first = false;
            }

            float phi_if_above_start = phi >= start_phi ? phi : phi + 2 * M_PI;
            float phi_if_below_start = phi <= start_phi ? phi : phi - 2 * M_PI;
            float new_end_if_above = std::max(phi_if_above_start, end_phi);
            float new_start_if_below = std::min(phi_if_below_start, start_phi);
            float arc_len_if_above = new_end_if_above - start_phi;
            float arc_len_if_below = end_phi - new_start_if_below;
            if (arc_len_if_above < arc_len_if_below) {
              end_phi = new_end_if_above;
            } else {
              start_phi = new_start_if_below;
            }
          }
        }
      }

      if (end_phi - start_phi > M_PI) {
        start_phi = -M_PI;
        end_phi = M_PI;
      }

      min_theta = 0; // unfortunate approximation

      float max_possible_theta =
          float(material.is_bsdf() ? M_PI : M_PI / 2) - 1e-5f;

      assert(min_theta <= float(M_PI));
      assert(max_theta <= float(M_PI));

      if (min_theta > max_possible_theta) {
        return {{{}}, 0};
      }

      max_theta = std::min(max_theta, max_possible_theta);

      float region_area =
          (-std::cos(max_theta) + std::cos(min_theta)) * (end_phi - start_phi);

      // deal with wrapping:

      float v0 = rng.next();
      float v1 = rng.next();

      float phi = start_phi + (end_phi - start_phi) * v0;
      float theta = min_theta + (max_theta - min_theta) * v1;

      if (phi < 0) {
        phi += 2 * M_PI;
      }

      if (phi > 2 * M_PI) {
        phi -= 2 * M_PI;
      }

      assert(phi >= 0);
      assert(phi <= 2 * M_PI);

      return {std::array<DirSample, max_sample_size>{
                  {{find_relative_vec(normal, phi, theta), 1 / region_area}}},
              1};
    }

  private:
    Span<const intersect::accel::AABB> aabbs_;
    SpanSized<const float> cumulative_weights_;
  };

  auto gen(const Settings &settings,
           Span<const scene::EmissiveGroup> emissive_groups,
           Span<const unsigned> emissive_group_ends_per_mesh,
           Span<const material::Material> materials,
           SpanSized<const intersect::TransformedObject> transformed_objects,
           Span<const intersect::Triangle>) {
    std::vector<float> cumulative_weights;
    std::vector<intersect::accel::AABB> aabbs;
    float cumulative_weight = 0.0f;
    for (const auto &transformed_object : transformed_objects) {
      auto [start, end] = group_start_end(transformed_object.idx(),
                                          emissive_group_ends_per_mesh);
      for (unsigned i = start; i < end; i++) {
        // weight is proportional to intensity and volume
        auto group = emissive_groups[i];
        cumulative_weight += group.aabb.surface_area() *
                             materials[group.material_idx].emission().sum();
        cumulative_weights.push_back(cumulative_weight);
        aabbs.push_back(
            group.aabb.transform(transformed_object.object_to_world()));
      }
    }

    // normalize
    for (auto &weight : cumulative_weights) {
      weight /= cumulative_weight;
    }

    unsigned size = aabbs.size();

    aabbs_.resize(size);
    cumulative_weights_.resize(size);

    thrust::copy(aabbs.data(), aabbs.data() + size, aabbs_.begin());

    thrust::copy(cumulative_weights.data(), cumulative_weights.data() + size,
                 cumulative_weights_.begin());

    return Ref(settings, aabbs_, cumulative_weights_);
  }

private:
  template <typename T> using ExecVecT = ExecVector<execution_model, T>;

  ExecVecT<intersect::accel::AABB> aabbs_; // in world space
  ExecVecT<float> cumulative_weights_;
};

template <ExecutionModel execution_model>
struct LightSamplerImpl<LightSamplerType::RandomTriangle, execution_model> {
public:
  using Settings = LightSamplerSettings<LightSamplerType::RandomTriangle>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    // SPEED: don't store triangles in here, use indices?
    HOST_DEVICE Ref(const Settings &, Span<const intersect::Triangle> triangles,
                    SpanSized<const float> cumulative_weights)
        : triangles_(triangles), cumulative_weights_(cumulative_weights) {}

    static const unsigned max_sample_size = 1;
    static const bool performs_samples = true;

    template <rng::RngState R>
    HOST_DEVICE LightSamples<max_sample_size>
    operator()(const Eigen::Vector3f &position, const material::Material &,
               const Eigen::Vector3f &, const Eigen::Vector3f &normal,
               R &rng) const {
      if (cumulative_weights_.size() == 0) {
        return LightSamples<max_sample_size>{{}, 0};
      }

      // TODO: SPEED, complexity...
      unsigned sample_idx = binary_search<float>(
          0, cumulative_weights_.size(), rng.next(), cumulative_weights_);
      assert(sample_idx < cumulative_weights_.size());

      const auto &triangle = triangles_[sample_idx];

      float weight0 = rng.next();
      float weight1 = rng.next();
      float weight2 = rng.next();

      const auto &vertices = triangle.vertices();

      Eigen::Vector3f point =
          vertices[0] * weight0 + vertices[1] * weight1 + vertices[2] * weight2;

      Eigen::Vector3f direction_unnormalized = point - position;
      Eigen::Vector3f direction = direction_unnormalized.normalized();

      Eigen::Vector3f triangle_normal =
          ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]))
              .normalized();

      float prob =
          std::abs(normal.dot(direction) * triangle_normal.dot(direction)) /
          direction_unnormalized.squaredNorm();

      return {std::array<DirSample, max_sample_size>{{{direction, 1 / prob}}},
              1};
    }

  private:
    Span<const intersect::Triangle> triangles_;
    SpanSized<const float> cumulative_weights_;
  };

  auto gen(const Settings &settings,
           Span<const scene::EmissiveGroup> emissive_groups,
           Span<const unsigned> emissive_group_ends_per_mesh,
           Span<const material::Material> materials,
           SpanSized<const intersect::TransformedObject> transformed_objects,
           Span<const intersect::Triangle> triangles) {
    std::vector<float> cumulative_weights;
    std::vector<intersect::Triangle> transformed_triangles;
    float cumulative_weight = 0.0f;
    for (const auto &transformed_object : transformed_objects) {
      auto [start, end] = group_start_end(transformed_object.idx(),
                                          emissive_group_ends_per_mesh);
      for (unsigned i = start; i < end; i++) {
        auto group = emissive_groups[i];
        const auto &transform = transformed_object.object_to_world();
        for (unsigned triangle_idx = group.start_idx;
             triangle_idx < group.end_idx; triangle_idx++) {
          // weight is proportional to intensity and volume
          //
          const auto &triangle = triangles[triangle_idx];
          const auto &orig_vertices = triangle.vertices();
          intersect::Triangle transformed_triangle = {
              {transform * orig_vertices[0], transform * orig_vertices[1],
               transform * orig_vertices[2]}};
          const auto &vertices = transformed_triangle.vertices();

          Eigen::Vector3f edge_0 = vertices[1] - vertices[0];
          Eigen::Vector3f edge_1 = vertices[2] - vertices[0];

          float cos_between = edge_0.normalized().dot(edge_1.normalized());
          float surface_area = edge_0.norm() * edge_1.norm() *
                               std::sqrt(1 - cos_between * cos_between);

          cumulative_weight +=
              surface_area * materials[group.material_idx].emission().sum();
          cumulative_weights.push_back(cumulative_weight);
          transformed_triangles.push_back(transformed_triangle);
        }
      }
    }

    // normalize
    for (auto &weight : cumulative_weights) {
      weight /= cumulative_weight;
    }

    unsigned size = transformed_triangles.size();

    triangles_.resize(size);
    cumulative_weights_.resize(size);

    thrust::copy(transformed_triangles.data(),
                 transformed_triangles.data() + size, triangles_.begin());

    thrust::copy(cumulative_weights.data(), cumulative_weights.data() + size,
                 cumulative_weights_.begin());

    return Ref(settings, triangles_, cumulative_weights_);
  }

private:
  template <typename T> using ExecVecT = ExecVector<execution_model, T>;

  ExecVecT<intersect::Triangle> triangles_; // in world space
  ExecVecT<float> cumulative_weights_;
};
} // namespace detail
} // namespace render
