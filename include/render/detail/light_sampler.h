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

    static constexpr unsigned max_sample_size = 0;
    static constexpr bool performs_samples = false;

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

    static constexpr unsigned max_sample_size = 1;
    static constexpr bool performs_samples = true;

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

      if (weight0 + weight1 > 1.f) {
        weight0 = 1 - weight0;
        weight1 = 1 - weight1;
      }

      const auto &vertices = triangle.vertices();

      // SPEED: cache vecs?
      auto vec0 = vertices[1] - vertices[0];
      auto vec1 = vertices[2] - vertices[0];

      Eigen::Vector3f point = vertices[0] + vec0 * weight0 + vec1 * weight1;

      Eigen::Vector3f direction_unnormalized = point - position;
      Eigen::Vector3f direction = direction_unnormalized.normalized();

      // SPEED: cache normal?
      // scaled by area
      Eigen::Vector3f triangle_normal =
          0.5 * ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]));

      float prob_this_triangle =
          cumulative_weights_[sample_idx] -
          get_previous<const float>(sample_idx, cumulative_weights_);

      float weight =
          std::abs(normal.dot(direction) * triangle_normal.dot(direction)) /
          direction_unnormalized.squaredNorm();

      return {std::array<DirSample, max_sample_size>{
                  {{direction, prob_this_triangle / weight}}},
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
