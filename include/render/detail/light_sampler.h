#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/transformed_object.h"
#include "lib/binary_search.h"
#include "lib/group.h"
#include "lib/optional.h"
#include "lib/projection.h"
#include "lib/span.h"
#include "lib/vector_group.h"
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

struct LightSample {
  DirSample dir_sample;
  float expected_distance;
};

template <unsigned n> struct LightSamples {
  std::array<LightSample, n> samples;
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
      SpanSized<const intersect::TransformedObject> transformed_mesh_objects,
      Span<const unsigned> transformed_mesh_idxs,
      Span<const intersect::Triangle> triangles) {
    {
      light_sampler.gen(settings, emissive_groups, emissive_group_ends_per_mesh,
                        materials, transformed_mesh_objects,
                        transformed_mesh_idxs, triangles)
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
           SpanSized<const intersect::TransformedObject>, Span<const unsigned>,
           Span<const intersect::Triangle>) {
    return Ref(settings);
  }
};

constexpr thrust::optional<unsigned>
search(const float target, SpanSized<const float> values,
       const unsigned binary_search_threshold) {
  thrust::optional<unsigned> solution;

  if (values.size() < binary_search_threshold) {
    for (unsigned i = 0; i < values.size(); ++i) {
      if (values[i] >= target) {
        solution = i;
        break;
      }
    }

  } else {
    // binary search
    // UNIMPLEMENTED...
    assert(false);
  }

  return solution;
}

template <ExecutionModel execution_model>
struct LightSamplerImpl<LightSamplerType::RandomTriangle, execution_model> {
private:
  template <template <typename> class VecT>
  using GroupItems = VectorGroup<VecT, intersect::Triangle, float>;

public:
  using Settings = LightSamplerSettings<LightSamplerType::RandomTriangle>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    static constexpr unsigned max_sample_size = 1;
    static constexpr bool performs_samples = true;

    template <rng::RngState R>
    HOST_DEVICE LightSamples<max_sample_size>
    operator()(const Eigen::Vector3f &position,
               const material::Material & /*mat*/,
               const Eigen::Vector3f & /*incoming_dir*/,
               const Eigen::Vector3f &normal, R &rng) const {
      if (cumulative_weights_.size() == 0) {
        return LightSamples<max_sample_size>{{}, 0};
      }
      const float search_value = rng.next();
      const auto sample_idx_op =
          search(search_value, cumulative_weights_, binary_search_threshold_);

      if (!sample_idx_op.has_value()) {
        return LightSamples<max_sample_size>{{}, 0};
      }

      const unsigned sample_idx = *sample_idx_op;

      // TODO: SPEED, complexity, ...
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
      const auto vec0 = vertices[1] - vertices[0];
      const auto vec1 = vertices[2] - vertices[0];

      const Eigen::Vector3f point =
          vertices[0] + vec0 * weight0 + vec1 * weight1;

      const Eigen::Vector3f direction_unnormalized = point - position;
      const Eigen::Vector3f direction = direction_unnormalized.normalized();

      // SPEED: cache normal?
      // scaled by area
      const Eigen::Vector3f triangle_normal =
          0.5 * ((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]));

      const float prob_this_triangle =
          get_size<float>(sample_idx, cumulative_weights_);

      const float normal_weight =
          abs(normal.dot(direction) * triangle_normal.dot(direction));

      // case where we sample from the current triangle (or a parallel triangle)
      if (normal_weight < 1e-8) {
        return {{}, 0};
      }

      const float weight = normal_weight / direction_unnormalized.squaredNorm();

      const DirSample sample = {direction, prob_this_triangle / weight};

      return {{{{sample, direction_unnormalized.norm()}}}, 1};
    }

  private:
    // SPEED: don't store triangles in here, use indices instead?
    // would requre applying transformation.
    HOST_DEVICE Ref(const Settings &settings,
                    Span<const intersect::Triangle> triangles,
                    SpanSized<const float> cumulative_weights)
        : binary_search_threshold_(settings.binary_search_threshold),
          triangles_(triangles), cumulative_weights_(cumulative_weights) {}

    friend struct LightSamplerImpl;

    unsigned binary_search_threshold_;
    Span<const intersect::Triangle> triangles_;
    SpanSized<const float> cumulative_weights_;
  };

  auto
  gen(const Settings &settings,
      Span<const scene::EmissiveGroup> emissive_groups,
      Span<const unsigned> emissive_group_ends_per_mesh,
      Span<const material::Material> materials,
      SpanSized<const intersect::TransformedObject> transformed_mesh_objects,
      Span<const unsigned> transformed_mesh_idxs,
      Span<const intersect::Triangle> triangles) {
    GroupItems<HostVector> items;
    float cumulative_weight = 0.0f;
    for (unsigned object_idx = 0; object_idx < transformed_mesh_objects.size();
         ++object_idx) {
      const auto &transformed_object = transformed_mesh_objects[object_idx];
      auto [start, end] = group_start_end(transformed_mesh_idxs[object_idx],
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
              surface_area * materials[group.material_idx].emission.sum();
          items.push_back_all(transformed_triangle, cumulative_weight);
        }
      }
    }

    // normalize
    for (auto &weight : items.get<1>()) {
      weight /= cumulative_weight;
    }
    items.copy_to_other(items_);

    assert(items.size() == items_.size());

    return Ref(settings, items_.template get<0>(), items_.template get<1>());
  }

private:
  template <typename T> using ExecVecT = ExecVector<execution_model, T>;

  GroupItems<ExecVecT> items_;
};
} // namespace detail
} // namespace render
