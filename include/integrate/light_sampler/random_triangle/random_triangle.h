#pragma once

#include "execution_model/execution_model.h"
#include "execution_model/host_vector.h"
#include "integrate/light_sampler/random_triangle/settings.h"
#include "integrate/light_sampler/triangle_light_sampler.h"
#include "lib/assert.h"
#include "lib/edges.h"
#include "lib/optional.h"
#include "lib/vector_group.h"
#include "meta/predicate_for_all_values.h"

#include <memory>

namespace integrate {
namespace light_sampler {
namespace random_triangle {
namespace detail {
constexpr unsigned search(const float target, SpanSized<const float> values,
                          const unsigned binary_search_threshold) {
  if (values.size() < binary_search_threshold) {
    for (unsigned i = 0; i < values.size(); ++i) {
      if (values[i] >= target) {
        return i;
      }
    }

    unreachable_unchecked();
  } else {
    // binary search
    // UNIMPLEMENTED...
    unreachable_unchecked();
  }
}

class Ref {
public:
  // SPEED: don't store triangles in here, use indices instead?
  // would requre applying transformation and taking as input.
  HOST_DEVICE Ref(const Settings &settings,
                  Span<const intersect::Triangle> triangles,
                  SpanSized<const float> cumulative_weights)
      : triangles_(triangles), cumulative_weights_(cumulative_weights),
        binary_search_threshold_(settings.binary_search_threshold) {}

  static constexpr unsigned max_num_samples = 1;
  static constexpr bool performs_samples = true;

  template <bsdf::BSDF B, rng::RngState R>
  HOST_DEVICE ArrayVec<LightSample, max_num_samples>
  operator()(const Eigen::Vector3f &position, const bsdf::Material<B> & /*mat*/,
             const UnitVector & /*incoming_dir*/, const UnitVector &normal,
             R &rng) const {
    if (cumulative_weights_.size() == 0) {
      return {};
    }

    // TODO: SPEED, complexity, ...

    ArrayVec<LightSample, max_num_samples> out;
    for (unsigned i = 0; i < max_num_samples; ++i) {
      const float search_value = rng.next();
      const unsigned sample_idx = detail::search(
          search_value, cumulative_weights_, binary_search_threshold_);

      debug_assert(sample_idx < cumulative_weights_.size());

      const auto &triangle = triangles_[sample_idx];

      float weight0 = rng.next();
      float weight1 = rng.next();

      if (weight0 + weight1 > 1.f) {
        weight0 = 1 - weight0;
        weight1 = 1 - weight1;
      }

      const auto &vertices = triangle.vertices;

      // SPEED: cache vecs?
      const auto vec0 = vertices[1] - vertices[0];
      const auto vec1 = vertices[2] - vertices[0];

      const auto point = vertices[0] + vec0 * weight0 + vec1 * weight1;

      const Eigen::Vector3f direction_unnormalized = point - position;
      const UnitVector direction =
          UnitVector::new_normalize(direction_unnormalized);

      // SPEED: cache normal?
      const Eigen::Vector3f triangle_normal = triangle.normal_scaled_by_area();

      const float prob_this_triangle =
          edges_get_size<float>(sample_idx, cumulative_weights_);

      const float normal_weight =
          std::abs(normal->dot(*direction) * triangle_normal.dot(*direction));

      // case where we sample from the current triangle (or a parallel triangle)
      if (normal_weight < 1e-8) {
        // we could instead try to find a different sample, but that would
        // increase divergence on the gpu and could result in an infinite loop
        // if we aren't careful...
        continue;
      }

      const float weight = normal_weight / direction_unnormalized.squaredNorm();

      const FSample sample = {direction, prob_this_triangle / weight};
      out.push_back({sample, direction_unnormalized.norm()});
    }

    return out;
  }

private:
  Span<const intersect::Triangle> triangles_;
  SpanSized<const float> cumulative_weights_;
  unsigned binary_search_threshold_;
};

enum class TWItem {
  Triangle,
  Weight,
};
} // namespace detail

template <ExecutionModel> class RandomTriangle {
private:
  using TWItem = detail::TWItem;

  template <template <typename> class VecT>
  using TWGroup = VectorGroup<VecT, TWItem, intersect::Triangle, float>;

public:
  using Ref = detail::Ref;

  // need to be implementated when ExecStorage is defined
  RandomTriangle();
  ~RandomTriangle();
  RandomTriangle(RandomTriangle &&);
  RandomTriangle &operator=(RandomTriangle &&);

  template <bsdf::BSDF B>
  auto
  gen(const Settings &settings,
      Span<const scene::EmissiveCluster> emissive_groups,
      Span<const unsigned> emissive_group_ends_per_mesh,
      Span<const bsdf::Material<B>> materials,
      SpanSized<const intersect::TransformedObject> transformed_mesh_objects,
      Span<const unsigned> transformed_mesh_idxs,
      Span<const intersect::Triangle> triangles) {
    host_items_.clear_all();
    float cumulative_weight = 0.0f;
    for (unsigned object_idx = 0; object_idx < transformed_mesh_objects.size();
         ++object_idx) {
      const auto &transformed_object = transformed_mesh_objects[object_idx];
      auto [start, end] = edges_start_end(transformed_mesh_idxs[object_idx],
                                          emissive_group_ends_per_mesh);
      for (unsigned i = start; i < end; i++) {
        auto group = emissive_groups[i];
        const auto &transform = transformed_object.object_to_world();
        for (unsigned triangle_idx = group.start_idx;
             triangle_idx < group.end_idx; triangle_idx++) {
          // weight is proportional to intensity and surface area
          auto transformed_triangle =
              triangles[triangle_idx].transform(transform);

          float surface_area =
              transformed_triangle.normal_scaled_by_area().norm();

          cumulative_weight +=
              surface_area * materials[group.material_idx].emission.sum();
          host_items_.push_back_all(transformed_triangle, cumulative_weight);
        }
      }
    }

    // normalize
    for (auto &weight : host_items_.get(TAG(TWItem::Weight))) {
      weight /= cumulative_weight;
    }

    return finish_gen_internal(settings);
  }

private:
  Ref finish_gen_internal(const Settings &settings);

  // used to avoid instantiating device vectors in cpp files
  struct ExecStorage;

  TWGroup<HostVector> host_items_;
  std::unique_ptr<ExecStorage> exec_storage_;
};

template <ExecutionModel exec>
struct IsLightSampler
    : BoolWrapper<
          GeneralBSDFTriangleLightSampler<RandomTriangle<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsLightSampler>);
} // namespace random_triangle
} // namespace light_sampler
} // namespace integrate
