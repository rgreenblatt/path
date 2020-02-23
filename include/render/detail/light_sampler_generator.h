#pragma once

#include "intersect/mesh_instance.h"
#include "lib/binary_search.h"
#include "execution_model/execution_model_vector_type.h"
#include "lib/group.h"
#include "lib/projection.h"
#include "lib/span.h"
#include "material/material.h"
#include "render/light_sampler_type.h"
#include "rng/rng.h"
#include "scene/emissive_group.h"

#include <Eigen/Core>

#include <array>

namespace render {
namespace detail {
template <ExecutionModel execution_model, LightSamplerType type>
class LightSamplerGenerator;

struct LightSample {
  Eigen::Vector3f direction;
  float prob;
};

template <unsigned n> struct LightSamples {
  std::array<LightSample, n> samples;
  unsigned num_samples;
};

template <ExecutionModel execution_model>
class LightSamplerGenerator<execution_model,
                            LightSamplerType::NoDirectLighting> {
public:
  using Settings = LightSamplerSettings<LightSamplerType::NoDirectLighting>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &) {}

    HOST_DEVICE LightSamples<0> operator()(const Eigen::Vector3f &,
                                           const material::Material &,
                                           const Eigen::Vector3f &,
                                           const Eigen::Vector3f &,
                                           rng::Rng &) const {
      return {{}, 0};
    }

    static const bool performs_samples = false;
  };

  auto gen(const Settings &settings, Span<const scene::EmissiveGroup>,
           Span<const unsigned>, Span<const material::Material>,
           Span<const intersect::MeshInstance>) {
    return Ref(settings);
  }
};

template <ExecutionModel execution_model>
class LightSamplerGenerator<execution_model, LightSamplerType::WeightedAABB> {
public:
  using Settings = LightSamplerSettings<LightSamplerType::WeightedAABB>;

  class Ref {
  public:
    HOST_DEVICE Ref() = default;

    HOST_DEVICE Ref(const Settings &, Span<const intersect::accel::AABB> aabbs,
                    SpanSized<const float> cumulative_weights)
        : aabbs_(aabbs), cumulative_weights_(cumulative_weights) {}

    HOST_DEVICE LightSamples<1> operator()(const Eigen::Vector3f &position,
                                           const material::Material &material,
                                           const Eigen::Vector3f &normal,
                                           const Eigen::Vector3f &,
                                           rng::Rng &rng) const {
      // TODO: SPEED, complexity...
      unsigned sample_idx = binary_search<float>(
          0, cumulative_weights_.size(), rng.sample_1(), cumulative_weights_);
      assert(sample_idx < cumulative_weights_.size());

      const auto &aabb = aabbs_[sample_idx];
      const auto &min_bound = aabb.get_min_bound();
      const auto &max_bound = aabb.get_max_bound();

      const auto transform = find_rotate_vector_to_vector(normal, {0, 0, 1});

      float min_theta = std::numeric_limits<float>::max();
      float max_theta = std::numeric_limits<float>::lowest();
      float min_phi = std::numeric_limits<float>::max();
      float max_phi = std::numeric_limits<float>::lowest();

      for (bool x_is_min : {false, true}) {
        for (bool y_is_min : {false, true}) {
          for (bool z_is_min : {false, true}) {
            auto get_axis = [&](bool is_min, uint8_t axis) {
              return is_min ? min_bound[axis] : max_bound[axis];
            };
            // TODO: check
            auto point_on_aabb  = Eigen::Vector3f(get_axis(x_is_min, 0), get_axis(y_is_min, 1),
                                  get_axis(z_is_min, 2));
            auto world_space_direction = point_on_aabb - position;
            auto normal_is_up_space_direction = transform * world_space_direction;
            auto dir = normal_is_up_space_direction.normalized().eval();

            float theta = std::acos(dir.z());
            float phi = std::atan2(dir.y(), dir.x());

            min_theta = std::min(min_theta, theta);
            max_theta = std::max(max_theta, theta);
            min_phi = std::min(min_phi, phi);
            max_phi = std::max(max_phi, phi);
          }
        }
      }

      float max_possible_theta =
          float(material.is_bsdf() ? M_PI : M_PI / 2) - 1e-5f;

      assert(min_theta <= float(M_PI));
      assert(max_theta <= float(M_PI));

      if (min_theta > max_possible_theta) {
        return {{{}}, 0};
      }

      max_theta = std::min(max_theta, max_possible_theta);

      float region_area =
          (-std::cos(max_theta) + std::cos(min_theta)) * (max_phi - min_phi);

      auto [v0, v1] = rng.sample_2();

      float phi = min_phi + (max_phi - min_phi) * v0;
      float theta = min_theta + (max_theta - min_theta) * v0;

      return {std::array<LightSample, 1>{
                  {{find_relative_vec(normal, phi, theta), 1 / region_area}}},
              1};
    }

    static const bool performs_samples = true;

  private:
    Span<const intersect::accel::AABB> aabbs_;
    SpanSized<const float> cumulative_weights_;
  };

  auto gen(const Settings &settings,
           Span<const scene::EmissiveGroup> emissive_groups,
           Span<const unsigned> emissive_group_ends_per_mesh,
           Span<const material::Material> materials,
           SpanSized<const intersect::MeshInstance> mesh_instances) {
    std::vector<float> cumulative_weights;
    std::vector<intersect::accel::AABB> aabbs;
    float cumulative_weight = 0.0f;
    for (const auto &mesh_instance : mesh_instances) {
      auto [start, end] =
          group_start_end(mesh_instance.idx(), emissive_group_ends_per_mesh);
      for (unsigned i = start; i < end; i++) {
        // weight is proportional to intensity and volume
        auto group = emissive_groups[i];
        cumulative_weight += group.aabb.surface_area() *
                             materials[group.material_idx].emission().sum();
        cumulative_weights.push_back(cumulative_weight);
        aabbs.push_back(group.aabb.transform(mesh_instance.mesh_to_world()));
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
} // namespace detail
} // namespace render
