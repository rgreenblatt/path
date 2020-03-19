#pragma once

#include "compile_time_dispatch/one_per_instance.h"
#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/dir_tree.h"
#include "intersect/accel/kdtree.h"
#include "intersect/accel/loop_all.h"
#include "intersect/transformed_object.h"
#include "lib/bgra.h"
#include "render/detail/dir_sampler.h"
#include "render/detail/light_sampler.h"
#include "render/detail/term_prob.h"
#include "render/settings.h"
#include "rng/halton.h"
#include "rng/sobel.h"
#include "rng/uniform.h"
#include "scene/scene.h"

#include <map>
#include <set>

namespace render {
namespace detail {
template <ExecutionModel execution_model> class RendererImpl {
public:
  void render(Span<BGRA> pixels, const scene::Scene &s, unsigned samples_per,
              unsigned x_dim, unsigned y_dim, const Settings &settings,
              bool show_times);

  RendererImpl();

private:
  template <typename T> using ExecVecT = ExecVector<execution_model, T>;
  template <typename T> using SharedVecT = SharedVector<execution_model, T>;

  // TODO: consider eventually freeing...
  template <intersect::accel::AccelType type> class StoredTriangleAccels {
  public:
    using Triangle = intersect::Triangle;
    using Accel = intersect::accel::AccelT<type, execution_model, Triangle>;
    using Settings = intersect::accel::AccelSettings<type>;
    using RefType = typename Accel::Ref;

    void reset() {
      free_indexes_.clear();
      for (unsigned i = 0; i < accels_.size(); i++) {
        free_indexes_.insert(i);
      }
    }

    thrust::optional<RefType> query(const std::string &mesh_identifier,
                                    const Settings &new_settings) {
      auto it = existing_triangle_accel_vals_.find(mesh_identifier);
      if (it == existing_triangle_accel_vals_.end()) {
        return thrust::nullopt;
      }

      auto [index, settings] = it->second;

      if (settings != new_settings) {
        return thrust::nullopt;
      }

      free_indexes_.erase(index);

      return refs_[index];
    }

    RefType add(Span<const Triangle> triangles, unsigned start, unsigned end,
                const intersect::accel::AABB &aabb, const Settings &settings,
                const std::string &mesh_identifier) {
      // SPEED: try to get item which is closest in size...
      auto generate_new = [&](unsigned idx) {
        existing_triangle_accel_vals_.insert(
            std::make_pair(mesh_identifier, std::make_tuple(idx, settings)));
        return accels_[idx].gen(settings, triangles, start, end, aabb);
      };

      if (free_indexes_.empty()) {
        unsigned new_idx = accels_.size();
        accels_.push_back(Accel());
        auto new_ref = generate_new(new_idx);
        refs_.push_back(new_ref);

        return new_ref;
      } else {
        auto it = free_indexes_.begin();
        auto new_ref = generate_new(*it);
        refs_[*it] = new_ref;
        free_indexes_.erase(it);

        return new_ref;
      }
    }

  private:
    std::set<unsigned> free_indexes_;
    std::map<std::string, std::tuple<unsigned, Settings>>
        existing_triangle_accel_vals_;

    HostVector<RefType> refs_;
    HostVector<Accel> accels_;
  };

  OnePerInstance<intersect::accel::AccelType, StoredTriangleAccels>
      stored_triangle_accels_;

  template <intersect::accel::AccelType type>
  using MeshAccel = intersect::accel::AccelT<type, execution_model,
                                             intersect::TransformedObject>;

  OnePerInstance<intersect::accel::AccelType, MeshAccel> mesh_accel_;

  template <LightSamplerType type>
  using LightSamplerT = LightSamplerT<type, execution_model>;

  OnePerInstance<LightSamplerType, LightSamplerT> light_samplers_;

  template <DirSamplerType type>
  using DirSamplerT = DirSamplerT<type, execution_model>;

  OnePerInstance<DirSamplerType, DirSamplerT> dir_samplers_;

  template <TermProbType type>
  using TermProbT = TermProbT<type, execution_model>;

  OnePerInstance<TermProbType, TermProbT> term_probs_;

  template <rng::RngType type> using Rng = rng::RngT<type, execution_model>;

  OnePerInstance<rng::RngType, Rng> rngs_;

  ThrustData<execution_model> thrust_data_;

  ExecVecT<Eigen::Array3f> intensities_;
  ExecVecT<scene::TriangleData> triangle_data_;
  ExecVecT<material::Material> materials_;
  ExecVecT<BGRA> bgra_;
};
} // namespace detail
} // namespace render
