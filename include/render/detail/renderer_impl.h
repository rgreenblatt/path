#pragma once

#include "intersect/accel/accelerator_type.h"
#include "intersect/accel/accelerator_type_generator.h"
#include "lib/execution_model.h"
#include "lib/execution_model_vector_type.h"
#include "lib/rgba.h"
#include "lib/thrust_data.h"
#include "scene/scene.h"

#include <thrust/device_vector.h>

#include <map>
#include <set>

namespace render {
namespace detail {
template <ExecutionModel execution_model> class RendererImpl {
public:
  void render(RGBA *pixels, const scene::Scene &s, unsigned x_dim,
              unsigned y_dim, unsigned samples_per,
              intersect::accel::AcceleratorType mesh_accel_type,
              intersect::accel::AcceleratorType triangle_accel_type,
              bool show_times);

  RendererImpl();

private:
  template <typename T> using ExecVecT = ExecVector<execution_model, T>;
  template <typename T> using SharedVecT = SharedVector<execution_model, T>;

  // TODO: consider eventually freeing...
  template <intersect::accel::AcceleratorType type> class StoredMeshAccels {
  public:
    using Triangle = intersect::Triangle;
    using Generator =
        intersect::accel::Generator<Triangle, execution_model, type>;
    using Settings = intersect::accel::Settings<type>;
    using RefType = typename Generator::RefType;

    void set_settings(const Settings &settings) { settings_ = settings; }

    void reset() {
      free_indexes_.clear();
      for (unsigned i = 0; i < generators_.size(); i++) {
        free_indexes_.insert(i);
      }
    }

    thrust::optional<RefType> query(const std::string &mesh_identifier) {
      auto it = existing_triangle_accel_vals_.find(mesh_identifier);
      if (it == existing_triangle_accel_vals_.end()) {
        return thrust::nullopt;
      }

      free_indexes_.erase(it->second);

      return refs_[it->second];
    }

    RefType add(Span<const Triangle> triangles, unsigned start, unsigned end,
                const Eigen::Vector3f &min_bound,
                const Eigen::Vector3f &max_bound) {
      // SPEED: try to get item which is closest in size...
      auto generate_new = [&](unsigned idx) {
        return generators_[idx].gen(triangles, start, end, min_bound, max_bound,
                                    settings_);
      };

      if (free_indexes_.empty()) {
        unsigned new_idx = generators_.size();
        generators_.push_back(Generator());
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
    std::map<std::string, unsigned> existing_triangle_accel_vals_;

    Settings settings_;
    HostVector<RefType> refs_;
    HostVector<Generator> generators_;
  };

  intersect::accel::OnePerAcceleratorType<StoredMeshAccels> stored_mesh_accels_;

#if 0
  template <intersect::accel::AcceleratorType type> class StoredPrimAccel {
    using MeshInstance = intersect::accel::MeshInstance;
    using Generator =
        intersect::accel::Generator<MeshInstance, execution_model, type>;
    using Settings = intersect::accel::Settings<type>;
    using RefType = typename Generator::RefType;

    void set_settings(const Settings &settings) { settings_ = settings; }




  private:
    Settings settings_;
  };
#endif

  ThrustData<execution_model> thrust_data_;

  ExecVecT<Eigen::Vector3f> intensities;
  ExecVecT<RGBA> bgra_;
};
} // namespace detail
} // namespace render
