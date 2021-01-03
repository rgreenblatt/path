#pragma once

#include "intersectable_scene/intersectable_scene.h"
#include "execution_model/execution_model_vector_type.h"
#include "lib/settings.h"
#include "lib/optional.h"

#include <thrust/transform.h>

namespace intersectable_scene {
struct ToBulkSettings {
  unsigned max_size = 2097152;

  template <typename Archive> void serialize(Archive &ar) {
    ar(NVP(max_size));
  }

  ATTR_PURE constexpr bool operator==(const ToBulkSettings &) const = default;
};

template<ExecutionModel exec, IntersectableScene S>
requires S::individually_intersectable
class ToBulkGen {
public:
  using IntersectionOp = intersect::IntersectionOp<typename S::InfoType>;

  ToBulkGen set_settings(const ToBulkSettings& settings)  {
    settings_ = settings;
  }

  unsigned max_size()  const {
    always_assert(settings_.has_value());
    return settings_->max_size;
  }

  SpanRayWriter ray_writer(unsigned size) const { 
    rays_.resize(size);
    return {rays_};
  }

  Span<const IntersectionOp> get_intersections() {
    thrust::transform(
        rays_.begin(), rays_.end(), intersections_.begin(),
        [intersectable = scene_.intersectable()](const intersect::Ray &ray) {
          intersectable.intersect(ray);
        });
  }

  static constexpr bool individually_intersectable = false;



private:
  Optional<ToBulkSettings> settings_;

  template <typename T> using ExecVecT = ExecVector<exec, T>;

  S scene_;

  ExecVecT<intersect::Ray> rays_;
  ExecVecT<IntersectionOp> intersections_;
};
} // namespace intersectable_scene
