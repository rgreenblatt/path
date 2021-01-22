#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersectable_scene/intersectable_scene.h"
#include "lib/optional.h"
#include "lib/settings.h"

namespace intersectable_scene {
struct ToBulkSettings {
  unsigned max_size = 2097152;

  SETTING_BODY(ToBulkSettings, max_size);
};

template <ExecutionModel exec, intersect::Intersectable I> struct ToBulkGen {
public:
  ToBulkGen() = default;
  ToBulkGen(const ToBulkGen &) = delete;
  ToBulkGen &operator=(const ToBulkGen &) = delete;

  using InfoType = typename I::InfoType;
  using IntersectionOp = intersect::IntersectionOp<InfoType>;

  void set_settings_intersectable(const ToBulkSettings &settings,
                                  const I &intersectable) {
    scene_intersectable_ = SceneSettings{
        .settings = settings,
        .intersectable = intersectable,
    };
  }

  unsigned max_size() const {
    always_assert(scene_intersectable_.has_value());
    return scene_intersectable_->settings.max_size;
  }

  SpanRayWriter ray_writer(unsigned size) {
    rays_.resize(size);
    return {rays_};
  }

  Span<const IntersectionOp> get_intersections();

  static constexpr bool individually_intersectable = false;

private:
  struct SceneSettings {
    ToBulkSettings settings;
    I intersectable;
  };

  std::optional<SceneSettings> scene_intersectable_;

  template <typename T> using ExecVecT = ExecVector<exec, T>;

  ExecVecT<intersect::Ray> rays_;
  ExecVecT<IntersectionOp> intersections_;
};

static_assert(BulkIntersector<
              ToBulkGen<ExecutionModel::CPU, intersect::MockIntersectable>>);
} // namespace intersectable_scene
