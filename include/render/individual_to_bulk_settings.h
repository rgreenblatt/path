#pragma once

#include "intersectable_scene/to_bulk.h"
#include "render/individually_intersectable_settings.h"

namespace render {
struct IndividualToBulkSettings {
  intersectable_scene::ToBulkSettings to_bulk_settings;
  IndividuallyIntersectableSettings individual_settings;

  using CompileTime = IndividuallyIntersectableSettings::CompileTime;

  constexpr CompileTime compile_time() const {
    return individual_settings.compile_time();
  }

  SETTING_BODY(IndividualToBulkSettings, to_bulk_settings, individual_settings);
};
} // namespace render
