#pragma once

#include "kernel/work_division_settings.h"
#include "lib/settings.h"
#include "lib/tagged_union.h"
#include "render/individual_to_bulk_settings.h"

namespace render {
struct StreamingSettings {
  struct ComputationSettings {
    kernel::WorkDivisionSettings init_samples_division = {};
    unsigned max_num_samples_per_launch = 4194304;
    unsigned min_num_init_blocks = 16;

    SETTING_BODY(ComputationSettings, init_samples_division,
                 max_num_samples_per_launch, min_num_init_blocks);
  };

  ComputationSettings computation_settings = {};

  enum class BulkIntersectionApproaches {
    IndividualToBulk,
    // Optix // consider adding this...
  };

  TaggedUnion<BulkIntersectionApproaches, IndividualToBulkSettings> accel = {};

  using CompileTime = TaggedUnion<BulkIntersectionApproaches,
                                  IndividualToBulkSettings::CompileTime>;

  constexpr CompileTime compile_time() const {
    return accel.visit_tagged([&](auto tag, const auto &value) -> CompileTime {
      return {tag, value.compile_time()};
    });
  }

  SETTING_BODY(StreamingSettings, computation_settings, accel);
};

static_assert(Setting<StreamingSettings::ComputationSettings>);
static_assert(Setting<StreamingSettings>);
} // namespace render
