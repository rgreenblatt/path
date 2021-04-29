#pragma once

#include "execution_model/execution_model.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/enum_accel/accel_type.h"
#include "intersect/accel/enum_accel/settings.h"
#include "intersect/accel/loop_all/loop_all.h"
#include "intersect/accel/naive_partition_bvh/naive_partition_bvh.h"
#include "lib/settings.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/pick_type.h"
#include "meta/all_values/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace enum_accel {
template <AccelType type, ExecutionModel exec>
using EnumAccel = PickType<type, loop_all::LoopAll,
                           naive_partition_bvh::NaivePartitionBVH<exec>>;

template <AccelType type, ExecutionModel exec>
struct IsAccel : std::bool_constant<
                     BoundsOnlyAccel<EnumAccel<type, exec>, Settings<type>>> {};

static_assert(PredicateForAllValues<AccelType, ExecutionModel>::value<IsAccel>);
} // namespace enum_accel
} // namespace accel
} // namespace intersect
