#pragma once

#include "execution_model/execution_model.h"
#include "intersect/accel/direction_grid/direction_grid.h"
#include "intersect/accel/enum_accel/accel_type.h"
#include "intersect/accel/enum_accel/settings.h"
#include "intersect/accel/loop_all/loop_all.h"
#include "intersect/accel/naive_partition_bvh/naive_partition_bvh.h"
#include "intersect/accel/sbvh/sbvh.h"
#include "intersect/accel/triangle_accel.h"
#include "lib/settings.h"
#include "meta/all_values/impl/enum.h"
#include "meta/all_values/pick_type.h"
#include "meta/all_values/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace enum_accel {
template <AccelType type, ExecutionModel exec>
using EnumAccel =
    PickType<type, loop_all::LoopAll,
             naive_partition_bvh::NaivePartitionBVH<exec>, sbvh::SBVH<exec>,
             direction_grid::DirectionGrid<exec>>;

template <AccelType type, ExecutionModel exec>
struct IsTriangleAccel
    : std::bool_constant<TriangleAccel<EnumAccel<type, exec>, Settings<type>>> {
};

static_assert(
    PredicateForAllValues<AccelType, ExecutionModel>::value<IsTriangleAccel>);
} // namespace enum_accel
} // namespace accel
} // namespace intersect
