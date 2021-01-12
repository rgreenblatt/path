#pragma once

#include "execution_model/execution_model.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/dir_tree/dir_tree.h"
#include "intersect/accel/enum_accel/accel_type.h"
#include "intersect/accel/enum_accel/settings.h"
#include "intersect/accel/kdtree/kdtree.h"
#include "intersect/accel/loop_all/loop_all.h"
#include "lib/settings.h"
#include "meta/pick_type.h"
#include "meta/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace enum_accel {
template <AccelType type, ExecutionModel exec>
struct EnumAccel
    : public PickType<type, loop_all::LoopAll, kdtree::KDTree<exec>,
                      dir_tree::DirTree<exec>> {};

template <AccelType type, ExecutionModel exec>
struct IsAccel
    : BoolWrapper<BoundsOnlyAccel<EnumAccel<type, exec>, Settings<type>>> {};

static_assert(PredicateForAllValues<AccelType, ExecutionModel>::value<IsAccel>);
} // namespace enum_accel
} // namespace accel
} // namespace intersect
