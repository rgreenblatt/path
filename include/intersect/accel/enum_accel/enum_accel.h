#pragma once

#include "execution_model/execution_model.h"
#include "intersect/object.h"
#include "intersect/accel/enum_accel/accel_type.h"
// #include "intersect/accel/enum_accel/settings.h" // TODO: check
#include "intersect/accel/kdtree/kdtree.h"
#include "intersect/accel/loop_all/loop_all.h"
#include "intersect/accel/dir_tree/dir_tree.h"

namespace intersect {
namespace accel {
namespace enum_accel {
template <AccelType type, ExecutionModel execution_model, Object O>
struct EnumAccel
    : public PickType<AccelType, type, loop_all::LoopAll<execution_model, O>,
                      kdtree::KDTree<execution_model, O>,
                      dir_tree::DirTree<execution_model, O>> {};

// TODO: consider checking here?
} // enum_accel
} // namespace accel
} // namespace intersect
