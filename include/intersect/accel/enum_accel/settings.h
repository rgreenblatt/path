#pragma once

#include "intersect/accel/dir_tree/settings.h"
#include "intersect/accel/enum_accel/accel_type.h"
#include "intersect/accel/kdtree/settings.h"
#include "intersect/accel/loop_all/settings.h"
#include "lib/pick_type.h"

namespace intersect {
namespace accel {
namespace enum_accel {
template <AccelType type>
struct Settings : public PickType<AccelType, type, loop_all::Settings,
                                  kdtree::Settings, dir_tree::Settings> {};

// TODO: consider checking here?
} // namespace enum_accel
} // namespace accel
} // namespace intersect
