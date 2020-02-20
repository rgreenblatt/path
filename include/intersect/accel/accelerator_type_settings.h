#pragma once

#include "intersect/accel/accelerator_type.h"
#include "intersect/accel/s_a_heuristic_settings.h"

namespace intersect {
namespace accel {
template <AccelType type> struct AccelSettings;

template <> struct AccelSettings<AccelType::LoopAll> {};

template <> struct AccelSettings<AccelType::KDTree> {
  SAHeuristicSettings s_a_heuristic_settings;
};

template <> struct AccelSettings<AccelType::DirTree> {
  SAHeuristicSettings s_a_heuristic_settings;
  unsigned num_dir_trees;
};
} // namespace accel
} // namespace intersect
