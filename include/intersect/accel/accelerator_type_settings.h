#pragma once

#include "intersect/accel/accelerator_type.h"
#include "intersect/accel/s_a_heuristic_settings.h"

namespace intersect {
namespace accel {
template <AcceleratorType type> struct Settings;

template <> struct Settings<AcceleratorType::LoopAll> {};

template <> struct Settings<AcceleratorType::KDTree> {
  SAHeuristicSettings s_a_heuristic_settings;
};

template <> struct Settings<AcceleratorType::DirTree> {
  SAHeuristicSettings s_a_heuristic_settings;
  unsigned num_dir_trees;
};
} // namespace accel
} // namespace intersect
