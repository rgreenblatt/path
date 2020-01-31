#pragma once

#include "lib/cuda/utils.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
struct BestEdge {
  float cost;
  unsigned idx;

  HOST_DEVICE BestEdge() {}

  HOST_DEVICE BestEdge(float cost, unsigned idx) : cost(cost), idx(idx) {}

  HOST_DEVICE friend bool operator>(const BestEdge &b_e_1,
                                    const BestEdge &b_e_2);
  HOST_DEVICE friend bool operator<=(const BestEdge &b_e_1,
                                     const BestEdge &b_e_2);

  HOST_DEVICE friend bool operator<(const BestEdge &b_e_1,
                                    const BestEdge &b_e_2);
  HOST_DEVICE friend bool operator>=(const BestEdge &b_e_1,
                                     const BestEdge &b_e_2);
};

HOST_DEVICE inline bool operator>(const BestEdge &b_e_1,
                                  const BestEdge &b_e_2) {
  return b_e_1.cost > b_e_2.cost;
}

HOST_DEVICE inline bool operator>=(const BestEdge &b_e_1,
                                   const BestEdge &b_e_2) {
  return b_e_1.cost >= b_e_2.cost;
}

HOST_DEVICE inline bool operator<(const BestEdge &b_e_1,
                                  const BestEdge &b_e_2) {
  return b_e_1.cost < b_e_2.cost;
}

HOST_DEVICE inline bool operator<=(const BestEdge &b_e_1,
                                   const BestEdge &b_e_2) {
  return b_e_1.cost <= b_e_2.cost;
}
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
