#pragma once

#include "lib/cuda/utils.h"

#include "lib/optional.h"

#include <thrust/optional.h>

namespace intersect {
template <typename T> struct Intersection {
  using InfoType = T;

  HOST_DEVICE Intersection() {}

  HOST_DEVICE Intersection(float intersection_dist, const T &info)
      : intersection_dist(intersection_dist), info(info) {}

  float intersection_dist;
  T info;
};

template <typename Info>
HOST_DEVICE inline bool operator<(const Intersection<Info> &lhs,
                                  const Intersection<Info> &rhs) {
  return lhs.intersection_dist < rhs.intersection_dist;
}

template <typename Info>
HOST_DEVICE inline bool operator>(const Intersection<Info> &lhs,
                                  const Intersection<Info> &rhs) {
  return operator<(rhs, lhs);
}

template <typename Info>
HOST_DEVICE inline bool operator<=(const Intersection<Info> &lhs,
                                   const Intersection<Info> &rhs) {
  return !operator>(lhs, rhs);
}

template <typename Info>
HOST_DEVICE inline bool operator>=(const Intersection<Info> &lhs,
                                   const Intersection<Info> &rhs) {
  return !operator<(lhs, rhs);
}

template <typename InfoType>
using IntersectionOp = thrust::optional<Intersection<InfoType>>;

template <typename InfoType>
using AppendIndexInfoType = decltype(std::tuple_cat(
    std::declval<InfoType>(), std::declval<std::tuple<unsigned>>()));

template <typename InfoType>
HOST_DEVICE IntersectionOp<AppendIndexInfoType<InfoType>>
append_index(const IntersectionOp<InfoType> &i, unsigned idx) {
  return optional_map(i,
                      [&](const Intersection<InfoType> &i)
                          -> Intersection<AppendIndexInfoType<InfoType>> {
                        return {i.intersection_dist,
                                std::tuple_cat(i.info, std::make_tuple(idx))};
                      });
}
} // namespace intersect
