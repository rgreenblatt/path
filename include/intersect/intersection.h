#pragma once

#include "lib/cuda/utils.h"
#include "lib/optional.h"
#include "lib/concepts.h"

#include <map>

#include <thrust/optional.h>

namespace intersect {
template <typename T> struct Intersection {
  using InfoType = T;

  HOST_DEVICE Intersection() = default;

  HOST_DEVICE Intersection(float intersection_dist, const T &info)
      : intersection_dist(intersection_dist), info(info) {}

  float intersection_dist;
  T info;
};

template <typename InfoType>
HOST_DEVICE inline auto operator<=>(const Intersection<InfoType> &lhs,
                                    const Intersection<InfoType> &rhs) {
  return lhs.intersection_dist <=> rhs.intersection_dist;
}

template <typename InfoType>
using IntersectionOp = thrust::optional<Intersection<InfoType>>;

template <typename InfoType>
using AppendIndexInfoType =
    std::array<unsigned, std::tuple_size_v<InfoType> + 1>;

template <typename InfoType>
HOST_DEVICE IntersectionOp<AppendIndexInfoType<InfoType>>
append_index(const IntersectionOp<InfoType> &intersect_op, unsigned idx) {
  static_assert(StdArraySpecialization<InfoType>);
  return optional_map(intersect_op,
                      [&](const Intersection<InfoType> &intersect)
                          -> Intersection<AppendIndexInfoType<InfoType>> {
                        AppendIndexInfoType<InfoType> out;
                        for (unsigned i = 0; i < std::tuple_size_v<InfoType>;
                             ++i) {
                          out[i] = intersect.info[i];
                        }
                        out[out.size() - 1] = idx;

                        return {intersect.intersection_dist, out};
                      });
}
}

// Fix specialization check...
template <template <typename... Args> class Template, typename Arg>
struct is_specialization<intersect::IntersectionOp<Arg>, Template>
    : std::true_type {};
