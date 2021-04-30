#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/detail/bvh.h"
#include "intersect/accel/naive_partition_bvh/settings.h"
#include "lib/attribute.h"
#include "meta/all_values/impl/enum.h"

#include <memory>

namespace intersect {
namespace accel {
namespace naive_partition_bvh {
namespace detail {
struct Bounds {
  AABB aabb;
  Eigen::Vector3f center;
};
} // namespace detail

template <ExecutionModel exec> class NaivePartitionBVH {
public:
  // need to implementated when Generator is defined
  NaivePartitionBVH();
  ~NaivePartitionBVH();
  NaivePartitionBVH(NaivePartitionBVH &&);
  NaivePartitionBVH &operator=(NaivePartitionBVH &&);

  using Ref = accel::detail::BVH;

  template <Bounded B>
  RefPerm<Ref> gen(const Settings &settings, SpanSized<const B> objects) {
    bounds_.resize(objects.size());

    for (unsigned i = 0; i < objects.size(); ++i) {
      decltype(auto) aabb = objects[i].bounds();
      bounds_[i] = {aabb, aabb.max_bound - aabb.min_bound};
    }

    return gen_internal(settings);
  }

private:
  RefPerm<Ref> gen_internal(const Settings &settings);

  // PIMPL
  class Generator;

  std::unique_ptr<Generator> gen_;

  std::vector<detail::Bounds> bounds_;
};

template <ExecutionModel exec>
struct IsAccel
    : std::bool_constant<BoundsOnlyAccel<NaivePartitionBVH<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsAccel>);
} // namespace naive_partition_bvh
} // namespace accel
} // namespace intersect
