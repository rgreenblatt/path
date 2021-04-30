#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/naive_partition_bvh/detail/node.h"
#include "intersect/accel/naive_partition_bvh/settings.h"
#include "lib/attribute.h"
#include "meta/all_values/impl/enum.h"

#include <memory>

namespace intersect {
namespace accel {
namespace naive_partition_bvh {
namespace detail {
// In this case, the Ref type doesn't depend on the ExecutionModel
struct Ref {
  SpanSized<const Node> nodes;

  template <IntersectableAtIdx F>
  HOST_DEVICE inline AccelRet<F>
  intersect_objects(const intersect::Ray &ray,
                    const F &intersectable_at_idx) const;
};
} // namespace detail

template <ExecutionModel exec> class NaivePartitionBVH {
public:
  // need to implementated when Generator is defined
  NaivePartitionBVH();
  ~NaivePartitionBVH();
  NaivePartitionBVH(NaivePartitionBVH &&);
  NaivePartitionBVH &operator=(NaivePartitionBVH &&);

  using Ref = detail::Ref;

  template <Bounded B>
  RefPerm<detail::Ref> gen(const Settings &settings,
                           SpanSized<const B> objects) {
    bounds_.resize(objects.size());

    for (unsigned i = 0; i < objects.size(); ++i) {
      decltype(auto) aabb = objects[i].bounds();
      bounds_[i] = {aabb, aabb.max_bound - aabb.min_bound};
    }

    return gen_internal(settings);
  }

private:
  RefPerm<detail::Ref> gen_internal(const Settings &settings);

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
