#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/kdtree/node.h"
#include "intersect/accel/kdtree/settings.h"

namespace intersect {
namespace accel {
namespace kdtree {
namespace detail {
// In this case, the Ref type doesn't depend on the ExecutionModel
class Ref {
public:
  // TODO: why is this constructor needed...
  HOST_DEVICE Ref() {}

  Ref(SpanSized<const KDTreeNode<AABB>> nodes,
      Span<const unsigned> local_idx_to_global_idx, const AABB &aabb)
      : nodes_(nodes), local_idx_to_global_idx_(local_idx_to_global_idx),
        aabb_(aabb) {}

  constexpr inline const AABB &bounds() const { return aabb_; }

  template <Object O>
  HOST_DEVICE inline AccelRet<O> intersect_objects(const intersect::Ray &ray,
                                                   Span<const O> objects) const;

private:
  SpanSized<const KDTreeNode<AABB>> nodes_;
  Span<const unsigned> local_idx_to_global_idx_;
  AABB aabb_;
};
} // namespace detail

template <ExecutionModel execution_model> class KDTree {
public:
  // need to implementated when Generator is defined
  KDTree();
  ~KDTree();
  KDTree(KDTree &&);
  KDTree &operator=(KDTree &&);

  template <Bounded B>
  detail::Ref gen(const Settings &settings, SpanSized<const B> objects,
                  const AABB &aabb) {
    bounds_.resize(objects.size());

    for (unsigned i = 0; i < objects.size(); ++i) {
      auto &&aabb = objects[i].bounds();
      bounds_[i] = {aabb, aabb.max_bound - aabb.min_bound};
    }

    return gen_internal(settings, aabb);
  }

private:
  detail::Ref gen_internal(const Settings &settings, const AABB &aabb);

  // PIMPL
  class Generator;

  std::unique_ptr<Generator> gen_;

  std::vector<detail::Bounds> bounds_;
};

template <ExecutionModel exec>
struct KDTreeAccel : BoolWrapper<BoundsOnlyAccel<KDTree<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<KDTreeAccel>);
} // namespace kdtree
} // namespace accel
} // namespace intersect
