#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/aabb.h"
#include "intersect/accel/sbvh/sbvh.h"
#include "intersect/accel/sbvh/settings.h"
#include "lib/span.h"
#include "lib/vector_group.h"
#include "meta/tuple.h"

#include <tuple>

namespace intersect {
namespace accel {
using namespace detail::bvh;
namespace sbvh {
// should only be used from detail context
using namespace detail;

// "base_cost" refers to SAH cost, but without dividing by the outer
// node surface area and without the traversal cost

struct ObjectSplitCandidate {
  HostVector<unsigned> perm;
  unsigned split_point;
  float base_cost;
};

struct SpatialSplitCandidate {
  float base_cost;
  HostVector<unsigned> left_triangles;
  HostVector<unsigned> right_triangles;
  AABB left_aabb;
  AABB right_aabb;
};

template <ExecutionModel exec> class SBVH<exec>::Generator {
public:
  Generator() = default;

  RefPerm<BVH<>> gen(const Settings &settings,
                     SpanSized<const Triangle> triangles);

private:
  ObjectSplitCandidate best_object_split(SpanSized<const Triangle> triangles,
                                         unsigned axis);

  SpatialSplitCandidate best_spatial_split(SpanSized<const Triangle> triangles,
                                           unsigned axis);

  HostVector<unsigned> sort_by_axis(SpanSized<Triangle> triangles,
                                    unsigned axis);

  NodeTup create_node(SpanSized<Triangle> triangles, SpanSized<unsigned> idxs,
                      NodeGroup<HostVector> &nodes,
                      float traversal_per_intersect_cost, unsigned start_idx);

  template <typename T> using ExecVecT = ExecVector<exec, T>;

  NodeGroup<ExecVecT> nodes_;
};
} // namespace sbvh
} // namespace accel
} // namespace intersect
