#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/aabb.h"
#include "intersect/accel/sbvh/sbvh.h"
#include "intersect/accel/sbvh/settings.h"
#include "lib/span.h"

#include <tuple>

namespace intersect {
namespace accel {
using namespace detail;
namespace sbvh {
// should only be used from detail context
using namespace detail;

struct ObjectSplitCandidate {
  std::vector<unsigned> perm;
  unsigned split_point;
  float cost;
};

struct SpatialSplitCandidate {
  float cost;
  float position;
};

template <ExecutionModel exec> class SBVH<exec>::Generator {
public:
  Generator() = default;

  RefPerm<BVH> gen(const Settings &settings,
                   SpanSized<const Triangle> triangles);

private:
  ObjectSplitCandidate best_object_split(SpanSized<const Triangle> triangles,
                                         unsigned axis,
                                         float surface_area_above_node);

  SpatialSplitCandidate best_spatial_split(SpanSized<const Triangle> triangles,
                                           unsigned axis);

  std::vector<unsigned> sort_by_axis(SpanSized<Triangle> triangles,
                                     unsigned axis);

  Node create_node(SpanSized<Triangle> triangles, SpanSized<unsigned> idxs,
                   std::vector<Node> &nodes, unsigned start_idx);

  float sa_heurisitic_cost(SpanSized<const Node> nodes, unsigned start_node);

  Settings settings_;
  ExecVector<exec, Node> nodes_;
};
} // namespace sbvh
} // namespace accel
} // namespace intersect
