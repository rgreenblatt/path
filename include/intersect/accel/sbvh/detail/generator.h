#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/aabb.h"
#include "intersect/accel/detail/clipped_triangle.h"
#include "intersect/accel/sbvh/sbvh.h"
#include "intersect/accel/sbvh/settings.h"
#include "lib/span.h"

#include <tuple>

namespace intersect {
namespace accel {
using namespace detail::bvh;
namespace sbvh {
// should only be used from detail context
using namespace detail;

enum class SplitType {
  Object,
  Spatial,
};

struct SplitCandidate {
  float base_cost;

  struct Object {
    HostVector<unsigned> perm;
    unsigned split_point;
  };

  struct Spatial {
    HostVector<unsigned> left_triangles;
    HostVector<unsigned> right_triangles;
    HostVector<AABB> left_bounds;
    HostVector<AABB> right_bounds;
    AABB left_aabb;
    AABB right_aabb;
  };

  using Item = TaggedUnion<SplitType, Object, Spatial>;

  Item item;

  bool operator<(const SplitCandidate &other) const {
    return base_cost < other.base_cost;
  }
};

// "base_cost" refers to SAH cost, but without dividing by the outer
// node surface area and without the traversal cost

template <ExecutionModel exec> class SBVH<exec>::Generator {
public:
  Generator() = default;

  RefPerm<BVH<>> gen(const Settings &settings,
                     SpanSized<const Triangle> triangles);

private:
  SplitCandidate best_object_split(SpanSized<const ClippedTriangle> triangles,
                                   unsigned axis);

  SplitCandidate best_spatial_split(SpanSized<const ClippedTriangle> triangles,
                                    unsigned axis);

  HostVector<unsigned> sort_by_axis(SpanSized<ClippedTriangle> triangles,
                                    unsigned axis);

  Node create_node(SpanSized<ClippedTriangle> triangles,
                   SpanSized<unsigned> idxs,
                   SpanSized<std::optional<unsigned>> final_idxs_for_right_dups,
                   HostVector<Node> &nodes, HostVector<unsigned> &extra_idxs,
                   float traversal_per_intersect_cost, bool use_spatial_splits,
                   unsigned start_idx);

  ExecVector<exec, Node> nodes_;
  ExecVector<exec, unsigned> extra_idxs_;

  // just for debug asserts
  Span<const Triangle> triangles_in_;
};
} // namespace sbvh
} // namespace accel
} // namespace intersect
