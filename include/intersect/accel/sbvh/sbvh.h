#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "execution_model/thrust_data.h"
#include "intersect/accel/detail/bvh/bvh.h"
#include "intersect/accel/sbvh/settings.h"
#include "intersect/accel/triangle_accel.h"
#include "intersect/triangle.h"
#include "lib/attribute.h"
#include "meta/all_values/impl/enum.h"

#include <memory>

namespace intersect {
namespace accel {
namespace sbvh {
template <ExecutionModel exec> class SBVH {
public:
  // need to implementated when Generator is defined
  SBVH();
  ~SBVH();
  SBVH(SBVH &&);
  SBVH &operator=(SBVH &&);

  using Ref = accel::detail::bvh::BVH<>;

  RefPerm<Ref> gen(const Settings &settings,
                   SpanSized<const Triangle> triangles);

private:
  // PIMPL
  class Generator;

  std::unique_ptr<Generator> gen_;
};

// NOTE: this could be more general, but it doesn't need to be right now.
// We would need some notion in which objects could be "clipped".
template <ExecutionModel exec>
struct IsTriangleAccel
    : std::bool_constant<TriangleAccel<SBVH<exec>, Settings>> {};

static_assert(PredicateForAllValues<ExecutionModel>::value<IsTriangleAccel>);
} // namespace sbvh
} // namespace accel
} // namespace intersect
