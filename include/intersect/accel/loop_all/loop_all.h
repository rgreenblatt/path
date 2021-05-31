#pragma once

#include "execution_model/execution_model_vector_type.h"
#include "intersect/accel/accel.h"
#include "intersect/accel/add_idx.h"
#include "intersect/accel/loop_all/settings.h"
#include "intersect/optional_min.h"
#include "lib/attribute.h"
#include "lib/vector_type.h"
#include "meta/all_values/predicate_for_all_values.h"

namespace intersect {
namespace accel {
namespace loop_all {
struct LoopAll {
  struct Ref {
    unsigned size;

    template <IntersectableAtIdx F>
    HOST_DEVICE inline AccelRet<F>
    intersect_objects(const intersect::Ray &ray,
                      const F &intersectable_at_idx) const {
      AccelRet<F> best;

      for (unsigned idx = 0; idx < size; idx++) {
        best = optional_min(best, add_idx(intersectable_at_idx(idx, ray), idx));
      }

      return best;
    }
  };

  template <typename B>
  RefPerm<Ref> gen(const Settings &, SpanSized<const B> objects) {
    VectorT<unsigned> permutation(objects.size());
    for (unsigned i = 0; i < objects.size(); ++i) {
      permutation[i] = i;
    }
    return {.ref = Ref{static_cast<unsigned>(objects.size())},
            .permutation = permutation};
  }
};

static_assert(BoundsOnlyAccel<LoopAll, Settings>);
} // namespace loop_all
} // namespace accel
} // namespace intersect
