#include "intersect/ray.h"
#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <Eigen/Core>
#include <thrust/optional.h>

namespace intersect {
namespace accel {
// TODO: span or custom?
template <typename Object> class LoopAll {
public:
  LoopAll(Span<const Object> objects, unsigned start, unsigned end)
      : objects_(objects), start_(start), end_(end) {}

  using Intersection = decltype(std::declval<Object>()(Ray()));

  template <typename SolveIndex>
  HOST_DEVICE Intersection operator()(const Ray &ray) const {
    Intersection best_intersection;
    for (unsigned i = start_; i < end_; i++) {
      best_intersection = std::min(best_intersection, objects_[i](ray));
    }

    return best_intersection;
  }

private:
  Span<const Object> objects_;
  unsigned start_;
  unsigned end_;
};
} // namespace accel
} // namespace intersect
