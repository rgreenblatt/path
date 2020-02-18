#include "intersect/intersection.h"
#include "lib/cuda/utils.h"
#include "lib/span.h"

#include <Eigen/Core>
#include <thrust/optional.h>

namespace intersect {
namespace accel {
// TODO: generic over meshes and triangles...
class LoopAll {
public:
  LoopAll(unsigned num_shapes) : num_shapes_(num_shapes) {}

  template <typename SolveIndex>
  HOST_DEVICE void operator()(const Eigen::Vector3f &, const Eigen::Vector3f &,
                              const thrust::optional<BestIntersection> &,
                              const SolveIndex &solve_index) const {
    for (unsigned i = 0; i < num_shapes_; i++) {
      solve_index(i);
    }
  }

private:
  unsigned num_shapes_;
};
} // namespace accel
} // namespace intersect
