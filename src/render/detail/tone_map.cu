#include "render/detail/tone_map.h"

#include <thrust/transform.h>

namespace render {
namespace detail {
template <ExecutionModel execution_model>
void tone_map(SpanSized<const Eigen::Array3f> intensities,
              SpanSized<BGRA> bgr) {
  auto start_it = thrust::make_counting_iterator(0u);

  unsigned size = bgr.size();
  unsigned intensities_per_output = intensities.size() / size;

  // SPEED: could be parallel on cpu
  thrust::transform(ThrustData<execution_model>().execution_policy(), start_it,
                    start_it + size, bgr.begin(),
                    [=] __host__ __device__(unsigned idx) {
                      Eigen::Array3f sum = Eigen::Vector3f::Zero();

                      for (unsigned i = 0; i < intensities_per_output; i++) {
                        sum += intensities[i + idx * intensities_per_output];
                      }

                      return intensity_to_bgr(sum);
                    });
}

template void
tone_map<ExecutionModel::GPU>(SpanSized<const Eigen::Array3f> intensities,
                              SpanSized<BGRA> bgr);
template void
tone_map<ExecutionModel::CPU>(SpanSized<const Eigen::Array3f> intensities,
                              SpanSized<BGRA> bgr);
} // namespace detail
} // namespace render
