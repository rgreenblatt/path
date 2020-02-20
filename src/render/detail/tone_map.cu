#include "render/detail/tone_map.h"

#include <thrust/transform.h>

namespace render {
namespace detail {
template <ExecutionModel execution_model>
void tone_map(SpanSized<const Eigen::Array3f> intensities,
              SpanSized<RGBA> rgb) {
  auto start_it = thrust::make_counting_iterator(0u);

  unsigned size = rgb.size();
  unsigned intensities_per_output = intensities.size() / size;

  // SPEED: could be parallel on cpu
  thrust::transform(ThrustData<execution_model>().execution_policy(), start_it,
                    start_it + size, rgb.begin(),
                    [=] __host__ __device__(unsigned idx) {
                      Eigen::Array3f sum = Eigen::Vector3f::Zero();

                      for (unsigned i = 0; i < intensities_per_output; i++) {
                        sum += intensities[i + idx * intensities_per_output];
                      }

                      return intensity_to_rgb(sum);
                    });
}

template void
tone_map<ExecutionModel::GPU>(SpanSized<const Eigen::Array3f> intensities,
                              SpanSized<RGBA> rgb);
template void
tone_map<ExecutionModel::CPU>(SpanSized<const Eigen::Array3f> intensities,
                              SpanSized<RGBA> rgb);
} // namespace detail
} // namespace render
