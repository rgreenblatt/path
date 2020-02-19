#include "intersect/accel/loop_all.h"
#include "lib/compile_time_dispatch/dispatch_value.h"
#include "lib/compile_time_dispatch/tuple.h"
#include "lib/group.h"
#include "lib/info/timer.h"
#include "lib/span.h"
#include "render/detail/compute_intensities.h"
#include "render/detail/divide_work.h"
#include "render/detail/renderer_impl.h"
#include "render/detail/tone_map.h"
#include "scene/camera.h"

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/combine.hpp>
#include <thrust/copy.h>
#include <thrust/fill.h>

namespace render {
namespace detail {
template <ExecutionModel execution_model>
RendererImpl<execution_model>::RendererImpl() {}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(
    Span<RGBA> pixels, const scene::Scene &s, unsigned x_dim, unsigned y_dim,
    unsigned samples_per, PerfSettings settings, bool show_times) {
  unsigned block_size = 512;
  unsigned target_work_per_thread = 4;

  if (samples_per > std::numeric_limits<uint16_t>::max()) {
    std::cerr << "more samples than allowed" << std::endl;
  }

  auto division = divide_work(samples_per, x_dim, y_dim, block_size,
                              target_work_per_thread);
  unsigned required_size = division.num_sample_blocks * x_dim * y_dim;

  if (division.num_sample_blocks != 1) {
    intermediate_intensities_.resize(required_size);
  } else {
    intermediate_intensities_.clear();
  }

  final_intensities_.resize(x_dim * y_dim);

  Span<const scene::TriangleData> triangle_data;
  Span<const scene::Material> materials;

  if constexpr (execution_model == ExecutionModel::GPU) {
    auto inp_t_data = s.triangle_data();
    auto inp_materials = s.materials();

    triangle_data_.resize(inp_t_data.size());
    materials_.resize(inp_materials.size());

    thrust::copy(inp_t_data.begin(), inp_t_data.end(), triangle_data_.begin());
    thrust::copy(inp_materials.begin(), inp_materials.end(),
                 materials_.begin());

    triangle_data = triangle_data_;
    materials = materials_;
  } else {
    triangle_data = s.triangle_data();
    materials = s.materials();
  }

#if 0
  const float dir_tree_triangle_traversal_cost = 1;
  const float dir_tree_triangle_intersection_cost = 1;
  const unsigned num_dir_trees_triangle = 16;

  const float kd_tree_triangle_traversal_cost = 1;
  const float kd_tree_triangle_intersection_cost = 1;

  const float dir_tree_mesh_traversal_cost = 1;
  const float dir_tree_mesh_intersection_cost = 1;
  const unsigned num_dir_trees_mesh = 16;

  const float kd_tree_mesh_traversal_cost = 1;
  const float kd_tree_mesh_intersection_cost = 1;
#endif

  Span<RGBA> output_pixels;
  if constexpr (execution_model == ExecutionModel::GPU) {
    bgra_.resize(x_dim * y_dim);
    output_pixels = bgra_;
  } else {
    output_pixels = pixels;
  }

  tone_map<execution_model>(x_dim, y_dim, final_intensities_, output_pixels);

  if constexpr (execution_model == ExecutionModel::GPU) {
    thrust::copy(bgra_.begin(), bgra_.end(), pixels.begin());
  }
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace render
