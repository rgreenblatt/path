#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "ray/detail/accel/kdtree/kdtree_ref.h"
#include "ray/detail/accel/loop_all.h"
#include "ray/detail/render_impl.h"
#include "scene/camera.h"

#include "lib/timer.h"
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/combine.hpp>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <chrono>

namespace ray {
namespace detail {
template <ExecutionModel execution_model>
RendererImpl<execution_model>::RendererImpl(unsigned x_dim, unsigned y_dim,
                                            unsigned super_sampling_rate,
                                            unsigned recursive_iterations,
                                            std::unique_ptr<scene::Scene> &s)
    : block_data_(super_sampling_rate * x_dim, super_sampling_rate * y_dim, 32,
                  8),
      real_x_dim_(x_dim), real_y_dim_(y_dim),
      super_sampling_rate_(super_sampling_rate),
      recursive_iterations_(recursive_iterations), show_times_(false),
      scene_(std::move(s)), world_space_eyes_(block_data_.totalSize()),
      world_space_directions_(block_data_.totalSize()),
      ignores_(block_data_.totalSize()),
      color_multipliers_(block_data_.totalSize()),
      disables_(block_data_.totalSize()), colors_(block_data_.totalSize()),
      bgra_(real_x_dim_ * real_y_dim_) {}

template <ExecutionModel execution_model>
void RendererImpl<execution_model>::render(
    BGRA *pixels, const Eigen::Affine3f &m_film_to_world,
    const Eigen::Projective3f &world_to_film, bool use_kd_tree,
    bool use_dir_tree, bool show_times) {
  const auto lights = scene_->getLights();
  const unsigned num_lights = scene_->getNumLights();
  const auto textures = scene_->getTextures();
  const unsigned num_textures = scene_->getNumTextures();
  show_times_ = show_times;

  const unsigned general_num_blocks = block_data_.generalNumBlocks();

  group_disables_.resize(general_num_blocks);
  group_indexes_.resize(general_num_blocks);

  unsigned current_num_blocks = general_num_blocks;

  Timer fill_timer;

  // could be made async until...
  fill(scene::Color::Ones(), scene::Color::Zero(), m_film_to_world);

  if (show_times_) {
    fill_timer.report("fill");
  }

  const unsigned num_shapes = scene_->getNumShapes();
  ManangedMemVec<scene::ShapeData> moved_shapes_(num_shapes);

  {
    auto start_shape = scene_->getShapes();
    std::copy(start_shape, start_shape + num_shapes, moved_shapes_.begin());
  }
  if (use_kd_tree
#if 0
      && !use_dir_tree
#endif
  ) {
    Timer kdtree_timer;

    auto kdtree = accel::kdtree::construct_kd_tree(moved_shapes_.data(),
                                                   num_shapes, 25, 3);
    kdtree_nodes_.resize(kdtree.size());
    std::copy(kdtree.begin(), kdtree.end(), kdtree_nodes_.begin());

    std::fill(group_disables_.begin(), group_disables_.end(), false);

    if (show_times_) {
      kdtree_timer.report("kdtree");
    }
  }

  SpanSized<const scene::Light> lights_span(lights, num_lights);

  if (use_dir_tree) {
    dir_tree_generator_.generate(world_to_film, moved_shapes_, lights_span,
                                 scene_->getMinBound(), scene_->getMinBound());
  }

  for (unsigned depth = 0; depth < recursive_iterations_; depth++) {
    bool is_first = depth == 0;

    current_num_blocks = 0;

    for (unsigned i = 0; i < group_disables_.size(); i++) {
      if (!group_disables_[i]) {
        group_indexes_[current_num_blocks] = i;
        current_num_blocks++;
      }
    }

    Timer intersect_timer;

    Span textures_span(textures, num_textures);

    auto raytrace = [&](const auto &data_structure) {
      if (is_first) {
        raytrace_pass<true>(data_structure, current_num_blocks, moved_shapes_,
                            lights_span, textures_span);
      } else {
        raytrace_pass<false>(data_structure, current_num_blocks, moved_shapes_,
                             lights_span, textures_span);
      }
    };

    if (use_kd_tree) {
      raytrace(accel::kdtree::KDTreeRef(kdtree_nodes_, moved_shapes_.size()));
    } else {
      raytrace(accel::LoopAll());
    }

    if (show_times) {
      intersect_timer.report("intersect");
    }
  }

  float_to_bgra(pixels, colors_);

#if 0
  auto draw_point = [&](unsigned x, unsigned y, BGRA color) {
    for (unsigned y_draw = std::max(y, 6u) - 6; y_draw < std::min(y, height - 6);
         y_draw++) {
      for (unsigned x_draw = std::max(x, 6u) - 6;
           x_draw < std::min(x, width - 6); x_draw++) {
        pixels[x_draw + y_draw * width] = color;
      }
    }
  };

  for (const auto &triangle : output_triangles) {
    for (const auto &point : triangle.points()) {
      draw_point(((point[0] + 1) / 2) * width, ((point[1] + 1) / 2) * height,
                 BGRA(200, 200, 200, 0));
    }
  }
#endif
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace ray
