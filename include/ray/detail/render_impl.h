#pragma once

#include "lib/bgra.h"
#include "lib/cuda/managed_mem_vec.h"
#include "lib/execution_model.h"
#include "lib/execution_model_vector_type.h"
#include "ray/detail/accel/dir_tree/dir_tree_generator.h"
#include "ray/detail/accel/kdtree/kdtree.h"
#include "ray/detail/block_data.h"
#include "scene/scene.h"

#include <thrust/device_vector.h>

namespace ray {
namespace detail {
template <ExecutionModel execution_model> class RendererImpl {
public:
  void render(BGRA *pixels, const Eigen::Affine3f &m_film_to_world,
              const Eigen::Projective3f &world_to_film, bool use_kd_tree,
              bool use_dir_tree, bool show_times);

  RendererImpl(unsigned x_dim, unsigned y_dim, unsigned super_sampling_rate,
               unsigned recursive_iterations, std::unique_ptr<scene::Scene> &s);

  scene::Scene &get_scene() { return *scene_; }

  const scene::Scene &get_scene() const { return *scene_; }

private:
  template <bool is_first, typename Accel>
  void raytrace_pass(const Accel &accel, unsigned current_num_blocks,
                     SpanSized<const scene::ShapeData> shapes,
                     SpanSized<const scene::Light> lights,
                     Span<const scene::TextureImageRef> textures);

  void fill(const scene::Color &initial_multiplier,
            const scene::Color &initial_color,
            const Eigen::Affine3f &m_film_to_world);

#if 0
  void initial_world_space_directions(const Eigen::Affine3f &m_film_to_world);
#endif

  void float_to_bgra(BGRA *pixels, Span<const scene::Color> colors);

  template <typename T> using ExecVecT = ExecVector<execution_model, T>;
  template <typename T> using SharedVecT = SharedVector<execution_model, T>;

  const BlockData block_data_;
  unsigned real_x_dim_;
  unsigned real_y_dim_;
  const unsigned super_sampling_rate_;

  unsigned recursive_iterations_;

  bool show_times_;

  std::unique_ptr<scene::Scene> scene_;

  SharedVecT<accel::kdtree::KDTreeNode<accel::AABB>> kdtree_nodes_;
  SharedVecT<accel::kdtree::KDTreeNode<accel::AABB>> sort_nodes_;

  ExecVecT<Eigen::Vector3f> world_space_eyes_;
  ExecVecT<Eigen::Vector3f> world_space_directions_;
  ExecVecT<unsigned> ignores_;
  ExecVecT<scene::Color> color_multipliers_;
  ExecVecT<uint8_t> disables_;
  ExecVecT<scene::Color> colors_;
  ExecVecT<BGRA> bgra_;
  SharedVecT<uint8_t> group_disables_;
  SharedVecT<unsigned> group_indexes_;

  accel::dir_tree::DirTreeGenerator<execution_model> dir_tree_generator_;
};
} // namespace detail
} // namespace ray
