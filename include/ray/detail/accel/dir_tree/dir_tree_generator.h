#pragma once

#include "lib/cuda/managed_mem_vec.h"
#include "lib/execution_model.h"
#include "lib/execution_model_vector_type.h"
#include "lib/span.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/vector_group.h"
#include "lib/thrust_data.h"
#include "ray/detail/accel/dir_tree/bounding_points.h"
#include "ray/detail/accel/dir_tree/dir_tree.h"
#include "ray/detail/accel/dir_tree/idx_aabb.h"
#include "ray/detail/accel/dir_tree/sphere_partition.h"
#include "ray/detail/accel/dir_tree/best_edge.h"
#include "scene/light.h"
#include "scene/shape_data.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model> class DirTreeGenerator {
public:
  DirTreeGenerator() {}

  DirTreeLookup generate(const Eigen::Projective3f &world_to_film,
                         SpanSized<const scene::ShapeData> shapes,
                         SpanSized<const scene::Light> lights,
                         const Eigen::Vector3f &min_bound,
                         const Eigen::Vector3f &max_bound);

private:
  HalfSpherePartition setup(const Eigen::Projective3f &world_to_film,
                            SpanSized<const scene::Light> lights,
                            const Eigen::Vector3f &min_bound,
                            const Eigen::Vector3f &max_bound);

  void compute_aabbs();

  void copy_to_sortable();

  void fill_indexes();

  void sort();

  void permute();

  void construct();

  void fill_keys();
  
  void scan_edges(bool is_x);

  void find_best_edges(bool is_x);

  template <typename T> using ExecVecT = ExecVectorType<execution_model, T>;

  ManangedMemVec<HalfSpherePartition::Region> sphere_partition_regions_;

  ManangedMemVec<Eigen::Projective3f> transforms_;
  ManangedMemVec<BoundingPoints> bounds_;
  ExecVecT<IdxAABB> aabbs_;

  // x edges, y edges, z min, z max
  static constexpr unsigned num_sortings = 4;
  ManangedMemVec<Eigen::Vector3f> sort_offsets_;
  std::array<ExecVecT<float>, num_sortings> sorting_values_;
  std::array<ExecVecT<unsigned>, num_sortings> indexes_;

  using AllEdges = VectorGroup<ExecVecT, float, float, float, uint8_t>;

  AllEdges x_edges_;
  AllEdges y_edges_;
  ExecVecT<IdxAABB> sorted_by_z_min_;
  ExecVecT<IdxAABB> sorted_by_z_max_;

  AllEdges x_edges_working_;
  AllEdges y_edges_working_;
  ExecVecT<IdxAABB> sorted_by_z_min_working_;
  ExecVecT<IdxAABB> sorted_by_z_max_working_;

  ExecVecT<unsigned> x_edges_keys_;
  ExecVecT<unsigned> y_edges_keys_;
  ExecVecT<unsigned> z_keys_;

  template <template <typename> class VecT>
  using Groups = VectorGroup<VecT, unsigned, unsigned, unsigned>;

  // inclusive scan...
  Groups<ExecVecT> groups_;
  Groups<HostVectorType> groups_cpu_;
  
  VectorGroup<ExecVecT, unsigned, unsigned> diffs_before_group_;

  ExecVecT<unsigned> num_per_group_;
  ExecVecT<unsigned> starts_inclusive_;
  ExecVecT<BestEdge> best_edges_;

  ExecVecT<DirTreeNode> nodes_;

  unsigned num_shapes_;

#if 0
  ExecVecT<unsigned> x_edges_is_min;
  ExecVecT<unsigned> y_edges_is_min;
  ExecVecT<unsigned> sorted_by_y_edges_working_;
  ExecVecT<unsigned> sorted_by_z_min_working_;
  ExecVecT<unsigned> sorted_by_z_max_working_;
#endif

  std::vector<ThrustData<execution_model>> thrust_data_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
