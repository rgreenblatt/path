#pragma once

#include "lib/cuda/unified_memory_vector.h"
#include "lib/execution_model.h"
#include "lib/execution_model_datatype.h"
#include "lib/thrust_data.h"
#include "ray/detail/accel/dir_tree/bounding_points.h"
#include "ray/detail/accel/dir_tree/dir_tree.h"
#include "ray/detail/accel/dir_tree/idx_aabb.h"
#include "ray/detail/accel/dir_tree/sphere_partition.h"
#include "scene/light.h"
#include "scene/shape_data.h"

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model> class DirTreeGenerator {
public:
  DirTreeGenerator() {}

  DirTreeLookup get_dir_trees(const Eigen::Projective3f &world_to_film,
                              SpanSized<const scene::ShapeData> shapes,
                              SpanSized<const scene::Light> lights,
                              const Eigen::Vector3f &min_bound,
                              const Eigen::Vector3f &max_bound);

private:
  void setup_dir_tree(const Eigen::Projective3f &world_to_film,
                      SpanSized<const scene::Light> lights,
                      const Eigen::Vector3f &min_bound,
                      const Eigen::Vector3f &max_bound);

  void compute_aabbs();

  void copy_to_sortable(unsigned num_shapes);

  void fill_indexes();

  void sort();

  void permute();

  void construct_trees();

  template <typename T> using DataType = DataType<execution_model, T>;

  ManangedMemVec<HalfSpherePartition::Region> sphere_partition_regions_;

  ManangedMemVec<Eigen::Projective3f> transforms_;
  ManangedMemVec<BoundingPoints> bounds_;
  DataType<IdxAABB> aabbs_;

  // x edges, y edges, z min, z max
  static constexpr unsigned num_sortings = 4;
  ManangedMemVec<Eigen::Vector3f> sort_offsets_;
  std::array<DataType<float>, num_sortings> sorting_values_;
  std::array<DataType<unsigned>, num_sortings> indexes_;

  struct Edge {
    float other_min;
    float other_max;
    float value;
    bool is_min;

    HOST_DEVICE Edge(float other_min, float other_max, float value, bool is_min)
        : other_min(other_min), other_max(other_max), value(value),
          is_min(is_min) {}

    HOST_DEVICE Edge() {}
  };

  DataType<Edge> sorted_by_x_edges_;
  DataType<Edge> sorted_by_y_edges_;
  DataType<IdxAABB> sorted_by_z_min_;
  DataType<IdxAABB> sorted_by_z_max_;

  DataType<Edge> sorted_by_x_edges_working_;
  DataType<Edge> sorted_by_y_edges_working_;
  DataType<IdxAABB> sorted_by_z_min_working_;
  DataType<IdxAABB> sorted_by_z_max_working_;

  struct WorkingDivision {
    unsigned start_x_edges;
    unsigned end_x_edges;
    unsigned start_y_edges;
    unsigned end_y_edges;
    unsigned start_z_min;
    unsigned end_z_min;
    unsigned start_z_max;
    unsigned end_z_max;
  };

  DataType<WorkingDivision> divisions_;
  DataType<DirTreeNode> nodes_;

#if 0
  DataType<unsigned> x_edges_is_min;
  DataType<unsigned> y_edges_is_min;
  DataType<unsigned> sorted_by_y_edges_working_;
  DataType<unsigned> sorted_by_z_min_working_;
  DataType<unsigned> sorted_by_z_max_working_;
#endif

  std::vector<ThrustData<execution_model>> thrust_data_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
