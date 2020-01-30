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
struct WorkingDivision {
  std::array<unsigned, 3> starts;
  std::array<unsigned, 3> ends;
  unsigned num_outstanding;

  HOST_DEVICE WorkingDivision(unsigned start_x_edges, unsigned end_x_edges,
                              unsigned start_y_edges, unsigned end_y_edges,
                              unsigned start_z, unsigned end_z,
                              unsigned num_outstanding)
      : starts({start_x_edges, start_y_edges, start_z}),
        ends({end_z, end_x_edges, end_y_edges}),
        num_outstanding(num_outstanding) {}

  HOST_DEVICE WorkingDivision(unsigned start_edges, unsigned end_edges,
                              unsigned start_z, unsigned end_z)
      : WorkingDivision(start_edges, end_edges, start_edges, end_edges, start_z,
                        end_z, 0) {}

  HOST_DEVICE WorkingDivision() {}

  HOST_DEVICE std::array<unsigned, 3> sizes() const {
    return {ends[0] - starts[0], ends[1] - starts[1], ends[2] - starts[2]};
  }
};

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

  struct AllEdges {
    DataType<float> other_min;
    DataType<float> other_max;
    DataType<float> value;
    DataType<uint8_t> is_min;

    void resize_all(unsigned size) {
      other_min.resize(size);
      other_max.resize(size);
      value.resize(size);
      is_min.resize(size);
    }
  };

  AllEdges x_edges_;
  AllEdges y_edges_;
  DataType<IdxAABB> sorted_by_z_min_;
  DataType<IdxAABB> sorted_by_z_max_;

  AllEdges x_edges_working_;
  AllEdges y_edges_working_;
  DataType<IdxAABB> sorted_by_z_min_working_;
  DataType<IdxAABB> sorted_by_z_max_working_;

  DataType<unsigned> x_edges_keys_;
  DataType<unsigned> y_edges_keys_;
  DataType<unsigned> z_keys_;

  DataType<uint64_t> x_edges_min_max_prefixes_;
  DataType<uint64_t> y_edges_min_max_prefixes_;

  DataType<WorkingDivision> divisions_;
  std::vector<WorkingDivision> divisions_cpu_;
  DataType<DirTreeNode> nodes_;

  unsigned num_shapes_;

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
