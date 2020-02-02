#pragma once

#include "lib/cuda/managed_mem_vec.h"
#include "lib/execution_model.h"
#include "lib/execution_model_vector_type.h"
#include "lib/span.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/thrust_data.h"
#include "lib/vector_group.h"
#include "ray/detail/accel/dir_tree/best_edge.h"
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
template <ExecutionModel execution_model> class DirTreeGeneratorImpl {
public:
  DirTreeGeneratorImpl()
      : is_x_(true),
        x_edges_(x_edges_underlying_.first, x_edges_underlying_.second),
        y_edges_(y_edges_underlying_.first, y_edges_underlying_.second),
        sorted_by_z_min_(sorted_by_z_min_underlying_.first,
                         sorted_by_z_min_underlying_.second),
        sorted_by_z_max_(sorted_by_z_max_underlying_.first,
                         sorted_by_z_max_underlying_.second),
        current_edges_(x_edges_underlying_.first),
        other_edges_(y_edges_underlying_.first),
        other_edges_new_(y_edges_underlying_.second),
        current_edges_keys_(x_edges_keys_),
        other_edges_keys_(y_edges_keys_),
        axis_groups_(axis_groups_underlying_.first,
                     axis_groups_underlying_.second),
        better_than_no_split_(better_than_no_split_underlying_.first,
                              better_than_no_split_underlying_.second) {}

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

  void scan_edges();

  void find_best_edges();

  void test_splits();

  void filter_others();

  template <typename T> using ExecVecT = ExecVectorType<execution_model, T>;

  HostDeviceVectorType<HalfSpherePartition::Region> sphere_partition_regions_;

  HostDeviceVectorType<Eigen::Projective3f> transforms_;
  HostDeviceVectorType<BoundingPoints> bounds_;
  ExecVecT<IdxAABB> aabbs_;

  // x edges, y edges, z min, z max
  static constexpr unsigned num_sortings = 4;
  HostDeviceVectorType<Eigen::Vector3f> sort_offsets_;
  std::array<ExecVecT<float>, num_sortings> sorting_values_;
  std::array<ExecVecT<unsigned>, num_sortings> indexes_;

  class AllEdges : public VectorGroup<ExecVecT, float, float, float, uint8_t> {
  public:
    Span<float> other_mins() { return this->template get<0>(); }
    Span<float> other_maxs() { return this->template get<1>(); }
    Span<float> values() { return this->template get<2>(); }
    Span<uint8_t> is_mins() { return this->template get<3>(); }
  };

  // maybe one of z_min/z_max could be eliminated
  class ZValues : public VectorGroup<ExecVecT, float, float, float, float,
                                     float, float, unsigned> {
  public:
    Span<float> x_mins() { return this->template get<0>(); }
    Span<float> x_maxs() { return this->template get<1>(); }
    Span<float> y_mins() { return this->template get<2>(); }
    Span<float> y_maxs() { return this->template get<3>(); }
    Span<float> z_mins() { return this->template get<4>(); }
    Span<float> z_maxs() { return this->template get<5>(); }
    Span<unsigned> idxs() { return this->template get<6>(); }
  };

  template <typename T>
  using Pair = std::pair<T, T>;

  Pair<AllEdges> x_edges_underlying_;
  Pair<AllEdges> y_edges_underlying_;
  Pair<ZValues> sorted_by_z_min_underlying_;
  Pair<ZValues> sorted_by_z_max_underlying_;

  template <typename T> class RefT : public std::reference_wrapper<T> {
  public:
    RefT(T &v) : std::reference_wrapper<T>(v) {}
    T *operator->() { return &this->get(); }
    const T *operator->() const { return &this->get(); }
  };

  bool is_x_;

  Pair<RefT<AllEdges>> x_edges_;
  Pair<RefT<AllEdges>> y_edges_;
  Pair<RefT<ZValues>> sorted_by_z_min_;
  Pair<RefT<ZValues>> sorted_by_z_max_;

  RefT<AllEdges> current_edges_;
  RefT<AllEdges> other_edges_;
  RefT<AllEdges> other_edges_new_;

  ExecVecT<unsigned> x_edges_keys_;
  ExecVecT<unsigned> y_edges_keys_;
  ExecVecT<unsigned> z_keys_;

  RefT<ExecVecT<unsigned>> current_edges_keys_;
  RefT<ExecVecT<unsigned>> other_edges_keys_;

  template <template <typename> class VecT>
  using AxisGroups = VectorGroup<VecT, unsigned, unsigned, unsigned>;

  // inclusive scan...
  AxisGroups<HostVectorType> axis_groups_cpu_;

  Pair<AxisGroups<ExecVecT>> axis_groups_underlying_;

  Pair<RefT<AxisGroups<ExecVecT>>> axis_groups_;

  inline unsigned num_groups() const { return axis_groups_.first->size(); }

  inline unsigned get_current_edge_idx() {
    return is_x_ ? 0 : 1;
  }

  inline unsigned get_other_edge_idx() {
    return is_x_ ? 1 : 0;
  }

  inline Span<unsigned> current_edges_groups() {
    return axis_groups_.first.get()[get_current_edge_idx()];
  }
  
  inline Span<unsigned> other_edges_groups() {
    return axis_groups_.first.get()[get_other_edge_idx()];
  }

  // some subset of these may need to be made into pairs...
  ExecVecT<unsigned> open_mins_before_group_;
  ExecVecT<unsigned> num_per_group_;
  ExecVecT<unsigned> starts_inclusive_;
  ExecVecT<BestEdge> best_edges_;

  Pair<ExecVecT<uint8_t>> better_than_no_split_underlying_;
  
  Pair<RefT<ExecVecT<uint8_t>>> better_than_no_split_;

  ExecVecT<unsigned> new_edge_indexes_;
  ExecVecT<unsigned> new_z_min_indexes_;
  ExecVecT<unsigned> new_z_max_indexes_;

  ExecVecT<DirTreeNode> nodes_;

  unsigned num_shapes_;

  std::vector<ThrustData<execution_model>> thrust_data_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
