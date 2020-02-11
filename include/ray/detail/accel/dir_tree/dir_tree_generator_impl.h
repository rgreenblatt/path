#pragma once

#include "lib/cuda/managed_mem_vec.h"
#include "lib/execution_model.h"
#include "lib/execution_model_vector_type.h"
#include "lib/reference_type.h"
#include "lib/span.h"
#include "lib/span_convertable_device_vector.h"
#include "lib/span_convertable_vector.h"
#include "lib/thrust_data.h"
#include "lib/vector_group.h"
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
  DirTreeGeneratorImpl();

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

  /*
   * Fills keys using groups.
   *
   * The gpu version computes a max group size per dimension and then fills in
   * keys using a grid.
   *
   * Kernel launches: 2 (first gets launched async for each dim)
   * Uses:
   *  - axis_groups_
   * Fills in:
   *  - x_edges_keys_
   *  - y_edges_keys_
   *  - z_keys_
   */
  void fill_keys();

  /*
   * Computes inclusive sum of is mins of the current edges. (inclusive scan by
   * key)
   *
   * Kernel launches: 1
   * Uses:
   *  - current_edges_->is_mins()
   *  - current_edges_keys_ (either x_edges_keys_ or y_edges_keys_)
   * Fills in:
   *  - starts_inclusive_
   */
  void scan_edges();

  /*
   * Finds edges with lowest surface area heuristic cost in each group.
   *
   * Kernel launches: 1
   * Uses:
   *  - current_edges_keys_ (either x_edges_keys_ or y_edges_keys_)
   *  - current_edges_groups() (axis_groups_)
   *  - starts_inclusive_
   *  - current_edges->values()
   *  - current_edges->is_mins()
   *  - open_mins_before_group_
   *  - num_per_group_
   * Fills in:
   *  - best_edges_
   */
  void find_best_edges();

  /*
   * Checks if each split is better than no split
   *
   * Kernel launches: 1
   * Uses:
   *  - best_edges_
   *  - num_per_group_
   * Fills in:
   *  - better_than_no_split_
   */
  void test_splits();

  /*
   * Filters other edge and z values into new vectors
   *
   * Kernel launchs: 2
   * Uses:
   *  - best_edges_
   *  - current_edges_->values()
   *  - other_edges_
   *  - other_edges_keys_
   *  - other_edges_groups()
   *  - better_than_no_split_
   *  - z_keys_
   *  - sorted_by_z_min_
   *  - sorted_by_z_max_
   *  - axis_groups_.first.get()[2]
   * Fills:
   *  - new_edge_indexes_
   *  - other_edges_new_
   *  - new_z_min_indexes_
   *  - new_z_max_indexes_
   * Swaps:
   *  - current_edges_ <- other_edges_new_
   *  - other_edges_ <- current_edges_
   *  - current_edges_keys_ <-> other_edges_keys_ (not strictly needed I think)
   *  - sorted_by_z_min_.first <-> sorted_by_z_min_.second
   *  - sorted_by_z_max_.first <-> sorted_by_z_max_.second
   */
  void filter_others();

  /*
   * Sets up groups for the next iteration
   *
   * Kernel launchs: 2
   * Uses:
   *  - better_than_no_split_
   *  - axis_groups_
   *  - open_mins_before_group_
   *  - num_per_group_
   *  - new_edge_indexes_
   *  - new_z_min_indexes_
   *  - current_edges_->is_mins()
   *  - best_edges_
   *  - starts_inclusive_
   * Fills:
   *  - num_groups_before_
   *  - axis_groups_
   *  - open_mins_before_group_
   *  - num_per_group_
   *  - better_than_no_split_
   *  - nodes_
   * Swaps:
   *  - node_offset_ <- new_node_offset
   *  - axis_groups_.first <-> axis_groups_.second
   *  - open_mins_before_group_.first <-> open_mins_before_group_.second
   *  - num_per_group_.first <-> num_per_group_.second
   *  - better_than_no_split_.first <-> better_than_no_split_.second
   */
  void setup_groups();

  template <typename T> using ExecVecT = ExecVector<execution_model, T>;

  HostDeviceVector<HalfSpherePartition::ColatitudeDiv>
      sphere_partition_regions_;

  HostDeviceVector<Eigen::Projective3f> transforms_;
  HostDeviceVector<BoundingPoints> bounds_;
  ExecVecT<IdxAABB> aabbs_;

  // x edges, y edges, z min, z max
  static constexpr unsigned num_sortings = 4;
  HostDeviceVector<Eigen::Vector3f> sort_offsets_;
  std::array<ExecVecT<float>, num_sortings> sorting_values_;
  std::array<ExecVecT<unsigned>, num_sortings> indexes_;

  class AllEdges : public VectorGroup<ExecVecT, float, float, float, uint8_t> {
  public:
    SpanSized<float> other_mins() { return this->template get<0>(); }
    SpanSized<float> other_maxs() { return this->template get<1>(); }
    SpanSized<float> values() { return this->template get<2>(); }
    SpanSized<uint8_t> is_mins() { return this->template get<3>(); }
  };

  // maybe one of z_min/z_max could be eliminated
  class ZValues : public VectorGroup<ExecVecT, float, float, float, float,
                                     float, float, unsigned> {
  public:
    SpanSized<float> x_mins() { return this->template get<0>(); }
    SpanSized<float> x_maxs() { return this->template get<1>(); }
    SpanSized<float> y_mins() { return this->template get<2>(); }
    SpanSized<float> y_maxs() { return this->template get<3>(); }
    SpanSized<float> z_mins() { return this->template get<4>(); }
    SpanSized<float> z_maxs() { return this->template get<5>(); }
    SpanSized<unsigned> idxs() { return this->template get<6>(); }
  };

  template <typename T> using Pair = std::pair<T, T>;

  std::array<AllEdges, 3> edges_underlying_;
  Pair<ZValues> sorted_by_z_min_underlying_;
  Pair<ZValues> sorted_by_z_max_underlying_;

  bool use_async_;

  bool is_x_;

  RefT<AllEdges> current_edges_;
  RefT<AllEdges> other_edges_;
  RefT<AllEdges> other_edges_new_;
  Pair<RefT<ZValues>> sorted_by_z_min_;
  Pair<RefT<ZValues>> sorted_by_z_max_;

  std::array<ExecVecT<std::array<float, 2>>, 4> group_min_max_underlying_;

  RefT<ExecVecT<std::array<float, 2>>> current_edges_min_max_;
  RefT<ExecVecT<std::array<float, 2>>> other_edges_min_max_;
  RefT<ExecVecT<std::array<float, 2>>> current_edges_min_max_new_;
  RefT<ExecVecT<std::array<float, 2>>> other_edges_min_max_new_;

  ExecVecT<unsigned> x_edges_keys_;
  ExecVecT<unsigned> y_edges_keys_;
  ExecVecT<unsigned> z_keys_;

  RefT<ExecVecT<unsigned>> current_edges_keys_;
  RefT<ExecVecT<unsigned>> other_edges_keys_;

  template <template <typename> class VecT>
  using AxisGroups = VectorGroup<VecT, unsigned, unsigned, unsigned>;

  // inclusive scan...
  AxisGroups<HostVector> axis_groups_cpu_;

  Pair<AxisGroups<ExecVecT>> axis_groups_underlying_;

  Pair<RefT<AxisGroups<ExecVecT>>> axis_groups_;

  inline unsigned num_groups() const { return axis_groups_.first->size(); }

  inline unsigned get_current_edge_idx() { return is_x_ ? 0 : 1; }

  inline unsigned get_other_edge_idx() { return is_x_ ? 1 : 0; }

  inline SpanSized<unsigned> current_edges_groups() {

    return axis_groups_.first.get()[get_current_edge_idx()];
  }

  inline SpanSized<unsigned> other_edges_groups() {

    return axis_groups_.first.get()[get_other_edge_idx()];
  }

  inline SpanSized<unsigned> current_edges_new_groups() {

    return axis_groups_.second.get()[get_current_edge_idx()];
  }

  inline SpanSized<unsigned> other_edges_new_groups() {

    return axis_groups_.second.get()[get_other_edge_idx()];
  }

  ExecVecT<unsigned> starts_inclusive_;

  Pair<ExecVecT<unsigned>> open_mins_before_group_underlying_;
  Pair<ExecVecT<unsigned>> num_per_group_underlying_;

  Pair<RefT<ExecVecT<unsigned>>> open_mins_before_group_;
  Pair<RefT<ExecVecT<unsigned>>> num_per_group_;

  class BestEdges : public VectorGroup<ExecVecT, float, unsigned, uint8_t> {
  public:
    SpanSized<float> costs() { return this->template get<0>(); }
    SpanSized<unsigned> idxs() { return this->template get<1>(); }
    SpanSized<uint8_t> side_of_size_zero() { return this->template get<2>(); }
  };

  BestEdges best_edges_;

  Pair<ExecVecT<uint8_t>> better_than_no_split_underlying_;

  Pair<RefT<ExecVecT<uint8_t>>> better_than_no_split_;

  ExecVecT<unsigned> new_edge_indexes_;
  ExecVecT<unsigned> new_z_min_indexes_;
  ExecVecT<unsigned> new_z_max_indexes_;

  ExecVecT<unsigned> num_groups_inclusive_;
  ExecVecT<unsigned> z_outputs_inclusive_;

  unsigned node_offset_;
  unsigned output_values_offset_;

  ExecVecT<DirTreeNode> nodes_;

  ExecVecT<unsigned> output_keys_;

  ExecVecT<float> min_sorted_values_;
  ExecVecT<float> min_sorted_inclusive_maxes_;
  ExecVecT<unsigned> min_sorted_indexes_;

  ExecVecT<float> max_sorted_values_;
  ExecVecT<float> max_sorted_inclusive_mins_;
  ExecVecT<unsigned> max_sorted_indexes_;

  HostDeviceVector<DirTree> dir_trees_;

  unsigned num_shapes_;

  std::vector<ThrustData<execution_model>> thrust_data_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
