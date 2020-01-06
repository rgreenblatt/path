#include "ray/detail/render_impl.h"
#include "ray/detail/render_impl_utils.h"
#include "scene/camera.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace ray {
namespace detail {
  using namespace accel::dir_tree;

#if 0
struct CreateTraversal : public thrust::binary_function<int, int, Traversal> {
  HOST_DEVICE Traversal operator()(int start, int end) {
    return Traversal(start, end);
  }
};
#endif

template <ExecutionModel execution_model>
DirTreeLookup RendererImpl<execution_model>::dir_trees(
    const Eigen::Projective3f &world_to_film,
    Span<const scene::ShapeData, false> shapes,
    Span<const scene::Light, false> lights) {
  const auto start_setup_traversal_grid = current_time();

  unsigned region_target = 32;

  HalfSpherePartition partition(region_target, sphere_partition_regions_);

  dir_tree_transforms_.resize(1 + lights.size() + partition.size());

  unsigned transform_idx = 0;

  dir_tree_transforms_[0] = world_to_film;

  transform_idx++;

  auto add_transform =
      [&](bool is_loc, const Eigen::Vector3f &loc_or_dir) {
        if (is_loc) {
          dir_tree_transforms_[transform_idx] =
              scene::get_unhinging(30.0f) * Eigen::Translation3f(-loc_or_dir);
        } else {
          dir_tree_transforms_[transform_idx] =
              scene::look_at(loc_or_dir, Eigen::Vector3f(0, 1, 0));
        }
      };

  for (const auto &light : lights) {
    light.visit([&](auto &&light_data) {
      using T = std::decay_t<decltype(light_data)>;
      if constexpr (std::is_same<T, scene::DirectionalLight>::value) {
        add_transform(false, light_data.direction);
      } else {
        add_transform(true, light_data.position);
      }
    });
  }

  for (unsigned collar = 0; collar < partition.regions().size(); collar++) {
    unsigned start = partition.regions()[collar].start_index;
    unsigned end = partition.regions()[collar].end_index;
    for (unsigned region = 0; region < end - start; region++) {
      add_transform(false, partition.get_center_vec(collar, region));
    }
  }

  const auto end_setup_traversal_grid = current_time();
  double setup_traversal_grid_duration =
      to_secs(start_setup_traversal_grid, end_setup_traversal_grid);

  if (show_times_) {
    dbg(setup_traversal_grid_duration);
  }

  const auto start_project_traversal_grid = current_time();

  bounds_.resize(shapes.size());

  std::transform(shapes.begin(), shapes.end(), bounds_.begin(),
                 [](const scene::ShapeData &shape) {
                   return get_bounding(shape.get_transform());
                 });

  aabbs_.resize(bounds_.size() * dir_tree_transforms_.size());

  compute_aabbs<execution_model>(
      to_const_span(dir_tree_transforms_), dir_tree_transforms_.size(),
      to_span(aabbs_), aabbs_.size(), to_const_span(bounds_), bounds_.size());

  sorted_by_x_edges_.resize(aabbs_.size() * 2);
  sorted_by_y_edges_.resize(aabbs_.size() * 2);
  z_centers_.resize(aabbs_.size());
  sorted_by_z_center_.resize(aabbs_.size());

  unsigned num_shapes = shapes.size();

  auto sorted_by_x_edges_span = to_span(sorted_by_x_edges_);
  auto sorted_by_y_edges_span = to_span(sorted_by_y_edges_);

  /* thrust::for_each( */
  /*     thrust::make_counting_iterator(0u), */
  /*     thrust::make_counting_iterator(unsigned(dir_tree_transforms_.size())), */
  /*     [num_shapes, ] __host__ __device__(unsigned transform_idx) { */
  /*     thrust::transform(aabbs_ */
  /*     }); */

  const auto end_project_traversal_grid = current_time();
  double project_traversal_grid_duration =
      to_secs(start_project_traversal_grid, end_project_traversal_grid);

  const auto count_prefix_sum = chr::high_resolution_clock::now();

  action_starts_.resize(start_count_index);
  thrust::fill_n(to_thrust_iter(action_starts_), action_starts_.size(), 0);

  if constexpr (execution_model == ExecutionModel::GPU) {
    update_counts<shape_is_outer>(grid_span, to_const_span(shape_grids_),
                                  to_span(action_starts_), shapes.size(),
                                  block_dim_grid, block_dim_shape);
  } else {
    update_counts_cpu(grid_span, to_const_span(shape_grids_),
                      to_span(action_starts_), shapes.size());
  }

  unsigned last_size = action_starts_[action_starts_.size() - 1];

  thrust::exclusive_scan(to_thrust_iter(action_starts_),
                         to_thrust_iter(action_starts_) + action_starts_.size(),
                         to_thrust_iter(action_starts_));

  unsigned total_num_actions =
      action_starts_[action_starts_.size() - 1] + last_size;

  action_ends_.resize(action_starts_.size());

  thrust::copy(to_thrust_iter(action_starts_),
               to_thrust_iter(action_starts_) + action_starts_.size(),
               to_thrust_iter(action_ends_));

  min_sorted_actions_.resize(total_num_actions);

  if constexpr (execution_model == ExecutionModel::GPU) {
    add_actions<shape_is_outer>(grid_span, to_span(shape_grids_),
                                to_span(action_ends_),
                                to_span(min_sorted_actions_), shapes.size(),
                                block_dim_grid, block_dim_shape);
  } else {
    add_actions_cpu(grid_span, to_span(shape_grids_), to_span(action_ends_),
                    to_span(min_sorted_actions_), shapes.size());
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - count_prefix_sum)
            .count());
  }

  const auto get_traversal_sort_actions = chr::high_resolution_clock::now();

  traversals_.resize(action_starts_.size());

  auto transform_to_traversal = [&](const auto execution_type) {
    thrust::transform(execution_type, to_thrust_iter(action_starts_),
                      to_thrust_iter(action_starts_) + action_starts_.size(),
                      to_thrust_iter(action_ends_), to_thrust_iter(traversals_),
                      [] __host__ __device__(int start, int end) {
                        return Traversal(start, end);
                      });
  };

  if constexpr (execution_model == ExecutionModel::GPU) {
    transform_to_traversal(thrust::device);
  } else {
    transform_to_traversal(thrust::host);
  }

#if 0
  segments_.resize(min_sorted_actions_.size());

  auto segments_span = to_span(segments_);
#endif

  auto fill_segments_sort = [&](const auto execution_type) {
#if 0
    thrust::for_each(
        execution_type,
        thrust::make_zip_iterator(
            thrust::make_tuple(to_thrust_iter(traversals_),
                               thrust::counting_iterator<unsigned>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(
            to_thrust_iter(traversals_) + traversals_.size(),

            thrust::counting_iterator<unsigned>(traversals_.size()))),

        [segments =
             segments_span] __host__ __device__(const auto &traversal_index) {
          auto traversal = thrust::get<0>(traversal_index);
          auto index = thrust::get<1>(traversal_index);
          for (unsigned i = traversal.start; i < traversal.end; i++) {
            segments[i] = index;
          }
        });
#endif
    max_sorted_actions_.resize(min_sorted_actions_.size());

    thrust::copy(to_thrust_iter(min_sorted_actions_),
                 to_thrust_iter(min_sorted_actions_) +
                     min_sorted_actions_.size(),
                 to_thrust_iter(max_sorted_actions_));

    auto sort_actions = [&](DataType<Action> &actions, const auto &getter) {
      thrust::sort(execution_type, to_thrust_iter(actions),
                   to_thrust_iter(actions) + actions.size(),
                   [getter = getter] __host__ __device__(const Action &first,
                                                         const Action &second) {
                     if (first.sort_index == second.sort_index) {
                       return getter(first) < getter(second);
                     } else {
                       return first.sort_index < second.sort_index;
                     }
                   });
    };

    sort_actions(min_sorted_actions_,
                 [] __host__ __device__(const Action &action) {
                   return action.min_dist;
                 });

    sort_actions(max_sorted_actions_,
                 [] __host__ __device__(const Action &action) {
                   return action.max_dist;
                 });
  };

  if constexpr (execution_model == ExecutionModel::GPU) {
    fill_segments_sort(thrust::device);
  } else {
    fill_segments_sort(thrust::host);
  }

  dbg(action_starts_.size());
  dbg(min_sorted_actions_.size());

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - get_traversal_sort_actions)
            .count());
  }

  const auto copy_into_traversal_grid = chr::high_resolution_clock::now();

  auto transform_to_disable = [&](auto ptr_type) {
    thrust::transform(to_thrust_iter(traversals_),
                      to_thrust_iter(traversals_) +
                          block_data_.num_blocks_x * block_data_.num_blocks_y,
                      ptr_type,
                      [] __host__ __device__(const Traversal &traversal) {
                        return traversal.end - traversal.start == 0;
                      });
  };

  if constexpr (execution_model == ExecutionModel::GPU) {
    transform_to_disable(thrust::device_ptr<uint8_t>(to_ptr(group_disables_)));
  } else {
    transform_to_disable(to_ptr(group_disables_));
  }

  for (unsigned i = 1; i < traversal_grids_.size(); i++) {
    auto &traversal_grid = traversal_grids_[i];

    traversal_data_[i - 1] = traversal_grid.traversalData();
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - copy_into_traversal_grid)
            .count());
  }

  if (traversal_data_.size() != traversal_grid_index - 1) {
    dbg("INVALID SIZE");
    abort();
  }

  return TraversalGridsRef(
      to_const_span(min_sorted_actions_), to_const_span(max_sorted_actions_),
      to_const_span(traversal_data_), to_const_span(traversals_),
      traversal_data_starts, min_bound, max_bound, inverse_multipliers,
      min_side_bounds, max_side_bounds, min_side_diffs, max_side_diffs);
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace detail
} // namespace ray
