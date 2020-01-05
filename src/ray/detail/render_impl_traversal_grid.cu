#include "ray/projection_impl.h"
#include "ray/render_impl.h"
#include "ray/render_impl_utils.h"
#include "ray/sort_actions.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <chrono>
#include <dbg.h>

namespace ray {
using namespace detail;

#if 0
struct CreateTraversal : public thrust::binary_function<int, int, Traversal> {
  HOST_DEVICE Traversal operator()(int start, int end) {
    return Traversal(start, end);
  }
};
#endif

template <ExecutionModel execution_model>
TraversalGridsRef RendererImpl<execution_model>::traversal_grids(
    bool show_times, const Eigen::Projective3f &world_to_film,
    Span<const scene::ShapeData, false> shapes,
    Span<const scene::Light, false> lights) {
  namespace chr = std::chrono;
  const auto setup_traversal_grid = chr::high_resolution_clock::now();

  const unsigned total_size = 0;

  unsigned num_division_light_x = 32;
  unsigned num_division_light_y = 32;

  traversal_data_.resize(lights.size() + total_size);
  traversal_grids_.resize(1 + lights.size() + total_size);
  shape_grids_.resize(traversal_grids_.size() * shapes.size());

  unsigned traversal_grid_index = 0;
  unsigned start_shape_grids = 0;
  unsigned start_count_index = 0;

  traversal_grids_[traversal_grid_index] = TraversalGrid(
      Projector(world_to_film.matrix()), Eigen::Array2f(-1, -1),
      Eigen::Array2f(1, 1), block_data_.num_blocks_x, block_data_.num_blocks_y,
      start_shape_grids, start_count_index, 1.0f, 1.0f, false, true);

  start_shape_grids += shapes.size();
  start_count_index += block_data_.num_blocks_x * block_data_.num_blocks_y;

  traversal_grid_index++;

  const auto &max_bound = scene_->getMaxBound();
  const auto &min_bound = scene_->getMinBound();

  const auto center = ((min_bound + max_bound) / 2).eval();
  const auto dims = (max_bound - min_bound).eval();

  auto get_plane = [&](bool is_loc, const Eigen::Vector3f &loc_or_dir) {
    const auto dir = is_loc ? (center - loc_or_dir).eval() : loc_or_dir;
    const auto normalized_directions = dir.array() / dims.array();
    unsigned axis;
    float max_axis_v = std::numeric_limits<float>::lowest();
    for (unsigned test_axis = 0; test_axis < 3; test_axis++) {
      float abs_v = std::abs(normalized_directions[test_axis]);
      if (abs_v > max_axis_v) {
        axis = test_axis;
        max_axis_v = abs_v;
      }
    }

    float projection_value = max_bound[axis];

    return Plane(projection_value, axis);
  };

  Eigen::Affine3f bounding_transform = Eigen::Translation3f(center) *
                                       Eigen::Scaling(dims) *
                                       Eigen::Affine3f::Identity();

  BoundingPoints bounding_cube = get_bounding(bounding_transform);

  auto add_projection =
      [&](bool is_loc, const Eigen::Vector3f &loc_or_dir,
          unsigned num_divisions_x, unsigned num_divisions_y, Plane plane,
          const thrust::optional<std::tuple<Eigen::Array2f, Eigen::Array2f>>
              &projection_surface_min_max,
          float min_dist_multiplier, float max_dist_multiplier) {
        if (is_loc && (loc_or_dir.array() <= max_bound.array()).all() &&
            (loc_or_dir.array() >= min_bound.array()).all()) {
          dbg("INTERNAL POINT PROJECTIONS NOT SUPPORTED");
          abort();
        }

        Eigen::Array2f projected_min =
            Eigen::Array2f(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max());
        Eigen::Array2f projected_max =
            Eigen::Array2f(std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest());

        Projector projector(DirectionPlane(loc_or_dir, is_loc, plane));

        if (projection_surface_min_max.has_value()) {
          const auto &[p_min, p_max] = *projection_surface_min_max;
          projected_min = p_min;
          projected_max = p_max;
        } else {
          for (const auto &point : bounding_cube) {
            const auto [projected, _] = projector.project_point(point);
            projected_min = projected_min.cwiseMin(projected);
            projected_max = projected_max.cwiseMax(projected);
          }
        }

        traversal_grids_[traversal_grid_index] = TraversalGrid(
            projector, projected_min, projected_max, num_divisions_x,
            num_divisions_y, start_shape_grids, start_count_index,
            min_dist_multiplier, max_dist_multiplier);

        start_shape_grids += shapes.size();
        start_count_index += num_divisions_x * num_divisions_y;
        traversal_grid_index++;
      };

  for (const auto &light : lights) {
    auto add_light_projection = [&](bool is_loc, Eigen::Vector3f loc_or_dir) {
      auto plane = get_plane(is_loc, loc_or_dir);
      add_projection(is_loc, loc_or_dir, num_division_light_x,
                     num_division_light_y, plane, thrust::nullopt, 1.0f, 1.0f);
    };

    light.visit([&](auto &&light_data) {
      using T = std::decay_t<decltype(light_data)>;
      if constexpr (std::is_same<T, scene::DirectionalLight>::value) {
        add_light_projection(false, light_data.direction);
      } else {
        add_light_projection(true, light_data.position);
      }
    });
  }

  std::array<unsigned, 3> traversal_data_starts;

#if 0
  Eigen::Array3f multipliers = dims.array() / num_divisions.cast<float>();
#else
  Eigen::Array3f multipliers;
#endif
  std::array<Eigen::Array2f, 3> min_side_bounds;
  std::array<Eigen::Array2f, 3> max_side_bounds;
  std::array<Eigen::Array2<int>, 3> min_side_diffs;
  std::array<Eigen::Array2<int>, 3> max_side_diffs;

  Eigen::Array3f inverse_multipliers = 1.0f / multipliers;

#if 0

  auto num_divisions_p_1 = (num_divisions + 1).eval();

  for (uint8_t axis : {0, 1, 2}) {
    traversal_data_starts[axis] = traversal_grid_index - 1;
    uint8_t first_axis = (axis + 1) % 3;
    uint8_t second_axis = (axis + 2) % 3;
    float first_multip = multipliers[first_axis];
    float second_multip = multipliers[second_axis];
    int first_divisions = num_divisions_p_1[first_axis];
    int second_divisions = num_divisions_p_1[second_axis];

    Plane plane(max_bound[axis], axis);

    min_side_bounds[axis] =
        (plane.get_not_axis(min_bound) -
         plane.get_not_axis(multipliers) *
             plane.get_not_axis(num_divisions_p_1).cast<float>()) *
        plane.get_not_axis(inverse_multipliers);
    max_side_bounds[axis] =
        (plane.get_not_axis(max_bound) +
         plane.get_not_axis(multipliers) *
             plane.get_not_axis(num_divisions_p_1).cast<float>()) *
        plane.get_not_axis(inverse_multipliers);
    min_side_diffs[axis] = -plane.get_not_axis(num_divisions_p_1).cast<int>();
    max_side_diffs[axis] = plane.get_not_axis(num_divisions_p_1).cast<int>();

    auto min_other_bounds = plane.get_not_axis(min_bound);
    auto max_other_bounds = plane.get_not_axis(max_bound);

    for (int translation_second = -second_divisions;
         translation_second <= second_divisions; translation_second++) {
      for (int translation_first = -first_divisions;
           translation_first <= first_divisions; translation_first++) {
        Eigen::Vector3f dir;
        dir[axis] = max_bound[axis] - min_bound[axis];
        dir[first_axis] = translation_first * first_multip;
        dir[second_axis] = translation_second * second_multip;

        float general_dist = dir.norm();

        auto reduce_mag = [](int v) {
          if (v > 0) {
            return v - 1;
          } else if (v < 0) {
            return v + 1;
          }

          return 0;
        };

        auto increase_mag = [](int v) {
          if (v > 0) {
            return v + 1;
          } else if (v < 0) {
            return v - 1;
          }

          return 1;
        };

        Eigen::Vector3f dir_shortest;
        dir_shortest[axis] = max_bound[axis] - min_bound[axis];
        dir_shortest[first_axis] = reduce_mag(translation_first) * first_multip;
        dir_shortest[second_axis] =
            reduce_mag(translation_second) * second_multip;

        float shortest_dist = dir_shortest.norm();

        Eigen::Vector3f dir_longest;
        dir_longest[axis] = max_bound[axis] - min_bound[axis];
        dir_longest[first_axis] =
            increase_mag(translation_first) * first_multip;
        dir_longest[second_axis] =
            increase_mag(translation_second) * second_multip;

        float longest_dist = dir_longest.norm();

        float min_dist_multiplier = shortest_dist / general_dist;
        float max_dist_multiplier = longest_dist / general_dist;

        auto dir_other_axis = plane.get_not_axis(dir);

        Eigen::Array2f projected_min =
            min_other_bounds.cwiseMin(dir_other_axis + min_other_bounds);
        Eigen::Array2f projected_max =
            max_other_bounds.cwiseMax(dir_other_axis + max_other_bounds);

        add_projection(
            false, dir,
            num_divisions[first_axis] + unsigned(std::abs(translation_first)),
            num_divisions[second_axis] + unsigned(std::abs(translation_second)),
            plane, std::make_tuple(projected_min, projected_max),
            min_dist_multiplier, max_dist_multiplier);
      }
    }
  }
#endif

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - setup_traversal_grid)
            .count());
  }

  const auto project_traversal_grid = chr::high_resolution_clock::now();

  shape_bounds_.resize(shapes.size());

  std::transform(shapes.begin(), shapes.end(), shape_bounds_.begin(),
                 [](const scene::ShapeData &shape) {
                   return get_bounding(shape.get_transform());
                 });

  Span<TraversalGrid, false> grid_span(traversal_grids_.data(),
                                       traversal_grids_.size());

  constexpr bool shape_is_outer = false;
  unsigned block_dim_grid = 2;
  unsigned block_dim_shape = 64;

  if constexpr (execution_model == ExecutionModel::GPU) {
    update_shapes<shape_is_outer>(grid_span, to_span(shape_grids_),
                                  to_const_span(shape_bounds_), shapes.size(),
                                  block_dim_grid, block_dim_shape);
  } else {
    update_shapes_cpu(grid_span, to_span(shape_grids_),
                      to_const_span(shape_bounds_), shapes.size());
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - project_traversal_grid)
            .count());
  }

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
} // namespace ray
