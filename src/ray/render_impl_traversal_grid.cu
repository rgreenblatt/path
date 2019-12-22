#include "ray/projection_impl.h"
#include "ray/render_impl.h"
#include "ray/render_impl_utils.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <chrono>
#include <dbg.h>

namespace ray {
using namespace detail;

struct CreateTraversal : public thrust::binary_function<int, int, Traversal> {
  HOST_DEVICE Traversal operator()(int start, int end) {
    return Traversal(start, end);
  }
};

template <ExecutionModel execution_model>
TraversalGridsRef RendererImpl<execution_model>::traversal_grids(
    bool show_times, const Eigen::Projective3f &world_to_film,
    Span<const scene::ShapeData, false> shapes,
    Span<const scene::Light, false> lights) {
  namespace chr = std::chrono;
  const auto setup_traversal_grid = chr::high_resolution_clock::now();

  const Eigen::Array3<unsigned> num_divisions(16, 16, 16);

  const Eigen::Array3<unsigned> shifted_1_num_divisions(
      num_divisions[1], num_divisions[2], num_divisions[0]);
  const Eigen::Array3<unsigned> shifted_2_num_divisions(
      num_divisions[2], num_divisions[0], num_divisions[1]);

  const auto shifted_1_num_translations =
      (2 * shifted_1_num_divisions + 1).eval();
  const auto shifted_2_num_translations =
      (2 * shifted_2_num_divisions + 1).eval();

  const auto total_translations =
      (shifted_1_num_translations * shifted_2_num_translations).eval();

  const unsigned total_size = total_translations.sum();

  unsigned num_division_light_x = 32;
  unsigned num_division_light_y = 32;

  traversal_data_.resize(lights.size() + total_size);
  traversal_grids_.resize(1 + lights.size() + total_size);
  shape_grids_.resize(traversal_grids_.size() * shapes.size());

  unsigned traversal_grid_index = 0;
  unsigned start_shape_grids = 0;
  unsigned start_count_index = 0;

  traversal_grids_[traversal_grid_index] = TraversalGrid(
      TriangleProjector(world_to_film.matrix()), Eigen::Array2f(-1, -1),
      Eigen::Array2f(1, 1), block_data_.num_blocks_x, block_data_.num_blocks_y,
      start_shape_grids, start_count_index, false, true);

  start_shape_grids += shapes.size();
  start_count_index += block_data_.num_blocks_x * block_data_.num_blocks_y;

  traversal_grid_index++;

  const auto &max_bound = scene_->getMaxBound();
  const auto &min_bound = scene_->getMinBound();

  const auto center = ((min_bound + max_bound) / 2).eval();
  const auto dims = (max_bound - min_bound).eval();

  auto get_axis = [&](bool is_loc, const Eigen::Vector3f &loc_or_dir) {
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

    bool is_min = normalized_directions[axis] < 0;

    float projection_value = is_min ? min_bound[axis] : max_bound[axis];

    return std::make_tuple(axis, projection_value);
  };

  Eigen::Affine3f bounding_transform = Eigen::Translation3f(center) *
                                       Eigen::Scaling(dims) *
                                       Eigen::Affine3f::Identity();

  BoundingPoints bounding_cube = get_bounding(bounding_transform);

  auto add_projection =
      [&](bool is_loc, const Eigen::Vector3f &loc_or_dir,
          unsigned num_divisions_x, unsigned num_divisions_y, uint8_t axis,
          float projection_value,
          const thrust::optional<std::tuple<Eigen::Array2f, Eigen::Array2f>>
              &projection_surface_min_max) {
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

        TriangleProjector projector(
            DirectionPlane(loc_or_dir, is_loc, projection_value, axis));

        if (projection_surface_min_max.has_value()) {
          const auto &[p_min, p_max] = *projection_surface_min_max;
          projected_min = p_min;
          projected_max = p_max;
        } else {
          for (const auto &point : bounding_cube) {
            const auto projected = project_point(point, projector);
            projected_min = projected_min.cwiseMin(projected);
            projected_max = projected_max.cwiseMax(projected);
          }
        }

        traversal_grids_[traversal_grid_index] = TraversalGrid(
            projector, projected_min, projected_max, num_divisions_x,
            num_divisions_y, start_shape_grids, start_count_index);

        start_shape_grids += shapes.size();
        start_count_index += num_divisions_x * num_divisions_y;
        traversal_grid_index++;
      };

  for (const auto &light : lights) {
    light.visit([&](auto &&light_data) {
      using T = std::decay_t<decltype(light_data)>;
      if constexpr (std::is_same<T, scene::DirectionalLight>::value) {
        auto [axis, value] = get_axis(false, light_data.direction);
        add_projection(false, light_data.direction, num_division_light_x,
                       num_division_light_y, axis, value, thrust::nullopt);
      } else {
        auto [axis, value] = get_axis(true, light_data.position);
        add_projection(true, light_data.position, num_division_light_x,
                       num_division_light_y, axis, value, thrust::nullopt);
      }
    });
  }

  std::array<unsigned, 3> traversal_data_starts;

  Eigen::Array3f multipliers = dims.array() / num_divisions.cast<float>();
  std::array<Eigen::Array2f, 3> min_side_bounds;
  std::array<Eigen::Array2f, 3> max_side_bounds;
  std::array<Eigen::Array2<int>, 3> min_side_diffs;
  std::array<Eigen::Array2<int>, 3> max_side_diffs;

  Eigen::Array3f inverse_multipliers = 1.0f / multipliers;

  for (uint8_t axis : {0, 1, 2}) {
    traversal_data_starts[axis] = traversal_grid_index - 1;
    uint8_t first_axis = (axis + 1) % 3;
    uint8_t second_axis = (axis + 2) % 3;
    float first_multip = multipliers[first_axis];
    float second_multip = multipliers[second_axis];
    int first_divisions = num_divisions[first_axis];
    int second_divisions = num_divisions[second_axis];

    min_side_bounds[axis] =
        (get_not_axis(min_bound, axis) -
         get_not_axis(multipliers, axis) *
             get_not_axis(num_divisions, axis).cast<float>()) *
        get_not_axis(inverse_multipliers, axis);
    max_side_bounds[axis] =
        (get_not_axis(max_bound, axis) +
         get_not_axis(multipliers, axis) *
             get_not_axis(num_divisions, axis).cast<float>()) *
        get_not_axis(inverse_multipliers, axis);
    min_side_diffs[axis] = -get_not_axis(num_divisions, axis).cast<int>();
    max_side_diffs[axis] = get_not_axis(num_divisions, axis).cast<int>();

    auto min_other_bounds = get_not_axis(min_bound, axis);
    auto max_other_bounds = get_not_axis(max_bound, axis);

    for (int translation_second = -second_divisions;
         translation_second <= second_divisions; translation_second++) {
      for (int translation_first = -first_divisions;
           translation_first <= first_divisions; translation_first++) {
        Eigen::Vector3f dir;
        dir[axis] = max_bound[axis] - min_bound[axis];
        dir[first_axis] = translation_first * first_multip;
        dir[second_axis] = translation_second * second_multip;

        auto dir_other_axis = get_not_axis(dir, axis);

        Eigen::Array2f projected_min =
            min_other_bounds.cwiseMin(dir_other_axis + min_other_bounds);
        Eigen::Array2f projected_max =
            max_other_bounds.cwiseMax(dir_other_axis + max_other_bounds);

        add_projection(
            false, dir,
            num_divisions[first_axis] + unsigned(std::abs(translation_first)),
            num_divisions[second_axis] + unsigned(std::abs(translation_second)),
            axis, max_bound[axis],
            std::make_tuple(projected_min, projected_max));
      }
    }
  }

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
                  to_span(action_starts_), shapes.size(), block_dim_grid,
                  block_dim_shape);
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

  actions_.resize(total_num_actions);

  if constexpr (execution_model == ExecutionModel::GPU) {
    add_actions<shape_is_outer>(grid_span, to_const_span(shape_grids_),
                                to_span(action_ends_), to_span(actions_),
                                shapes.size(), block_dim_grid, block_dim_shape);
  } else {
    add_actions_cpu(grid_span, to_const_span(shape_grids_),
                    to_span(action_ends_), to_span(actions_), shapes.size());
  }

  traversals_.resize(action_starts_.size());

  auto transform_to_traversal = [&](const auto &execution_type) {
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

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - count_prefix_sum)
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
      to_const_span(actions_), to_const_span(traversal_data_),
      to_const_span(traversals_), traversal_data_starts, min_bound, max_bound,
      inverse_multipliers, min_side_bounds, max_side_bounds, min_side_diffs,
      max_side_diffs);
}

template class RendererImpl<ExecutionModel::CPU>;
template class RendererImpl<ExecutionModel::GPU>;
} // namespace ray
