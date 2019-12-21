#include "ray/projection_impl.h"
#include "ray/render_impl.h"
#include "ray/render_impl_utils.h"

#include <boost/iterator/counting_iterator.hpp>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <chrono>

namespace ray {
using namespace detail;

struct modulo_n : public thrust::unary_function<unsigned, unsigned> {
  unsigned n;

  modulo_n(unsigned n) : n(n) {}

  __host__ __device__ unsigned operator()(unsigned x) { return x % n; }
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

  traversal_data_cpu_.resize(lights.size() + total_size);
  traversal_grids_.resize(1 + lights.size() + total_size);
  shape_grids_.resize(traversal_grids_.size() * shapes.size());

  unsigned traversal_grid_index = 0;
  unsigned start_shape_grids = 0;
  unsigned start_hash_index = 0;

  traversal_grids_[traversal_grid_index] = TraversalGrid(
      TriangleProjector(world_to_film), Eigen::Array2f(-1, -1),
      Eigen::Array2f(1, 1), block_data_.num_blocks_x, block_data_.num_blocks_y,
      start_shape_grids, start_hash_index, false, true);

  start_shape_grids += shapes.size();
  start_hash_index += block_data_.num_blocks_x * block_data_.num_blocks_y;

  traversals_cpu_.clear();
  actions_cpu_.clear();

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

  scene::ShapeData bounding_cube(bounding_transform, scene::Material(),
                                 scene::Shape::Cube);

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
          std::array<ProjectedTriangle, max_proj_tris> triangles;

          unsigned num_triangles = project_shape(bounding_cube, projector, triangles);

          for (unsigned triangle_idx = 0; triangle_idx < num_triangles;
               triangle_idx++) {
            const auto &triangle = triangles[triangle_idx];
            for (const auto &point : triangle.points()) {
              projected_min = projected_min.cwiseMin(point);
              projected_max = projected_max.cwiseMax(point);
            }
          }
        }

        traversal_grids_[traversal_grid_index] = TraversalGrid(
            projector, projected_min, projected_max, num_divisions_x,
            num_divisions_y, start_shape_grids, start_hash_index);

        start_shape_grids += shapes.size();
        start_hash_index += num_divisions_x * num_divisions_y;
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

  Span<TraversalGrid, false> grids_span(traversal_grids_.data(),
                                        traversal_grids_.size());

  update_shapes(grids_span, to_span(shape_grids_), shapes);

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - project_traversal_grid)
            .count());
  }

  const auto hashing_traversal_grid = chr::high_resolution_clock::now();

  shape_hashes_.resize(shapes.size());

  get_shape_hashes(
      Span<unsigned, false>(shape_hashes_.data(), shape_hashes_.size()));

  hashes_.resize(start_hash_index);
  std::fill_n(hashes_.begin(), hashes_.size(), 0);

  hash_shapes(grids_span, to_const_span(shape_grids_),
              to_const_span(shape_hashes_), to_span(hashes_), shapes.size());

  copied_hashes_.resize(hashes_.size());
  std::copy(hashes_.begin(), hashes_.end(), copied_hashes_.begin());

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - hashing_traversal_grid)
            .count());
  }


  const auto sort_unique_traversal_grid = chr::high_resolution_clock::now();

  indexes_.resize(hashes_.size());

  thrust::sequence(indexes_.data(), indexes_.data() + indexes_.size());

  thrust::sort_by_key(thrust::host, hashes_.data(),
                      hashes_.data() + hashes_.size(), indexes_.data());

  auto iter_out =
      thrust::unique_by_key(thrust::host, hashes_.data(),
                            hashes_.data() + hashes_.size(), indexes_.data());

  unsigned num_unique_hashes =
      std::distance(hashes_.data(), thrust::get<0>(iter_out));

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - sort_unique_traversal_grid)
            .count());
  }
  
  const auto construct_hash_map = chr::high_resolution_clock::now();

  unsigned hash_map_multiplier = 4;
  unsigned base_size = hash_map_multiplier * num_unique_hashes;

  hash_map_base_.resize(base_size);

  hash_hash_map_indexes_.resize(num_unique_hashes);

  thrust::transform(thrust::host, hashes_.data(),
                    hashes_.data() + num_unique_hashes,
                    hash_hash_map_indexes_.data(), modulo_n(base_size));


  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - construct_hash_map)
            .count());
  }

  const auto copy_into_traversal_grid = chr::high_resolution_clock::now();

  std::vector<unsigned> temp;

  Span<const ShapePossibles> shape_grids_span(shape_grids_.data(),
                                              shape_grids_.size());

  for (unsigned i = 0; i < traversal_grids_.size(); i++) {
    auto &traversal_grid = traversal_grids_[i];

    if (i > 0) {
      traversal_data_cpu_[i - 1] =
          traversal_grid.traversalData(traversals_cpu_.size());
    }

    traversal_grid.copy_into(shape_grids_span, shapes.size(), traversals_cpu_,
                             actions_cpu_, temp);

    if (i == 0) {
      std::transform(traversals_cpu_.begin(), traversals_cpu_.end(),
                     group_disables_.begin(), [&](const Traversal &traversal) {
                       return traversal.size == 0;
                     });
    }
  }

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - copy_into_traversal_grid)
            .count());
  }

  if (traversal_data_cpu_.size() != traversal_grid_index - 1) {
    dbg("INVALID SIZE");
    abort();
  }

  auto copy = [](auto start, auto end, auto start_copy) {
    if constexpr (execution_model == ExecutionModel::GPU) {
      thrust::copy(start, end, start_copy);
    } else {
      std::copy(start, end, start_copy);
    }
  };

  const auto copy_gpu_traversal_grid = chr::high_resolution_clock::now();

  traversals_.resize(traversals_cpu_.size());
  copy(traversals_cpu_.data(), traversals_cpu_.data() + traversals_cpu_.size(),
       traversals_.begin());

  actions_.resize(actions_cpu_.size());
  copy(actions_cpu_.data(), actions_cpu_.data() + actions_cpu_.size(),
       actions_.begin());

  traversal_data_.resize(traversal_data_cpu_.size());
  std::copy(traversal_data_cpu_.begin(), traversal_data_cpu_.end(),
            traversal_data_.begin());

  if (show_times) {
    dbg(chr::duration_cast<chr::duration<double>>(
            chr::high_resolution_clock::now() - copy_gpu_traversal_grid)
            .count());
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
