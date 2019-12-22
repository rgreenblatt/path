#include "ray/block_data.h"
#include "ray/traversal_grid.h"
#include "ray/traversal_grid_impl.h"

#include <tuple>

namespace ray {
namespace detail {
template <bool shape_is_outer>
__device__ std::tuple<unsigned, unsigned> get_grid_shape_idx() {
  unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
  if constexpr (shape_is_outer) {
    return std::make_tuple(x, y);
  } else {
    return std::make_tuple(y, x);
  }
}

template <bool shape_is_outer>
__global__ void update_shapes_impl(Span<TraversalGrid, false> grids,
                                   Span<ShapePossibles> shape_grids,
                                   Span<const BoundingPoints> shape_bounds,
                                   unsigned num_shapes) {
  auto [grid_idx, shape_idx] = get_grid_shape_idx<shape_is_outer>();

  if (grid_idx >= grids.size() || shape_idx >= num_shapes) {
    return;
  }

  grids[grid_idx].updateShape(shape_bounds, shape_grids, shape_idx);
}

template <bool shape_is_outer>
__global__ void update_counts_impl(Span<TraversalGrid, false> grids,
                                   Span<const ShapePossibles> shape_grids,
                                   Span<int> counts, unsigned num_shapes) {
  auto [grid_idx, shape_idx] = get_grid_shape_idx<shape_is_outer>();

  if (grid_idx >= grids.size() || shape_idx >= num_shapes) {
    return;
  }

  grids[grid_idx].getCount(shape_grids, shape_idx, counts);
}

template <bool shape_is_outer>
__global__ void add_actions_impl(Span<TraversalGrid, false> grids,
                                 Span<const ShapePossibles> shape_grids,
                                 Span<int> action_indexes, Span<Action> actions,
                                 unsigned num_shapes) {
  auto [grid_idx, shape_idx] = get_grid_shape_idx<shape_is_outer>();

  if (grid_idx >= grids.size() || shape_idx >= num_shapes) {
    return;
  }

  grids[grid_idx].addActions(shape_grids, shape_idx, action_indexes, actions);
}

template <bool shape_is_outer>
std::tuple<dim3, dim3> get_grid_block(unsigned num_grids, unsigned num_shapes,
                                      unsigned block_dim_grid,
                                      unsigned block_dim_shape) {
  unsigned num_grid_blocks = num_blocks(num_grids, block_dim_grid);
  unsigned num_shape_blocks = num_blocks(num_shapes, block_dim_shape);

  dim3 grid;
  dim3 block;

  if constexpr (shape_is_outer) {
    grid = dim3(num_grid_blocks, num_shape_blocks);
    block = dim3(block_dim_grid, block_dim_shape);
  } else {
    grid = dim3(num_shape_blocks, num_grid_blocks);
    block = dim3(block_dim_shape, block_dim_grid);
  }

  return std::tuple{grid, block};
}

template <bool shape_is_outer>
void update_shapes(Span<TraversalGrid, false> grids,
                   Span<ShapePossibles> shape_grids,
                   Span<const BoundingPoints> shape_bounds, unsigned num_shapes,
                   unsigned block_dim_grid, unsigned block_dim_shape) {
  auto [grid, block] = get_grid_block<shape_is_outer>(
      grids.size(), num_shapes, block_dim_grid, block_dim_shape);

  update_shapes_impl<shape_is_outer>
      <<<grid, block>>>(grids, shape_grids, shape_bounds, num_shapes);
  
  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template <bool shape_is_outer>
void update_counts(Span<TraversalGrid, false> grids,
                   Span<const ShapePossibles> shape_grids, Span<int> counts,
                   unsigned num_shapes, unsigned block_dim_grid,
                   unsigned block_dim_shape) {
  auto [grid, block] = get_grid_block<shape_is_outer>(
      grids.size(), num_shapes, block_dim_grid, block_dim_shape);

  update_counts_impl<shape_is_outer>
      <<<grid, block>>>(grids, shape_grids, counts, num_shapes);

  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template <bool shape_is_outer>
void add_actions(Span<TraversalGrid, false> grids,
                 Span<const ShapePossibles> shape_grids,
                 Span<int> action_indexes, Span<Action> actions,
                 unsigned num_shapes, unsigned block_dim_grid,
                 unsigned block_dim_shape) {
  auto [grid, block] = get_grid_block<shape_is_outer>(
      grids.size(), num_shapes, block_dim_grid, block_dim_shape);

  add_actions_impl<shape_is_outer><<<grid, block>>>(
      grids, shape_grids, action_indexes, actions, num_shapes);
  
  CUDA_ERROR_CHK(cudaDeviceSynchronize());
}

template void update_shapes<true>(Span<TraversalGrid, false> grids,
                                  Span<ShapePossibles> shape_grids,
                                  Span<const BoundingPoints> shape_bounds,
                                  unsigned num_shapes, unsigned block_dim_grid,
                                  unsigned block_dim_shape);
template void update_shapes<false>(Span<TraversalGrid, false> grids,
                                   Span<ShapePossibles> shape_grids,
                                   Span<const BoundingPoints> shape_bounds,
                                   unsigned num_shapes, unsigned block_dim_grid,
                                   unsigned block_dim_shape);
template void update_counts<true>(Span<TraversalGrid, false> grids,
                                  Span<const ShapePossibles> shape_grids,
                                  Span<int> counts, unsigned num_shapes,
                                  unsigned block_dim_grid,
                                  unsigned block_dim_shape);
template void update_counts<false>(Span<TraversalGrid, false> grids,
                                   Span<const ShapePossibles> shape_grids,
                                   Span<int> counts, unsigned num_shapes,
                                   unsigned block_dim_grid,
                                   unsigned block_dim_shape);
template void add_actions<false>(Span<TraversalGrid, false> grids,
                                 Span<const ShapePossibles> shape_grids,
                                 Span<int> action_indexes, Span<Action> actions,
                                 unsigned num_shapes, unsigned block_dim_grid,
                                 unsigned block_dim_shape);
template void add_actions<true>(Span<TraversalGrid, false> grids,
                                Span<const ShapePossibles> shape_grids,
                                Span<int> action_indexes, Span<Action> actions,
                                unsigned num_shapes, unsigned block_dim_grid,
                                unsigned block_dim_shape);
} // namespace detail
} // namespace ray
