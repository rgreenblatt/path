#pragma once

#include "ray/traversal_grid.h"
#include "ray/best_intersection.h"
#include "ray/block_data.h"
#include "ray/kdtree_nodes_ref.h"
#include "ray/cone.h"
#include "ray/cube.h"
#include "ray/cuda_ray_utils.cuh"
#include "ray/cylinder.h"
#include "ray/intersect.h"
#include "ray/kdtree.h"
#include "ray/ray_utils.h"
#include "ray/sphere.h"
#include "scene/shape.h"
#include "scene/shape_data.h"

namespace ray {
namespace detail {
template <bool normal_and_uv>
__inline__ __host__ __device__ IntersectionOp<normal_and_uv>
solve_type(scene::Shape shape_type, const Eigen::Vector3f &point,
           const Eigen::Vector3f &direction, bool texture_map) {
  switch (shape_type) {
  case scene::Shape::Sphere:
    return solve_sphere<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cylinder:
    return solve_cylinder<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cube:
    return solve_cube<normal_and_uv>(point, direction, texture_map);
  case scene::Shape::Cone:
    return solve_cone<normal_and_uv>(point, direction, texture_map);
  }
}

template <bool normal_and_uv>
__inline__ __host__
    __device__ thrust::optional<BestIntersectionGeneral<normal_and_uv>>
    get_shape_intersection(Span<const scene::ShapeData> shapes,
                           const uint16_t shape_idx,
                           const Eigen::Vector3f &world_space_eye,
                           const Eigen::Vector3f &world_space_direction) {
  const auto &shape = shapes[shape_idx];
  const auto object_space_eye = shape.get_world_to_object() * world_space_eye;
  const auto object_space_direction =
      shape.get_world_to_object().linear() * world_space_direction;

  return optional_map(
      solve_type<normal_and_uv>(
          shape.get_shape(), object_space_eye, object_space_direction,
          normal_and_uv && shape.get_material().texture_data.has_value()),
      [&](const auto &value) {
        return BestIntersectionGeneral<normal_and_uv>(value, shape_idx);
      });
}

// Rename...
template <typename F>
__inline__ __host__ __device__ void solve_general_intersection(
    const KDTreeNodesRef &kdtree_nodes, const Traversal &traversal,
    Span<const Action> actions, Span<const scene::ShapeData> shapes,
    const Eigen::Vector3f &world_space_eye,
    const Eigen::Vector3f &world_space_direction, const unsigned &ignore_v,
    const uint8_t &disable, thrust::optional<BestIntersection> &best,
    bool is_first, bool use_traversals, bool use_kd_tree, const F &f) {
  if (!is_first && disable) {
    return;
  }

  auto solve_index = [&](unsigned shape_idx) {
    if (!is_first && ignore_v == shape_idx) {
      return false;
    }

    return f(get_shape_intersection<false>(shapes, shape_idx, world_space_eye,
                                           world_space_direction));
  };

  if (use_traversals) {
    for (unsigned action_index = traversal.start; action_index < traversal.end;
         action_index++) {
      if (solve_index(actions[action_index].shape_idx)) {
        return;
      }
    }
  } else if (use_kd_tree) {
    auto inv_direction = (1.0f / world_space_direction.array()).eval();

    struct StackData {
      uint16_t node_index;
      uint8_t depth;

      __device__ __host__ StackData(uint16_t node_index, uint8_t depth)
          : node_index(node_index), depth(depth) {}

      __device__ __host__ StackData() {}
    };

    if (kdtree_nodes.nodes.size() != 0) {
      std::array<StackData, 64> node_stack;
      node_stack[0] = StackData(kdtree_nodes.nodes.size() - 1, 0);
      uint8_t node_stack_size = 1;

      thrust::optional<std::array<uint16_t, 2>> start_end = thrust::nullopt;
      uint16_t current_shape_index = 0;

      while (node_stack_size != 0 || start_end.has_value()) {
        while (!start_end.has_value() && node_stack_size != 0) {
          const auto &stack_v = node_stack[node_stack_size - 1];

          const auto &current_node = kdtree_nodes.nodes[stack_v.node_index];

          auto bounding_intersection =
              current_node.get_contents().solveBoundingIntersection(
                  world_space_eye, inv_direction);

          if (bounding_intersection.has_value() &&
              (!best.has_value() ||
               best->intersection > *bounding_intersection)) {
            current_node.case_split_or_data(
                [&](const KDTreeSplit &split) {
                  const uint8_t axis = stack_v.depth % 3;
                  const auto intersection_point =
                      world_space_eye[axis] +
                      world_space_direction[axis] * *bounding_intersection;
                  auto first = split.left_index;
                  auto second = split.right_index;

                  if (intersection_point > split.division_point) {
                    auto temp = first;
                    first = second;
                    second = temp;
                  }

                  uint8_t new_depth = stack_v.depth + 1;
                  node_stack[node_stack_size - 1] =
                      StackData(second, new_depth);
                  node_stack_size++;
                  node_stack[node_stack_size - 1] = StackData(first, new_depth);
                  node_stack_size++; // counter act --;
                },
                [&](const std::array<uint16_t, 2> &data) {
                  start_end = thrust::make_optional(data);
                  current_shape_index = data[0];
                });
          }

          node_stack_size--;
        }

        if (start_end.has_value()) {
          auto local_end_shape = (*start_end)[1];
          uint16_t shape_idx = current_shape_index;
          current_shape_index++;
          if (current_shape_index >= local_end_shape) {
            start_end = thrust::nullopt;
          }

          if (solve_index(shape_idx)) {
            return;
          }
        }
      }
    }
  } else {
    for (uint16_t shape_idx = 0; shape_idx < 0 + kdtree_nodes.num_shape;
         shape_idx++) {
      if (solve_index(shape_idx)) {
        return;
      }
    }
  }
}
} // namespace detail
} // namespace ray
