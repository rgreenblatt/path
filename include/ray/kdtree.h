#pragma once

#include "ray/cube.h"

#include <thrust/optional.h>

namespace ray {
namespace detail {
struct KDTreeSplit {
  uint16_t left_index;
  uint16_t right_index;
  float division_point;
  KDTreeSplit(uint16_t left_index, uint16_t right_index, float division_point)
      : left_index(left_index), right_index(right_index),
        division_point(division_point) {}
  HOST_DEVICE
  KDTreeSplit() {}
};

struct ProjectedPoint {
  Eigen::Vector2f point;
  float dist;
  float shift_multiplier;

  HOST_DEVICE ProjectedPoint(const Eigen::Vector2f &point, float dist,
                             float shift_multiplier)
      : point(point), dist(dist), shift_multiplier(shift_multiplier) {}

  HOST_DEVICE ProjectedPoint() {}

  // Can we know more???
  HOST_DEVICE Eigen::Vector2f shift(const Eigen::Vector2f &shift) {
    return point + shift * shift_multiplier;
  }
};

struct ShiftedProjectedAABB {
  Eigen::Vector2f flattened_min;
  Eigen::Vector2f flattened_max;
  float axis_min;
  float axis_max;

  HOST_DEVICE
  ShiftedProjectedAABB(const Eigen::Vector2f &flattened_min,
                       const Eigen::Vector2f &flattened_max, float axis_min,
                       float axis_max)
      : flattened_min(flattened_min), flattened_max(flattened_max),
        axis_min(axis_min), axis_max(axis_max) {}

  HOST_DEVICE ShiftedProjectedAABB() {}

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ProjectedAABBInfo {
  Eigen::Vector2f flattened_min;
  Eigen::Vector2f flattened_max;
  Eigen::Vector2f intermediate;
  Eigen::Vector3f min_bound;
  Eigen::Vector3f max_bound;
  Eigen::Array2<bool> is_intermediate_max;
  float a_line_term;
  float c_line_term;
  float dist_pos;
  float dist_neg;
  float dist_axis_face;
  float dist_pos_face;
  float dist_neg_face;

  ProjectedAABBInfo(const Eigen::Vector2f &flattened_min,
                    const Eigen::Vector2f &flattened_max,
                    const Eigen::Vector2f &intermediate,
                    const Eigen::Vector3f &min_bound,
                    const Eigen::Vector3f max_bound,
                    const Eigen::Array2<bool> &is_intermediate_max,
                    float a_line_term, float c_line_term, float dist_pos,
                    float dist_neg)
      : flattened_min(flattened_min), flattened_max(flattened_max),
        intermediate(intermediate), min_bound(min_bound), max_bound(max_bound),
        is_intermediate_max(is_intermediate_max), a_line_term(a_line_term),
        c_line_term(c_line_term), dist_pos(dist_pos), dist_neg(dist_neg) {}

  // TODO
  ProjectedAABBInfo(const Eigen::Vector2f &flattened_min,
                    const Eigen::Vector2f &flattened_max)
      : flattened_min(flattened_min), flattened_max(flattened_max) {}

  HOST_DEVICE ProjectedAABBInfo() {}

  struct DistType {
    float dist;
    uint8_t type;

    DistType(float dist, uint8_t type) : dist(dist), type(type) {}
  };

  HOST_DEVICE thrust::optional<DistType>
  getInsideDist(const Eigen::Vector2f &location) const {
    if ((location.array() <= flattened_max.array()).all() &&
        (location.array() >= flattened_min.array()).all()) {
      auto inside_intermediate =
          ((location.array() >= intermediate.array()) != is_intermediate_max)
              .eval();
      if (inside_intermediate.all()) {
        return DistType(dist_axis_face, 0);
      } else {
        float signed_dist =
            location[0] * a_line_term + location[1] + c_line_term;
        if (signed_dist > 0) {
          return make_optional(signed_dist <= dist_pos,
                               DistType(dist_pos_face, 1));
        } else {
          return make_optional(signed_dist >= dist_neg,
                               DistType(dist_neg_face, 2));
        }
      }
    } else {
      return thrust::nullopt;
    }
  }

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ProjectedAABB {
  ProjectedPoint closer_min;
  ProjectedPoint closer_max;
  ProjectedPoint further_min;
  ProjectedPoint further_max;
  Eigen::Vector3f min_bound;
  Eigen::Vector3f max_bound;

  HOST_DEVICE
  ProjectedAABB(ProjectedPoint closer_min, ProjectedPoint closer_max,
                ProjectedPoint further_min, ProjectedPoint further_max,
                Eigen::Vector3f min_bound, Eigen::Vector3f max_bound)
      : closer_min(closer_min), closer_max(closer_max),
        further_min(further_min), further_max(further_max),
        min_bound(min_bound), max_bound(max_bound) {}

  HOST_DEVICE ProjectedAABB() {}

  ProjectedAABBInfo get_info() {
#if 1
    return ProjectedAABBInfo(closer_min.point.cwiseMin(further_min.point),
                             closer_max.point.cwiseMax(further_max.point));
#else
    return ProjectedAABBInfo();
#endif
  }

#if 0
  HOST_DEVICE ShiftedProjectedAABB shift(const Eigen::Vector2f &shift) {
    return ShiftedProjectedAABB(
        closer_min.shift(shift).cwiseMin(further_min.shift(shift)),
        closer_max.shift(shift).cwiseMax(further_max.shift(shift)), axis_min,
        axis_max);
  }
#endif

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline HOST_DEVICE Eigen::Vector2f get_not_axis(const Eigen::Vector3f &vec,
                                                uint8_t axis) {
  return Eigen::Vector2f(vec[(axis + 1) % 3], vec[(axis + 2) % 3]);
}

class AABB {
public:
  HOST_DEVICE
  AABB() {}

  HOST_DEVICE
  AABB(const Eigen::Vector3f &min_bound, const Eigen::Vector3f &max_bound)
      : min_bound_(min_bound), max_bound_(max_bound) {}

  HOST_DEVICE const Eigen::Vector3f &get_min_bound() const {
    return min_bound_;
  }

  HOST_DEVICE const Eigen::Vector3f &get_max_bound() const {
    return max_bound_;
  }

  // template for efficiency?
  template <uint8_t axis>
  HOST_DEVICE ProjectedAABB project_to_axis(
      bool is_loc, float value, const Eigen::Vector3f &loc_or_dir) const {
    bool max_is_closer =
        std::abs(value - max_bound_[axis]) < std::abs(value - min_bound_[axis]);

    Eigen::Vector3f closer_max = max_bound_;
    closer_max[axis] = max_is_closer ? closer_max[axis] : min_bound_[axis];
    Eigen::Vector3f closer_min = min_bound_;
    closer_min[axis] = closer_max[axis];

    Eigen::Vector3f further_max = max_bound_;
    further_max[axis] = max_is_closer ? min_bound_[axis] : further_max[axis];
    Eigen::Vector3f further_min = min_bound_;
    further_min[axis] = further_max[axis];

    auto project = [&](const Eigen::Vector3f &point) {
      const auto dir = is_loc ? (point - loc_or_dir).eval() : loc_or_dir;
      float intersect_dist = (value - point[axis]) / dir[axis];
      const auto intersect_point = (point + dir * intersect_dist).eval();
      const auto camera_point_dist = dir.norm();
      const auto intersect_point_dist = (point - intersect_point).norm();

      return ProjectedPoint(get_not_axis(intersect_point, axis),
                            // TODO dist???
                            camera_point_dist,
                            intersect_point_dist / camera_point_dist);
    };

    return ProjectedAABB(project(closer_min), project(closer_max),
                         project(further_min), project(further_max), min_bound_,
                         max_bound_);
  }

  HOST_DEVICE ProjectedAABB
  project_to_axis(bool is_loc, uint8_t axis, float value,
                  const Eigen::Vector3f &loc_or_dir) const {
    switch (axis) {
    case 0:
      return project_to_axis<0>(is_loc, value, loc_or_dir);
    case 1:
      return project_to_axis<1>(is_loc, value, loc_or_dir);
    case 2:
      return project_to_axis<2>(is_loc, value, loc_or_dir);
    }

    abort();
  }

  // needs to be inline
  HOST_DEVICE thrust::optional<float>
  solveBoundingIntersection(const Eigen::Vector3f &point,
                            const Eigen::Vector3f &inv_direction) const {
    auto t_0 = (min_bound_ - point).cwiseProduct(inv_direction).eval();
    auto t_1 = (max_bound_ - point).cwiseProduct(inv_direction).eval();
    auto t_min = t_0.cwiseMin(t_1);
    auto t_max = t_0.cwiseMax(t_1);

    float max_of_min = t_min.maxCoeff();
    float min_of_max = t_max.minCoeff();

    return make_optional(max_of_min <= min_of_max, max_of_min);
  }

  // needs to be inline
  // negative check?????
  HOST_DEVICE thrust::optional<float> solveBoundingFrustumIntersection(
      const Eigen::Vector3f &point, const Eigen::Vector3f &top_right,
      const Eigen::Vector3f &bottom_right, const Eigen::Vector3f &top_left,
      const Eigen::Vector3f &bottom_left) const {
    auto solve_intersect = [&](const Eigen::Vector3f &dir) {
      return solveBoundingIntersection(point, dir);
    };

    auto intersect_sol =
        optional_min(solve_intersect(top_right), solve_intersect(bottom_right),
                     solve_intersect(top_left), solve_intersect(bottom_left));

    if (intersect_sol.has_value()) {
      return intersect_sol;
    }

    auto check_in_half_space = [&](const Eigen::Vector3f &first,
                                   const Eigen::Vector3f &second) {
      const Eigen::Vector3f normal = first.cross(second);
      auto check_point_in_half_space =
          [&](const Eigen::Vector3f &test_point) -> bool {
        return (test_point - point).dot(normal) > 0.0f;
        ;
      };

      return check_point_in_half_space(min_bound_) &&
             check_point_in_half_space(max_bound_);
    };

    bool within = check_in_half_space(top_left, top_right) &&
                  check_in_half_space(top_right, bottom_right) &&
                  check_in_half_space(bottom_right, bottom_left) &&
                  check_in_half_space(bottom_left, top_left);

    if (within) {
      float min_dist = std::numeric_limits<float>::max();

      for (bool is_min_x = false; !is_min_x; is_min_x = true) {
        for (bool is_min_y = false; !is_min_y; is_min_y = true) {
          for (bool is_min_z = false; !is_min_z; is_min_z = true) {
            Eigen::Vector3f aa_bb_point(
                is_min_x ? min_bound_.x() : max_bound_.x(),
                is_min_y ? min_bound_.y() : max_bound_.y(),
                is_min_z ? min_bound_.z() : max_bound_.z());

            min_dist = std::min(min_dist, (point - aa_bb_point).norm());
          }
        }
      }

      return thrust::make_optional(min_dist);
    } else {
      return thrust::nullopt;
    }
  }

#if 0
  HOST_DEVICE thrust::optional<float>
  solveBoundingIntersectionOld(const Eigen::Vector3f &point,
                               const Eigen::Vector3f &direction) const {
    Eigen::Vector3f kd_space_point = point.cwiseProduct(scale) + translate;

    float half_plus_epsilon = 0.5f + epsilon;
    if (std::abs(kd_space_point.x()) < half_plus_epsilon &&
        std::abs(kd_space_point.y()) < half_plus_epsilon &&
        std::abs(kd_space_point.z()) < half_plus_epsilon) {
      // point is inside bounding box
      return 0.0f;
    }

    return solve_cube<false>(kd_space_point, direction.cwiseProduct(scale),
                             false);
  }
#endif

private:
  Eigen::Vector3f min_bound_;
  Eigen::Vector3f max_bound_;
#if 0
  Eigen::Vector3f scale;
  Eigen::Vector3f translate;
#endif

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Action {
#if 0
  bool is_known;
  float min_dist;
#endif
  uint16_t shape_idx;
  /* uint16_t shape_idx_end; */

  Action(uint16_t shape_idx) : shape_idx(shape_idx) {}

  HOST_DEVICE
  Action() {}
};

struct Traversal {
#if 0
  bool is_final;
#endif
  uint16_t start;
  uint16_t size;
#if 0
  uint8_t axis;
  float comp;
  uint16_t traversal_index_smaller;
  uint16_t traversal_index_larger;
#endif

  HOST_DEVICE Traversal() {}

  HOST_DEVICE Traversal(uint16_t start, uint16_t size)
      :
#if 0
        is_final(true),
#endif
        start(start), size(size) {
  }

#if 0
  Traversal(uint8_t axis, float comp, uint16_t traversal_index_smaller,
           uint16_t traversal_index_larger)
      : is_final(false), axis(axis), comp(comp),
        traversal_index_smaller(traversal_index_smaller),
        traversal_index_larger(traversal_index_larger) {}
#endif
};

struct LightTraversalData {
  unsigned offset;
  Eigen::Vector2f side_min;
  Eigen::Vector2f side_max;
  uint8_t axis;
  float value;

  HOST_DEVICE LightTraversalData(unsigned offset, Eigen::Vector2f side_min,
                                 Eigen::Vector2f side_max, uint8_t axis,
                                 float value)
      : offset(offset), side_min(side_min), side_max(side_max), axis(axis),
        value(value) {}

  HOST_DEVICE LightTraversalData() {}
};

template <typename Contents> struct KDTreeNode {
  HOST_DEVICE KDTreeNode() {}

  HOST_DEVICE KDTreeNode(const KDTreeSplit &split, const Contents &contents)
      : contents_(contents), is_split_(true), split_(split) {
    split_ = split;
    is_split_ = true;
  }

  KDTreeNode(const std::array<uint16_t, 2> &data, const Contents &contents)
      : contents_(contents), is_split_(false), data_(data) {}

  template <typename FSplit, typename FData>
  constexpr auto case_split_or_data(const FSplit &split_case,
                                    const FData &data_case) const {
    if (is_split_) {
      return split_case(split_);
    } else {
      return data_case(data_);
    }
  }

  HOST_DEVICE bool get_is_split() const { return is_split_; }

  template <typename F> auto transform(const F &f) const {
    auto get_out = [&](const auto &v) {
      return KDTreeNode<decltype(f(contents_))>(v, f(contents_));
    };

    if (is_split_) {
      return get_out(split_);
    } else {
      return get_out(data_);
    }
  }

  HOST_DEVICE const Contents &get_contents() const { return contents_; }

private:
  Contents contents_;

  bool is_split_;
  KDTreeSplit split_;
  std::array<uint16_t, 2> data_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Bounds {
  Eigen::Vector3f min;
  Eigen::Vector3f center;
  Eigen::Vector3f max;

  Bounds() {}
  Bounds(const Eigen::Vector3f &min, const Eigen::Vector3f &center,
         const Eigen::Vector3f &max)
      : min(min), center(center), max(max) {}
};

inline std::tuple<Eigen::Vector3f, Eigen::Vector3f>
get_shape_bounds(const scene::ShapeData &shape) {
  Eigen::Vector3f min_bound(std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max());
  Eigen::Vector3f max_bound(std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::lowest(),
                            std::numeric_limits<float>::lowest());
  for (auto x : {-0.5f, 0.5f}) {
    for (auto y : {-0.5f, 0.5f}) {
      for (auto z : {-0.5f, 0.5f}) {
        Eigen::Vector3f transformed_edge =
            Eigen::Vector3f(shape.get_transform() * Eigen::Vector3f(x, y, z));
        min_bound = min_bound.cwiseMin(transformed_edge);
        max_bound = max_bound.cwiseMax(transformed_edge);
      }
    }
  }

  return std::make_tuple(min_bound, max_bound);
}

std::vector<KDTreeNode<AABB>> construct_kd_tree(scene::ShapeData *shapes,
                                                uint16_t num_shapes);

inline HOST_DEVICE Eigen::Vector3f
initial_world_space_direction(unsigned x, unsigned y, unsigned width,
                              unsigned height,
                              const Eigen::Vector3f &world_space_eye,
                              const scene::Transform &m_film_to_world) {
  const Eigen::Vector3f camera_space_film_plane(
      (2.0f * static_cast<float>(x)) / static_cast<float>(width) - 1.0f,
      (-2.0f * static_cast<float>(y)) / static_cast<float>(height) + 1.0f,
      -1.0f);
  const auto world_space_film_plane = m_film_to_world * camera_space_film_plane;

  return (world_space_film_plane - world_space_eye).normalized();
}

inline HOST_DEVICE Eigen::Vector2f
get_intersection_point(const Eigen::Vector3f &dir, float value_to_project_to,
                       const Eigen::Vector3f &world_space_eye, uint8_t axis) {
  float dist = (value_to_project_to - world_space_eye[axis]) / dir[axis];

  return (dist * get_not_axis(dir, axis) + get_not_axis(world_space_eye, axis))
      .eval();
}

std::tuple<std::vector<Traversal>, std::vector<uint8_t>, std::vector<Action>>
get_traversal_grid_from_transform(
    const std::vector<std::pair<ProjectedAABBInfo, uint16_t>> &shapes,
    unsigned width, unsigned height, const scene::Transform &m_film_to_world,
    unsigned block_dim_x, unsigned block_dim_y, unsigned num_blocks_x,
    unsigned num_blocks_y, uint8_t axis, float value_to_project_to);

std::tuple<std::vector<Traversal>, std::vector<uint8_t>, std::vector<Action>>
get_traversal_grid_from_bounds(
    const std::vector<std::pair<ProjectedAABBInfo, uint16_t>> &shapes,
    const Eigen::Array2f &min_bound, const Eigen::Array2f &max_bound,
    unsigned num_blocks_x, unsigned num_blocks_y);
} // namespace detail
} // namespace ray
