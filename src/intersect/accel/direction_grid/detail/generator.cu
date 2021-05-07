#include "data_structure/copyable_to_vec.h"
#include "intersect/accel/direction_grid/detail/generator.h"
#include "intersect/accel/direction_grid/direction_grid_impl.h"
#include "intersect/triangle_impl.h"
#include "lib/assert.h"
#include "lib/info/timer.h"

#include <unordered_set>

namespace intersect {
namespace accel {
namespace direction_grid {

template <ExecutionModel exec>
RefPerm<typename DirectionGrid<exec>::Ref>
DirectionGrid<exec>::Generator::gen(const Settings &,
                                    SpanSized<const Triangle> triangles) {
  unsigned grid = 5;

  // TODO: z-curve sort...

  struct PosInfo {
    bool is_positive;
    unsigned constant_axis;
    unsigned i_axis;
    unsigned j_axis;
    Eigen::Vector3<unsigned> min_face_coords;
    Eigen::Vector3<unsigned> max_face_coords;
  };

  // holds the overall idx
  HostVector<HostVector<unsigned>> voxel_connecting_faces(grid * grid * grid);

  const unsigned num_box_faces = 6;

  unsigned face_grid_pair_idx = 0;
  for (unsigned face = 0; face < num_box_faces; ++face) {
    for (unsigned other_face = 0; other_face < face; ++other_face) {
      for (unsigned i_face = 0; i_face < grid; ++i_face) {
        for (unsigned j_face = 0; j_face < grid; ++j_face) {
          for (unsigned i_other_face = 0; i_other_face < grid; ++i_other_face) {
            for (unsigned j_other_face = 0; j_other_face < grid;
                 ++j_other_face, ++face_grid_pair_idx) {
              [[maybe_unused]] unsigned expected_face_grid_pair_idx =
                  detail::DirectionGridRef::idx(
                      {
                          detail::DirectionGridRef::IntersectionIdxs{
                              .face = face,
                              .i = i_face,
                              .j = j_face,
                          },
                          {
                              .face = other_face,
                              .i = i_other_face,
                              .j = j_other_face,
                          },
                      },
                      grid);
              debug_assert(expected_face_grid_pair_idx == face_grid_pair_idx);
              auto get_info = [grid](unsigned face, unsigned i,
                                     unsigned j) -> PosInfo {
                unsigned constant_axis = face % 3;

                bool is_positive = face / 3 == 1;

                unsigned i_axis = (constant_axis + 1) % 3;
                unsigned j_axis = (constant_axis + 2) % 3;

                Eigen::Vector3<unsigned> min_face_coords;
                min_face_coords[constant_axis] = is_positive ? grid : 0;
                min_face_coords[i_axis] = i;
                min_face_coords[j_axis] = j;
                Eigen::Vector3<unsigned> max_face_coords = min_face_coords;
                ++max_face_coords[i_axis];
                ++max_face_coords[j_axis];

                return {
                    .is_positive = is_positive,
                    .constant_axis = constant_axis,
                    .i_axis = i_axis,
                    .j_axis = j_axis,
                    .min_face_coords = min_face_coords,
                    .max_face_coords = max_face_coords,
                };
              };

              PosInfo face_info = get_info(face, i_face, j_face);
              PosInfo other_face_info =
                  get_info(other_face, i_other_face, j_other_face);

              auto get_loc =
                  [grid](const PosInfo &info) -> Eigen::Vector3<unsigned> {
                Eigen::Vector3<unsigned> loc = info.min_face_coords;
                if (info.is_positive) {
                  debug_assert(loc[info.constant_axis] == grid);
                  loc[info.constant_axis] -= 1;
                }
                return loc;
              };

              Eigen::Vector3<unsigned> start_loc = get_loc(face_info);
              Eigen::Vector3<unsigned> end_loc = get_loc(other_face_info);

              if ((start_loc.array() == end_loc.array()).all()) {
                voxel_connecting_faces[start_loc[0] * grid * grid +
                                       start_loc[1] * grid + start_loc[2]]
                    .push_back(face_grid_pair_idx);
                continue;
              }

              std::array<Eigen::Vector3f, 4> connecting_edges;
              std::array<Eigen::Vector3<unsigned>, 4> connecting_start_point;

              if (face_info.constant_axis == other_face_info.constant_axis) {
                debug_assert(face_info.is_positive !=
                             other_face_info.is_positive);
                debug_assert(face_info.i_axis == other_face_info.i_axis);
                debug_assert(face_info.j_axis == other_face_info.j_axis);

                unsigned overall_idx = 0;
                for (bool i_is_min : {false, true}) {
                  for (bool j_is_min : {false, true}) {
                    auto get_point = [&](const PosInfo &info) {
                      Eigen::Vector3<unsigned> point;
                      auto assign_axis = [&](unsigned axis, bool is_min) {
                        point[axis] = (is_min ? info.min_face_coords
                                              : info.max_face_coords)[axis];
                      };
                      assign_axis(info.constant_axis, false);
                      assign_axis(info.i_axis, i_is_min);
                      assign_axis(info.j_axis, j_is_min);

                      return point;
                    };

                    auto face_point = get_point(face_info);
                    auto other_face_point = get_point(other_face_info);

                    connecting_edges[overall_idx] =
                        other_face_point.template cast<float>() -
                        face_point.template cast<float>();
                    connecting_start_point[overall_idx] = face_point;

                    ++overall_idx;
                  }
                }
              } else {
                // TODO: consider cleaning this up!
                unsigned shared_axis =
                    face_info.i_axis != other_face_info.constant_axis
                        ? face_info.i_axis
                        : face_info.j_axis;

                debug_assert_assume(shared_axis != unsigned(-1));

                unsigned not_shared_face_axis = face_info.i_axis != shared_axis
                                                    ? face_info.i_axis
                                                    : face_info.j_axis;
                unsigned not_shared_other_face_axis =
                    other_face_info.i_axis != shared_axis
                        ? other_face_info.i_axis
                        : other_face_info.j_axis;

                debug_assert(not_shared_face_axis !=
                             not_shared_other_face_axis);
                debug_assert(not_shared_face_axis != shared_axis);
                debug_assert(not_shared_other_face_axis != shared_axis);

                // pos to pos, min to min
                // pos to neg, max to min
                // neg to pos, max to min
                bool not_shared_is_min_to_max =
                    face_info.is_positive != other_face_info.is_positive;

                unsigned overall_idx = 0;
                for (bool shared_is_min : {false, true}) {
                  for (bool not_shared_face_is_min : {false, true}) {
                    bool not_shared_other_face_is_min =
                        not_shared_is_min_to_max ? !not_shared_face_is_min
                                                 : not_shared_face_is_min;

                    auto get_point = [&](const PosInfo &info,
                                         unsigned not_shared_axis,
                                         bool not_shared_is_min) {
                      Eigen::Vector3<unsigned> point;
                      auto assign_axis = [&](unsigned axis, bool is_min) {
                        point[axis] = (is_min ? info.min_face_coords
                                              : info.max_face_coords)[axis];
                      };
                      assign_axis(info.constant_axis, false);
                      assign_axis(shared_axis, shared_is_min);
                      assign_axis(not_shared_axis, not_shared_is_min);

                      return point;
                    };

                    auto face_point = get_point(face_info, not_shared_face_axis,
                                                not_shared_face_is_min);
                    auto other_face_point =
                        get_point(other_face_info, not_shared_other_face_axis,
                                  not_shared_other_face_is_min);

                    connecting_edges[overall_idx] =
                        other_face_point.template cast<float>() -
                        face_point.template cast<float>();
                    connecting_start_point[overall_idx] = face_point;

                    ++overall_idx;
                  }
                }
              }

              std::unordered_set<unsigned> voxels;

              for (unsigned i = 0; i < connecting_edges.size(); ++i) {
                debug_assert((connecting_edges[i].array() != 0.f).any());

                Eigen::Vector3f edge_no_zeros = connecting_edges[i];
                for (unsigned i = 0; i < unsigned(edge_no_zeros.size()); ++i) {
                  if (edge_no_zeros[i] == 0.f) {
                    edge_no_zeros[i] = 1e-20f;
                  }
                }
                Eigen::Vector3f dists = 1.f / edge_no_zeros.array();
                Eigen::Vector3<bool> is_pos = dists.array() > 0.f;
                Eigen::Vector3f current_point =
                    connecting_start_point[i].template cast<float>();

                Eigen::Vector3<unsigned> current_loc = start_loc;

                for (unsigned axis : {face_info.i_axis, face_info.j_axis}) {
                  if (connecting_start_point[i][axis] > start_loc[axis] &&
                      connecting_edges[i][axis] > 0.f) {
                    debug_assert(connecting_start_point[i][axis] < grid);
                    ++start_loc[axis];
                    debug_assert(start_loc[axis] < grid);
                  }
                  if (connecting_start_point[i][axis] == start_loc[axis] &&
                      connecting_edges[i][axis] < 0.f) {
                    debug_assert(start_loc[axis] > 0);
                    --start_loc[axis];
                  }
                }

                while (true) {
                  voxels.insert(current_loc[0] * grid * grid +
                                current_loc[1] * grid + current_loc[2]);

                  auto get_value = [&](unsigned axis) {
                    return is_pos[axis] ? std::floor(current_point[axis] + 1.f)
                                        : std::ceil(current_point[axis] - 1.f);
                  };

                  Eigen::Vector3f next_points = {get_value(0), get_value(1),
                                                 get_value(2)};

                  Eigen::Vector3f next_props =
                      dists.array() * (next_points - current_point).array();

                  unsigned min_axis = -1;
                  float min_prop = std::numeric_limits<float>::max();
                  for (unsigned axis = 0; axis < 3; ++axis) {
                    if (next_props[axis] < min_prop) {
                      min_axis = axis;
                      min_prop = next_props[axis];
                    }
                  }
                  debug_assert(min_prop > 0.f);

                  current_point += connecting_edges[i] * min_prop;
                  for (unsigned axis = 0; axis < 3; ++axis) {
                    // floating point equality check is actually right here (I
                    // think...)
                    if (next_props[axis] == min_prop) {
                      current_loc[axis] += is_pos[axis] ? 1u : -1u;
                      if (current_loc[axis] == grid ||
                          current_loc[axis] ==
                              std::numeric_limits<unsigned>::max()) {
                        goto break_outer;
                      }
                    }
                  }
                }
              break_outer : {}
              }

              for (unsigned voxel : voxels) {
                voxel_connecting_faces[voxel].push_back(face_grid_pair_idx);
              }
            }
          }
        }
      }
    }
  }

  unsigned num_face_pairs = num_box_faces * (num_box_faces - 1) / 2;
  unsigned size = num_face_pairs * grid * grid * grid * grid;
  always_assert(face_grid_pair_idx == size);

  HostVector<HostVector<unsigned>> direction_idxs(size);

  AABB overall_aabb = AABB::empty();

  for (const Triangle &triangle : triangles) {
    overall_aabb = overall_aabb.union_other(triangle.bounds());
  }

  // intersect::accel::detail::bvh::print_out_bvh(
  //     std::array{intersect::accel::detail::bvh::Node{.aabb = overall_aabb}},
  //     std::array<unsigned, 0>{}, 0);

  Eigen::Vector3f total = overall_aabb.max_bound - overall_aabb.min_bound;
  debug_assert((total.array() >= 0.f).all());
  // fix zeros
  total = total.cwiseMax(1e-20f);
  Eigen::Vector3f inv_total = 1.f / total.array();

  for (unsigned i = 0; i < triangles.size(); ++i) {
    const Triangle &triangle = triangles[i];
    const auto bounds = triangle.bounds();

    // dbg(i);
    // print_triangle(triangle);

    Eigen::Vector3f min_props =
        (bounds.min_bound - overall_aabb.min_bound).array() * inv_total.array();
    Eigen::Vector3f max_props =
        (bounds.max_bound - overall_aabb.min_bound).array() * inv_total.array();

    Eigen::Vector3<unsigned> begin_bound =
        (min_props * grid).array().floor().template cast<unsigned>();
    Eigen::Vector3<unsigned> end_bound =
        (max_props * grid).array().ceil().template cast<unsigned>();

    for (unsigned axis = 0; axis < 3; ++axis) {
      if (overall_aabb.min_bound[axis] == overall_aabb.max_bound[axis]) {
        debug_assert(begin_bound[axis] == 0);
        debug_assert(end_bound[axis] == 0);
        end_bound[axis] = grid;
      } else if (begin_bound[axis] == end_bound[axis]) {
        // we are on the border so we need to choose
        if (begin_bound[axis] > 0) {
          --begin_bound[axis];
        } else {
          ++end_bound[axis];
          debug_assert(end_bound[axis] <= grid);
        }
      }
    }

    for (unsigned x = begin_bound.x(); x < end_bound.x(); ++x) {
      for (unsigned y = begin_bound.y(); y < end_bound.y(); ++y) {
        for (unsigned z = begin_bound.z(); z < end_bound.z(); ++z) {
          for (unsigned idx :
               voxel_connecting_faces[x * grid * grid + y * grid + z]) {
            direction_idxs[idx].push_back(i);
          }
        }
      }
    }
  }

  HostVector<unsigned> overall_idxs;
  HostVector<StartEnd<unsigned>> direction_idxs_final(size);

  for (unsigned i = 0; i < size; ++i) {
    unsigned start = overall_idxs.size();
    for (unsigned triangle_idx : direction_idxs[i]) {
      overall_idxs.push_back(triangle_idx);
    }
    unsigned end = overall_idxs.size();
    direction_idxs_final[i] = {
        .start = start,
        .end = end,
    };
  }

  copy_to_vec(overall_idxs, overall_idxs_);
  copy_to_vec(direction_idxs_final, direction_idxs_);

  std::vector<unsigned> permutation(triangles.size());

  for (unsigned i = 0; i < triangles.size(); ++i) {
    permutation[i] = i;
  }

  copy_to_vec(std::array{overall_aabb}, node_bounds_);
  copy_to_vec(std::array{grid}, node_grid_);

  return {
      .ref =
          {
              .overall_idxs = overall_idxs_,
              .direction_idxs = direction_idxs_,
              .node_bounds = node_bounds_,
              .node_grid = node_grid_,
          },
      .permutation = permutation,
  };
}

template class DirectionGrid<ExecutionModel::CPU>::Generator;
#ifndef CPU_ONLY
template class DirectionGrid<ExecutionModel::GPU>::Generator;
#endif
} // namespace direction_grid
} // namespace accel
} // namespace intersect
