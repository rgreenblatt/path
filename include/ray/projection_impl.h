#pragma once

#include "ray/intersect.h"
#include "ray/projection.h"
#include <boost/range/adaptor/indexed.hpp>

#include <dbg.h>

namespace ray {
namespace detail {
inline Eigen::Array2f get_xy(const Eigen::Vector3f &vec) {
  return Eigen::Array2f(vec[0], vec[1]);
}

using Triangle = std::array<Eigen::Vector3f, 3>;

inline Triangle transform_triangle(const Eigen::Projective3f &transform,
                                   const Triangle &tri) {
  auto get_nth = [&](uint8_t n) { return apply_projective(tri[n], transform); };

  return {get_nth(0), get_nth(1), get_nth(2)};
}

inline bool get_triangle_normal_sign(const Triangle &tri) {
  return (tri[1] - tri[0]).cross(tri[2] - tri[1]).normalized().z() > 0.0f;
}

const static std::array<Triangle, 12> cube_polys = {{
    {{{0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}}},
    {{{0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}}},

    {{{0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}}},
    {{{-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f}}},

    {{{0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}}},
    {{{0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}}},

    {{{-0.5f, 0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, 0.5f}}},
    {{{-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}}},

    {{{-0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}}},
    {{{-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}}},

    {{{0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}}},
    {{{-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}}},
}};

constexpr float pi = static_cast<float>(M_PI);

// TODO bounding....
template <bool bounding, unsigned num_sides, unsigned num_divisions,
          unsigned num_triangles_per_side = 2 * num_divisions - 2,
          unsigned num_triangles = num_triangles_per_side *num_sides>
inline std::array<Triangle, num_triangles> get_sphere_polys() {
  static_assert(num_sides >= 3);
  static_assert(num_divisions >= 3);

  constexpr float radius = 0.5f;

  std::array<Triangle, num_triangles> triangles;

  size_t triangle_index = 0;

  for (uint16_t side = 0; side < num_sides; side++) {
    float theta_0 = ((static_cast<float>(side) / num_sides) - 0.5f) * 2 * pi;
    float theta_1 =
        ((static_cast<float>(side + 1) / num_sides) - 0.5f) * 2 * pi;
    for (unsigned div = 0; div < num_divisions; div++) {
      float phi_0 = (static_cast<float>(div) / num_divisions) * pi;
      float phi_1 = (static_cast<float>(div + 1) / num_divisions) * pi;

      auto get_vertex = [&](float theta, float phi) {
        return Eigen::Vector3f(radius * std::sin(phi) * std::cos(theta),
                               radius * std::cos(phi),
                               radius * std::sin(phi) * std::sin(theta));
      };

      if (div != 0) {
        triangles[triangle_index] = {get_vertex(theta_1, phi_0),
                                     get_vertex(theta_0, phi_1),
                                     get_vertex(theta_0, phi_0)};
        triangle_index++;
      }

      if (div != num_divisions - 1) {
        triangles[triangle_index] = {get_vertex(theta_1, phi_0),
                                     get_vertex(theta_1, phi_1),
                                     get_vertex(theta_0, phi_1)};
        triangle_index++;
      }
    }
  }

  return triangles;
}

const static auto within_sphere_polys = get_sphere_polys<false, 3, 3>();
const static auto bounding_sphere_polys = cube_polys;
const static auto bounding_cone_polys = cube_polys;
const static auto bounding_cylinder_polys = cube_polys;

template <size_t num_triangles, typename F>
inline void
project_triangles(const Eigen::Projective3f &transform,
                  const TriangleProjector &projector,
                  const std::array<Triangle, num_triangles> &triangles,
                  bool flip_x, bool flip_y, const F &add_tri) {
  for (const auto &triangle : triangles) {
    Triangle transformed_triangle = transform_triangle(transform, triangle);

    projector.visit([&](const auto &v) {
      using T = std::decay_t<decltype(v)>;
      if constexpr (std::is_same<T, DirectionPlane>::value) {
        for (auto &transformed_point : transformed_triangle) {
          transformed_point.head<2>() = get_intersection_point(
              v.is_loc ? (v.loc_or_dir - transformed_point).eval()
                       : v.loc_or_dir,
              v.projection_value, transformed_point, v.axis);
        }
      }
    });

    if (get_triangle_normal_sign(transformed_triangle)) {
      std::array<Eigen::Array2f, 3> projected_points;
      std::transform(transformed_triangle.begin(), transformed_triangle.end(),
                     projected_points.begin(),
                     [&](const Eigen::Vector3f &point) {
                       Eigen::Array2f new_point = point.head<2>();
                       if (flip_x) {
                         new_point.x() *= -1.0f;
                       }
                       if (flip_y) {
                         new_point.y() *= -1.0f;
                       }
                       return new_point;
                     });

      add_tri(projected_points);
    }
  }
}

inline void project_shape(const scene::ShapeData &shape,
                          const TriangleProjector &projector,
                          std::vector<ProjectedTriangle> &projected_triangles,
                          bool flip_x = false, bool flip_y = false) {
  Eigen::Projective3f transform =
      projector.get_total_transform(shape.get_transform());
  auto project_triangles_s = [&](const auto &triangles) {
    project_triangles(transform, projector, triangles, flip_x, flip_y,
                      [&](const std::array<Eigen::Array2f, 3> &points) {
                        projected_triangles.push_back(
                            ProjectedTriangle(points));
                      });
  };

  switch (shape.get_shape()) {
  case scene::Shape::Sphere:
    /* project_triangles_s(within_sphere_polys, true); */
    project_triangles_s(bounding_sphere_polys);
    break;
  case scene::Shape::Cube:
    project_triangles_s(cube_polys);
    break;
  case scene::Shape::Cone:
    project_triangles_s(bounding_cone_polys);
    break;
  case scene::Shape::Cylinder:
    project_triangles_s(bounding_cylinder_polys);
    break;
  }
}
} // namespace detail
} // namespace ray
