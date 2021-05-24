from mayavi import mlab
import numpy as np


def draw_tri(vertices, color=(0, 0, 0)):
    mlab.triangular_mesh(vertices[:, 0],
                         vertices[:, 1],
                         vertices[:, 2], [[0, 1, 2]],
                         color=color)


def plot_points(points, color=(0, 0, 0), scale_factor=0.1):
    points = np.array(points)
    mlab.points3d(points[:, 0],
                  points[:, 1],
                  points[:, 2],
                  scale_factor=scale_factor,
                  color=color)


def draw_centroid(tri, color=(0, 0, 0)):
    plot_points(tri.mean(axis=0, keepdims=True), color=color)


def draw_region(tri, region, color=(0, 0, 0)):
    if region is None:
        return
    if region == 'all':
        points = tri
    else:
        region = np.array(region)
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]

        points = (v0[None, :] * region[:, 0, None] +
                  v1[None, :] * region[:, 1, None] + tri[0])

    plot_points(points, color=color, scale_factor=0.05)


def draw_vec(origin, dir, color=(0, 0, 0)):
    mlab.quiver3d(origin[0],
                  origin[1],
                  origin[2],
                  dir[0],
                  dir[1],
                  dir[2],
                  color=color)


def connect_tris(vertices_first, vertices_second, color=(0, 0, 0)):
    for i in range(3):
        for j in range(3):
            mlab.plot3d(*[[vertices_first[i][k], vertices_second[j][k]]
                          for k in range(3)],
                        color=color)


def main():
    onto_some_blocking = [
        [1.94204e-07, 0.90485],
        [0.268593, 0.548799],
        [0.196544, 0.499953],
        [0, 0.457839],
        [1.94204e-07, 0.90485],
    ]
    onto_totally_blocked = None
    onto_from_each_point_0 = None
    onto_from_each_point_1 = [
        [0.268593, 0.548799],
        [-5.55112e-17, 0.562037],
        [0, 0.904851],
        [0.268593, 0.548799],
    ]
    onto_from_each_point_2 = [
        [0, 0.457838],
        [0, 0.493864],
        [0.196544, 0.499952],
        [0, 0.457838],
    ]
    onto_centroid_shadow = [
        [0.200095, 0.522615],
        [0, 0.502184],
        [0, 0.560288],
        [0.200095, 0.522615],
    ]
    onto_light_some_blocking = [
        [0, 0.158543],
        [0, 1],
        [1, 0],
        [0.0952211, 0],
        [0, 0.158543],
    ]
    onto_light_totally_blocked = None
    onto_light_from_each_point_0 = None
    onto_light_from_each_point_1 = None
    onto_light_from_each_point_2 = None
    onto_light_from_each_point_3 = None
    onto_light_centroid_shadow = None
    triangle_onto = [
        [0.0691192, -1.08507, -8.88178e-16],
        [0.614401, 1.38778e-17, 0],
        [-0.68352, 1.08507, 2.22045e-16],
    ]

    triangle_blocking = [
        [-0.130465, 0.454776, 0.128844],
        [-0.0644418, 1.01657, 0.73711],
        [-0.176645, 0.00391542, 1.00172],
    ]

    triangle_light = [
        [0.261354, 1.5507, 0.338938],
        [0.452193, 0.954207, 1.24924],
        [0.331038, 2.05854, 0.984071],
    ]

    onto_region = [
        [0, 1],
        [0.755049, 0.244951],
        [0.518761, 0],
        [0, 0],
        [0, 1],
    ]
    blocking_region = 'all'
    light_region = 'all'

    triangle_onto = np.array(triangle_onto)
    triangle_blocking = np.array(triangle_blocking)
    triangle_light = np.array(triangle_light)

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    # draw_tri(triangle_onto, color=(0, 0, 1))
    # draw_tri(triangle_blocking, color=(0, 0, 0))
    # draw_tri(triangle_light, color=(1, 0, 0))

    l_vals = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [0, 0],
    ])
    r_vals = np.array([
        [0.79548724912300062, 0.20451275087699938],
        [0.90832627069587246, 0],
        [0.81606331509780206, -0.000026046377636923523],
        [0.45017683775912865, 0.54978004359427868],
        [0.79548724912300062, 0.20451275087699938],
    ])

    mlab.points3d(l_vals[:, 0],
                  l_vals[:, 1],
                  np.zeros_like(l_vals[:, 1]),
                  scale_factor=0.01)
    mlab.points3d(r_vals[:, 0],
                  r_vals[:, 1],
                  np.zeros_like(r_vals[:, 1]),
                  scale_factor=0.01,
                  color=(1, 0, 0))

    values = np.array([
        # [0.79468791946706574, 0.20531198072125234],
        [0.45017683775912865, 0.54978004359427868],
        [0.81604601209617344, -4.570185518915082E-8],
        [0, 0],
        [0, 1],
        [0.45017683775912865, 0.54978004359427868],
        # [0.79468791946706574, 0.20531198072125234],
    ])

    mlab.plot3d(
        values[:, 0],
        values[:, 1],
        np.zeros_like(values[:, 1]),
        # scale_factor=0.01,
        color=(0, 1, 0))

    # draw_centroid(triangle_light, color=(0, 0, 0))
    # draw_centroid(triangle_onto, color=(0, 0, 0))

    # draw_region(triangle_onto, onto_region, color=(0, 0, 1))
    # draw_region(triangle_blocking, blocking_region, color=(0, 0, 0))
    # draw_region(triangle_light, light_region, color=(1, 0, 0))

    # draw_region(triangle_onto, onto_centroid_shadow, color=(0, 0, 0))
    # draw_region(triangle_onto, onto_some_blocking, color=(1, 0, 0))
    # draw_region(triangle_onto, onto_totally_blocked, color=(0, 1, 0))
    # draw_region(triangle_onto, onto_from_each_point_0, color=(0, 0, 0))
    # draw_region(triangle_onto, onto_from_each_point_1, color=(0, 0, 0))
    # draw_region(triangle_onto, onto_from_each_point_2, color=(0, 0, 0))

    # draw_region(triangle_light, onto_light_centroid_shadow, color=(0, 0, 0))
    # draw_region(triangle_light, onto_light_some_blocking, color=(1, 0, 0))
    # draw_region(triangle_light, onto_light_totally_blocked, color=(0, 1, 0))
    # draw_region(triangle_light, onto_light_from_each_point_0, color=(0, 0, 0))
    # draw_region(triangle_light, onto_light_from_each_point_1, color=(0, 0, 0))
    # draw_region(triangle_light, onto_light_from_each_point_2, color=(0, 0, 0))

    # connect_tris(triangle_0, triangle_1, color=(1, 1, 0))

    mlab.show()


if __name__ == "__main__":
    main()
