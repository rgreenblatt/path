from itertools import product, combinations

import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import numpy as np


def draw_tri(ax, vertices, edge_color='k'):
    tri = a3.art3d.Poly3DCollection([vertices])
    tri.set_edgecolor(edge_color)
    ax.add_collection3d(tri)


def connect_tris(ax, vertices_first, vertices_second, edge_color='k'):
    for i in range(3):
        for j in range(3):
            ax.plot3D(*zip(vertices_first[i], vertices_second[j]),
                      color=edge_color)


def draw_aabb(ax, min_bound, max_bound, color='b'):
    grid = 2
    for (axis, edge_next, edge_next_next) in product([0, 1, 2],
                                                     range(grid + 1),
                                                     range(grid + 1)):
        s = [0] * len(min_bound)
        e = [0] * len(min_bound)

        s[axis] = min_bound[axis]
        e[axis] = max_bound[axis]

        next_axis = (axis + 1) % 3
        next_next = (axis + 2) % 3

        s[next_axis] = e[next_axis] = (
            max_bound[next_axis] -
            min_bound[next_axis]) * edge_next / grid + min_bound[next_axis]
        s[next_next] = e[next_next] = (
            max_bound[next_next] - min_bound[next_next]
        ) * edge_next_next / grid + min_bound[next_next]
        ax.plot3D(*zip(s, e), color=color)


def main():
    ax = a3.Axes3D(plt.figure())

    triangle_0 = [
        [0.441879, -0.528226, 0.629064],
        [0.997233, -0.289026, -0.629806],
        [-0.337353, 0.0197618, -0.0778064],
    ]

    triangle_1 = [
        [-0.388798, 0.489003, 0.927466],
        [0.0411417, -0.214853, 0.393545],
        [0.916666, 0.81895, -0.494201],
    ]

    triangle_2 = [
        [-0.101216, -0.1975, -0.51163],
        [0.928564, 0.966285, -0.428953],
        [0.330397, -0.306568, -0.741083],
    ]

    draw_tri(ax, triangle_0, edge_color='r')
    draw_tri(ax, triangle_1, edge_color='g')
    draw_tri(ax, triangle_2, edge_color='b')
    # connect_tris(ax, triangle_0, triangle_2, edge_color='g')

    ax.scatter([-0.641504], [ -2.92644], [-0.976747])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-2., 2.)
    ax.set_zlim(-2., 2.)

    # end = np.array([0.52142638, 0.36578992, 0.182400629])
    # origin = np.array([0.362145573, 0.0209254418, 0.572940052])
    # direction = end - origin
    # ax.quiver(*origin, *direction * 2, color='r')

    plt.show()


if __name__ == "__main__":
    main()
