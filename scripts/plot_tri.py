from itertools import product, combinations

import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import numpy as np


def draw_tri(ax, vertices, edge_color='k'):
    tri = a3.art3d.Poly3DCollection([vertices])
    tri.set_edgecolor(edge_color)
    ax.add_collection3d(tri)


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
        [0.185689, 0.430379, 0.688532],
        [0.205527, 0.715891, 0.0897664],
        [0.694503, -0.15269, 0.247127],
    ]

    triangle_1 = [
        [0.291788, -0.231237, -0.124826],
        [-0.404931, 0.783546, -0.886574],
        [0.927325, -0.454687, -0.233117],
    ]

    draw_tri(ax, triangle_0, edge_color='r')
    draw_tri(ax, triangle_1, edge_color='r')

    min_bound = [-0.404931, -0.454687, -0.886574]
    max_bound = [0.927325, 0.783546, 0.688532]

    draw_aabb(ax, min_bound, max_bound)

    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)

    for direc, start in np.array([
        [[-2, 1, -1], [2, 1, 2]],
        [[-2, 1, -1], [2, 1, 1]],
        [[-2, 1, -1], [2, 0, 2]],
        [[-2, 1, -1], [2, 0, 1]],
    ]):
        grid = 2
        start = (max_bound - min_bound ) * start / grid + min_bound
        direc = (max_bound - min_bound ) * direc / grid

        ax.plot3D(*zip(start, start + direc), color='g')

    origin = np.array([0.914902, 0.0364014, -0.0564969])
    direction = 3. * np.array([-0.795732, 0.354946, -0.490738])
    origin -= direction / 3
    ax.quiver(*origin, *direction, color='r')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    main()
