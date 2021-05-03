from itertools import product, combinations

import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import numpy as np


def draw_tri(ax, vertices, edge_color='k'):
    tri = a3.art3d.Poly3DCollection([vertices])
    tri.set_edgecolor(edge_color)
    ax.add_collection3d(tri)


def draw_aabb(ax, min_bound, max_bound, color='b'):
    for (axis, is_min_next, is_min_next_next) in product([0, 1, 2],
                                                         [False, True],
                                                         [False, True]):
        s = [0] * len(min_bound)
        e = [0] * len(min_bound)

        s[axis] = min_bound[axis]
        e[axis] = max_bound[axis]

        next_axis = (axis + 1) % 3
        next_next = (axis + 2) % 3

        s[next_axis] = e[next_axis] = min_bound[
            next_axis] if is_min_next else max_bound[next_axis]
        s[next_next] = e[next_next] = min_bound[
            next_next] if is_min_next_next else max_bound[next_next]

        ax.plot3D(*zip(s, e), color=color)


def main():
    ax = a3.Axes3D(plt.figure())

    tri_1 = [
        [-0.823972, 0.278462, -0.201798],
        [0.247798, 0.561288, 0.556085],
        [0.551506, -0.0437729, 0.311236],
    ]

    tri_2 = [
        [0.918268, 0.91624, 0.994868],
        [-0.891941, 0.665027, -0.00640327],
        [-0.59231, 0.482882, -0.7319],
    ]

    draw_tri(ax, tri_1, edge_color='r')
    draw_tri(ax, tri_2, edge_color='b')

    # 1
    draw_aabb(ax, [-0.823972, -0.0437729, -0.201798],
              [0.551506, 0.561288, 0.556085],
              color='tab:blue')

    # 2
    draw_aabb(ax, [-0.891941, 0.482882, -0.7319],
              [0.918268, 0.91624, 0.994868],
              color='tab:red')
    # 3
    draw_aabb(ax, [-0.891941, 0.482882, -0.7319],
              [0.162979, 0.699622, 0.131484],
              color='tab:orange')

    # 4
    draw_aabb(ax, [-0.642653, 0.699561, 0.131484],
              [0.918268, 0.91624, 0.994868],
              color='tab:green')

    # # 5
    # draw_aabb(ax, [-0.370497, -0.859653, -0.0181154],
    #           [0.536472, -0.294918, 0.927199],
    #           color='tab:purple')

    # # 6
    # draw_aabb(ax, [-0.399936, -0.727924, 0.219983],
    #           [0.395036, -0.0408303, 0.9123],
    #           color='tab:brown')

    origin = np.array([-0.172138, 0.928332, -0.264846])
    direction = 3. * np.array([-0.109091, -0.461965, 0.880164])
    ax.quiver(*origin, *direction)
    plt.show()


if __name__ == "__main__":
    main()
