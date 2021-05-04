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

    triangle_0 = [
        [-0.163629, -0.547184, -0.801778],
        [0.520279, 0.472582, -0.517931],
        [-0.356694, -0.898345, -0.723389],
    ]

    triangle_1 = [
        [0.558946, -0.721341, 0.732779],
        [-0.805455, -0.473149, -0.340121],
        [0.641068, -0.523222, -0.734293],
    ]

    triangle_2 = [
        [-0.805014, -0.737331, 0.750426],
        [-0.961558, -0.0543894, 0.138433],
        [0.570991, -0.0809908, 0.484386],
    ]

    triangle_3 = [
        [0.519129, -0.207841, -0.201789],
        [0.654039, 0.298654, 0.186167],
        [-0.663383, 0.119337, -0.379015],
    ]

    triangle_4 = [
        [-0.204109, -0.243385, -0.475869],
        [0.428201, 0.256143, 0.751666],
        [0.617557, 0.296879, -0.315627],
    ]

    triangle_5 = [
        [0.46532, 0.941424, -0.891032],
        [-0.0129349, -0.0601593, 0.977071],
        [-0.109235, -0.139625, 0.645591],
    ]

    triangle_6 = [
        [0.0183663, -0.0557863, -0.00292903],
        [0.961368, 0.0351143, 0.458465],
        [0.0897055, -0.445221, -0.324846],
    ]

    triangle_7 = [
        [0.824056, 0.857088, 0.999239],
        [-0.581915, -0.510128, -0.043038],
        [-0.351513, -0.704433, 0.911174],
    ]

    # draw_tri(ax, triangle_0, edge_color='r')
    # draw_tri(ax, triangle_1, edge_color='r')
    # draw_tri(ax, triangle_2, edge_color='r')
    # draw_tri(ax, triangle_3, edge_color='r')
    draw_tri(ax, triangle_4, edge_color='r')
    # draw_tri(ax, triangle_5, edge_color='r')
    # draw_tri(ax, triangle_6, edge_color='r')
    # draw_tri(ax, triangle_7, edge_color='r')

    # node: 0
    draw_aabb(ax, [-0.961558, -0.898345, -0.891032],
              [0.961368, 0.941424, 0.999239],
              color='tab:blue')

    # left node:
    # node: 1
    draw_aabb(ax, [-0.961558, -0.898345, -0.801778],
              [0.654039, 0.472582, 0.751666],
              color='tab:blue')

    # left node:
    # node: 3
    draw_aabb(ax, [-0.356694, -0.898345, -0.801778],
              [0.641068, 0.472582, -0.491089],
              color='tab:blue')

    # left node:
    # node: 5
    draw_aabb(ax, [-0.356694, -0.898345, -0.801778],
              [0.520279, 0.472582, -0.517931],
              color='tab:blue')
    # triangle idxs: 0,

    # right node:
    # node: 6
    draw_aabb(ax, [-0.251433, -0.556065, -0.734293],
              [0.641068, -0.492327, -0.491089],
              color='tab:blue')
    # triangle idxs: 1,

    # right node:
    # node: 4
    draw_aabb(ax, [-0.961558, -0.737331, -0.491089],
              [0.654039, 0.298654, 0.751666],
              color='tab:blue')

    # left node:
    # node: 7
    draw_aabb(ax, [-0.961558, -0.737331, -0.491089],
              [0.627454, -0.0543894, 0.750426],
              color='tab:blue')

    # left node:
    # node: 9
    draw_aabb(ax, [-0.805455, -0.721341, -0.491089],
              [0.627454, -0.473149, 0.732779],
              color='tab:blue')
    # triangle idxs: 1,

    # right node:
    # node: 10
    draw_aabb(ax, [-0.961558, -0.737331, 0.138433],
              [0.570991, -0.0543894, 0.750426],
              color='tab:blue')
    # triangle idxs: 2,

    # right node:
    # node: 8
    draw_aabb(ax, [-0.663383, -0.243385, -0.475869],
              [0.654039, 0.298654, 0.751666],
              color='tab:blue')

    # left node:
    # node: 11
    draw_aabb(ax, [-0.663383, -0.207841, -0.379015],
              [0.654039, 0.298654, 0.186167],
              color='tab:blue')
    # triangle idxs: 2,

    # right node:
    # node: 12
    draw_aabb(ax, [-0.204109, -0.243385, -0.475869],
              [0.617557, 0.296879, 0.751666],
              color='tab:blue')
    # triangle idxs: 3,

    # right node:
    # node: 2
    draw_aabb(ax, [-0.581915, -0.704433, -0.891032],
              [0.961368, 0.941424, 0.999239],
              color='tab:blue')

    # left node:
    # node: 13
    draw_aabb(ax, [-0.109235, -0.139625, -0.891032],
              [0.46532, 0.941424, 0.977071],
              color='tab:blue')
    # triangle idxs: 5,

    # right node:
    # node: 14
    draw_aabb(ax, [-0.581915, -0.704433, -0.324846],
              [0.961368, 0.857088, 0.999239],
              color='tab:blue')

    # left node:
    # node: 15
    draw_aabb(ax, [0.0183663, -0.445221, -0.324846],
              [0.961368, 0.0351143, 0.458465],
              color='tab:blue')
    # triangle idxs: 6,

    # right node:
    # node: 16
    draw_aabb(ax, [-0.581915, -0.704433, -0.043038],
              [0.824056, 0.857088, 0.999239],
              color='tab:blue')
    # triangle idxs: 7,

    origin = np.array([0.695482, -0.270496, -0.853243])
    direction = 3. * np.array([-0.257681, 0.546241, 0.797008])
    ax.quiver(*origin, *direction, color='r')
    plt.show()


if __name__ == "__main__":
    main()
