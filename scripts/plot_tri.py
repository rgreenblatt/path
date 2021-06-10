from functools import partial

from mayavi import mlab
import numpy as np


def draw_tri(vertices, color=(0, 0, 0)):
    mlab.triangular_mesh(vertices[:, 0],
                         vertices[:, 1],
                         vertices[:, 2], [[0, 1, 2]],
                         color=color)


def arrow(direction, point):
    mlab.quiver3d(
        point[0],
        point[1],
        point[2],
        direction[0],
        direction[1],
        direction[2],
    )


def plot_points(points, color=(0, 0, 0), scale_factor=0.1, is_line=False):
    points = np.array(points)
    f = mlab.plot3d if is_line else partial(mlab.points3d,
                                            scale_factor=scale_factor)
    f(points[:, 0], points[:, 1], points[:, 2], color=color)


def draw_centroid(tri, color=(0, 0, 0)):
    plot_points(tri.mean(axis=0, keepdims=True), color=color)


def draw_region(tri,
                region,
                color=(0, 0, 0),
                rays=np.array([]),
                is_line=False):
    if region is None:
        return
    if isinstance(region, str) and region == 'all':
        points = tri
    else:
        region = np.array(region)
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]

        points = (v0[None, :] * region[:, 0, None] +
                  v1[None, :] * region[:, 1, None] + tri[0])

        vert = points[0]
        for ray in rays:
            arrow(ray[0] * v0 + ray[1] * v1, vert)

    plot_points(points, color=color, scale_factor=0.05, is_line=is_line)


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
    l_triangle = [
        [0.15626, -1.16494, 1.36441],
        [0.132443, -1.0059, 0.61774],
        [0.12285, -1.55455, 1.0558],
    ]

    r_triangle = [
        [0.0631574, 1.43157, -2.22553],
        [0.630013, 1.85871, -2.3479],
        [-0.0416473, 1.8126, -1.41264],
    ]

    blocker_triangle_0 = [
        [0.0198179, 0.591691, -1.35748],
        [0.244881, 0.615893, -1.2751],
        [0.0768958, 0.553892, -1.12652],
    ]

    blocker_triangle_1 = [
        [0.0768958, 0.553892, -1.12652],
        [0.244881, 0.615893, -1.2751],
        [0.413177, 0.747155, -1.16342],
    ]

    blocker_triangle_2 = [
        [0.0768958, 0.553892, -1.12652],
        [0.413177, 0.747155, -1.16342],
        [0.122206, 0.639768, -0.906085],
    ]

    blocker_triangle_3 = [
        [0.122206, 0.639768, -0.906085],
        [0.413177, 0.747155, -1.16342],
        [0.479587, 0.950327, -1.05235],
    ]

    blocker_triangle_4 = [
        [0.122206, 0.639768, -0.906085],
        [0.479587, 0.950327, -1.05235],
        [0.143616, 0.826325, -0.755198],
    ]

    blocker_triangle_5 = [
        [0.143616, 0.826325, -0.755198],
        [0.479587, 0.950327, -1.05235],
        [0.426349, 1.17097, -0.97167],
    ]

    blocker_triangle_6 = [
        [0.143616, 0.826325, -0.755198],
        [0.426349, 1.17097, -0.97167],
        [0.135378, 1.06358, -0.714333],
    ]

    blocker_triangle_7 = [
        [0.244881, 0.615893, -1.2751],
        [0.317375, 0.910985, -1.51894],
        [0.413177, 0.747155, -1.16342],
    ]

    blocker_triangle_8 = [
        [0.413177, 0.747155, -1.16342],
        [0.317375, 0.910985, -1.51894],
        [0.368961, 1.13951, -1.46288],
    ]

    blocker_triangle_9 = [
        [0.413177, 0.747155, -1.16342],
        [0.368961, 1.13951, -1.46288],
        [0.479587, 0.950327, -1.05235],
    ]

    blocker_triangle_10 = [
        [0.479587, 0.950327, -1.05235],
        [0.368961, 1.13951, -1.46288],
        [0.330547, 1.3348, -1.32719],
    ]

    blocker_triangle_11 = [
        [0.189569, 0.710483, -1.48036],
        [-0.0693977, 0.967428, -1.61712],
        [0.317375, 0.910985, -1.51894],
    ]

    blocker_triangle_12 = [
        [0.317375, 0.910985, -1.51894],
        [-0.0693977, 0.967428, -1.61712],
        [-0.0776354, 1.20468, -1.57626],
    ]

    blocker_triangle_13 = [
        [0.317375, 0.910985, -1.51894],
        [-0.0776354, 1.20468, -1.57626],
        [0.368961, 1.13951, -1.46288],
    ]

    blocker_triangle_14 = [
        [0.368961, 1.13951, -1.46288],
        [-0.0776354, 1.20468, -1.57626],
        [-0.0562254, 1.39124, -1.42537],
    ]

    blocker_triangle_15 = [
        [0.368961, 1.13951, -1.46288],
        [-0.0562254, 1.39124, -1.42537],
        [0.330547, 1.3348, -1.32719],
    ]

    blocker_triangle_16 = [
        [0.330547, 1.3348, -1.32719],
        [-0.0562254, 1.39124, -1.42537],
        [-0.0109154, 1.47712, -1.20493],
    ]

    blocker_triangle_17 = [
        [-0.0688858, 1.7339, -0.671053],
        [0.51671, 1.1958, -1.85436],
        [0.506471, -0.941391, 0.859632],
    ]

    l_triangle = np.array(l_triangle)
    r_triangle = np.array(r_triangle)

    blocker_triangle_0 = np.array(blocker_triangle_0)
    blocker_triangle_1 = np.array(blocker_triangle_1)
    blocker_triangle_2 = np.array(blocker_triangle_2)
    blocker_triangle_3 = np.array(blocker_triangle_3)
    blocker_triangle_4 = np.array(blocker_triangle_4)
    blocker_triangle_5 = np.array(blocker_triangle_5)
    blocker_triangle_6 = np.array(blocker_triangle_6)
    blocker_triangle_7 = np.array(blocker_triangle_7)
    blocker_triangle_8 = np.array(blocker_triangle_8)
    blocker_triangle_9 = np.array(blocker_triangle_9)
    blocker_triangle_10 = np.array(blocker_triangle_10)
    blocker_triangle_11 = np.array(blocker_triangle_11)
    blocker_triangle_12 = np.array(blocker_triangle_12)
    blocker_triangle_13 = np.array(blocker_triangle_13)
    blocker_triangle_14 = np.array(blocker_triangle_14)
    blocker_triangle_15 = np.array(blocker_triangle_15)
    blocker_triangle_16 = np.array(blocker_triangle_16)
    blocker_triangle_17 = np.array(blocker_triangle_17)
    # blocker_triangle_18 = np.array(blocker_triangle_18)

    draw_tri(l_triangle, color=(1, 0, 0))
    draw_tri(r_triangle, color=(0, 0, 1))

    draw_tri(blocker_triangle_0, color=(0, 1, 0))
    draw_tri(blocker_triangle_1, color=(0, 1, 0))
    draw_tri(blocker_triangle_2, color=(0, 1, 0))
    draw_tri(blocker_triangle_3, color=(0, 1, 0))
    draw_tri(blocker_triangle_4, color=(0, 1, 0))
    draw_tri(blocker_triangle_5, color=(0, 1, 0))
    draw_tri(blocker_triangle_6, color=(0, 1, 0))
    draw_tri(blocker_triangle_7, color=(0, 1, 0))
    draw_tri(blocker_triangle_8, color=(0, 1, 0))
    draw_tri(blocker_triangle_9, color=(0, 1, 0))
    draw_tri(blocker_triangle_10, color=(0, 1, 0))
    draw_tri(blocker_triangle_11, color=(0, 1, 0))
    draw_tri(blocker_triangle_12, color=(0, 1, 0))
    draw_tri(blocker_triangle_13, color=(0, 1, 0))
    draw_tri(blocker_triangle_14, color=(0, 1, 0))
    draw_tri(blocker_triangle_15, color=(0, 1, 0))
    draw_tri(blocker_triangle_16, color=(0, 1, 0))
    draw_tri(blocker_triangle_17, color=(0, 1, 0))
    # draw_tri(blocker_triangle_18, color=(0, 1, 0))

    normal_0 = np.array([0.997747, -0.0516384, -0.042824])
    normal_1 = np.array([0.604999, -0.688138, 0.400552])

    point_0 = np.array([0.137184, -1.2418, 1.01265])
    point_1 = np.array([0.217174, 1.70096, -1.99536])

    partially_shadowed_on_l = 'all'
    partially_shadowed_on_r = [
        [
            [0.133458, -1.1111e-08],
            [0.143797, 0.0356703],
            [0.143813, 0.0357249],
            [0.189258, 0.192516],
            [0.199555, 0.228043],
            [0.231542, 0.338403],
            [0.273492, 0.483134],
            [0.274555, 0.486803],
            [0.274555, 0.486803],
            [0.293848, 0.553368],
            [0.293848, 0.553368],
            [0.301915, 0.581197],
            [0.328181, 0.671819],
            [0.49267, 0.50733],
            [0.707019, 0.292981],
            [0.949543, 0.0504569],
            [0.949543, 0.0504568],
            [1, -5.55112e-17],
            [0.133458, -1.1111e-08],
        ],
    ]
    totally_shadowed_on_l = [
        [
            [0, 0],
            [0, 0.439553],
            [0, 1],
            [1, 0],
            [0, 0],
        ],
    ]
    totally_shadowed_on_r = [
        [
            [0.83507, 0.0232517],
            [0.803554, 0.0893773],
            [0.58198, 0],
            [0.133458, 0],
            [0.156665, 0.0800663],
            [0.163284, 0.102904],
            [0.191098, 0.198865],
            [0.194955, 0.212173],
            [0.213927, 0.27763],
            [0.247473, 0.393368],
            [0.266099, 0.457627],
            [0.271079, 0.474812],
            [0.274555, 0.486803],
            [0.300101, 0.57494],
            [0.302767, 0.584139],
            [0.307354, 0.599963],
            [0.31067, 0.611404],
            [0.31786, 0.636211],
            [0.328181, 0.671819],
            [1, 0],
            [0.852032, 0],
            [0.840074, 0.0250071],
            [0.83507, 0.0232517],
        ],
    ]

    draw_region(l_triangle, totally_shadowed_on_l[0], is_line=True)
    draw_region(r_triangle, totally_shadowed_on_r[0], is_line=True)

    arrow(normal_0, point_0)
    arrow(normal_1, point_1)

    mlab.show()


if __name__ == "__main__":
    main()
