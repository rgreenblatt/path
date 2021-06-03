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
    triangle_from = [
              [ 0.44171092147919999,  0.97958673054417755,  0.35944446900864052],
              [ 0.48089353630695031,  0.68866118628502071,  0.88047589322505126],
              [ 0.91823547296558105,  0.216822133181651,  0.56518886580422334],
    ]
    triangle_onto = [
              [ 0.41436858807068588,  0.47469750523248472,  0.62351010605795909],
              [ 0.33800761754600556,  0.6747523222841264,  0.31720174491785691],
              [ 0.77834548175758289,  0.94957105144778786,  0.66252686813676853],
    ]
    triangle_blocking = [
              [ 0.013571642064720679,  0.62284609328422258,  0.67365963124622241],
              [ 0.97194499897214081,  0.87819346878129512,  0.50962437187988574],
              [ 0.055714693147866312,  0.45115921404325532,  0.019987672506660125],
    ]

    triangle_from = np.array(triangle_from)
    triangle_blocking = np.array(triangle_blocking)
    triangle_onto = np.array(triangle_onto)

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    draw_tri(triangle_onto, color=(0, 0, 1))
    draw_tri(triangle_blocking, color=(0, 0, 0))
    draw_tri(triangle_from, color=(1, 0, 0))

    mlab.show()


if __name__ == "__main__":
    main()
