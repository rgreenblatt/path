from mayavi import mlab
import numpy as np


def draw_tri(vertices, color=(0, 0, 0)):
    mlab.triangular_mesh(vertices[:, 0],
                         vertices[:, 1],
                         vertices[:, 2], [[0, 1, 2]],
                         color=color)


def connect_tris(vertices_first, vertices_second, color=(0, 0, 0)):
    for i in range(3):
        for j in range(3):
            mlab.plot3d(*[[vertices_first[i][k], vertices_second[j][k]]
                          for k in range(3)],
                        color=color)


def main():
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

    (triangle_0, triangle_1, triangle_2) = (np.array(triangle_0),
                                            np.array(triangle_1),
                                            np.array(triangle_2))

    mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

    draw_tri(triangle_0, color=(0, 0, 1))
    draw_tri(triangle_1, color=(0, 1, 0))
    draw_tri(triangle_2, color=(1, 0, 0))
    connect_tris(triangle_0, triangle_2, color=(1, 1, 0))
    mlab.show()


if __name__ == "__main__":
    main()
