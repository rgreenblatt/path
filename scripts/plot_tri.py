import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import numpy as np

ax = a3.Axes3D(plt.figure())
vtx = np.array([
    [-0.240000, 1.800000, 0.160000],
    [-0.240000, 1.800000, -0.220000],
    [0.230000, 1.800000, -0.220000],
])

tri = a3.art3d.Poly3DCollection([vtx])
tri.set_edgecolor('k')
ax.add_collection3d(tri)
ax.set_xlim(-1., 1.)
ax.set_ylim(0., 2.5)
ax.set_zlim(-1., 1.)
origin = np.array([-0.577188, 0.000000, 0.609341])
direction = 3. * np.array([0.351965, 0.860200, -0.369020])
ax.quiver(*origin, *direction)
plt.show()

def intersectionRayModel(self, rayStart, rayEnd):
    w = rayEnd - rayStart
    w /= np.linalg.norm(w)

    data = self.temp_mesh

    counter = 0
    for tri in data.vectors+self.pos:
        b = [.0, .0, .0]

        e1 = tri[1]
        e1 -= tri[0]
        e2 = tri[2]
        e2 -= tri[0]

        n = self.temp_mesh.normals[counter]

        q = np.cross(w, e2)
        a = np.dot(e1, q)

        counter += 1
        if (np.dot(n, w) >= .0) or (abs(a) <= .0001):
            continue

        s = np.array(rayStart)
        s -= tri[0]
        s /= a

        r = np.cross(s, e1)
        b[0] = np.dot(s, q)
        b[1] = np.dot(r, w)
        b[2] = 1.0 - b[0] - b[1]

        if (b[0] < .0) or (b[1] < .0) or (b[2] < .0):
            continue

        t = np.dot(e2, r)
        if t >= .0:
            point = rayStart + t*w
            return True, point
        else:
            continue

    return False, None
