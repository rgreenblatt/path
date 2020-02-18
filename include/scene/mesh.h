#pragma once

/* #include "triangle.h" */
#include "scene/CS123SceneData.h"

#include <Eigen/StdVector>
#include <tiny_obj_loader.h>

#include <vector>

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3i)

namespace scene {
class Mesh {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Mesh(const std::vector<Eigen::Vector3f> &vertices,
       const std::vector<Eigen::Vector3f> &normals,
       const std::vector<Eigen::Vector2f> &uvs,
       const std::vector<Eigen::Vector3f> &colors,
       const std::vector<Eigen::Vector3i> &faces,
       const std::vector<int> &materialIds,
       const std::vector<tinyobj::material_t> &materials);

  virtual ~Mesh();

  /* virtual BBox getBBox() const; */

  /* virtual Eigen::Vector3f getCentroid() const; */

  /* const Eigen::Vector3i getTriangleIndices(int faceIndex) const; */
  /* const tinyobj::material_t& getMaterial(int faceIndex) const; */

  /* const Eigen::Vector3f getVertex(int vertexIndex) const; */
  /* const Eigen::Vector3f getNormal(int vertexIndex) const; */
  /* const Eigen::Vector3f getColor(int vertexIndex) const; */
  /* const Eigen::Vector2f getUV(int vertexIndex) const; */

  /* virtual void setTransform(Eigen::Affine3f transform) override; */

private:
  // Properties fromt the scene file
  // CS123SceneMaterial _wholeObjectMaterial;

  // Properties from the .obj file
  std::vector<Eigen::Vector3f> _vertices;
  std::vector<Eigen::Vector3f> _normals;
  std::vector<Eigen::Vector3f> _colors;
  std::vector<Eigen::Vector2f> _uvs;
  std::vector<Eigen::Vector3i> _faces;
  std::vector<int> _materialIds;
  std::vector<tinyobj::material_t> _materials;

  Eigen::Vector3f _centroid;

  void calculateMeshStats();
  void createMeshBVH();
};
} // namespace scene
