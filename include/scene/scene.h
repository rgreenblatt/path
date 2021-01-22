#pragma once

#include "intersect/accel/aabb.h"
#include "intersect/transformed_object.h"
#include "intersect/triangle.h"
#include "lib/span.h"
#include "lib/vector_group.h"
#include "meta/all_values/impl/enum.h"
#include "scene/emissive_cluster.h"
#include "scene/material.h"
#include "scene/triangle_data.h"

#include <vector>

namespace scene {
namespace scenefile_compat {
class ScenefileLoader;
}

// TODO: eventually scene should allow for non triangle
// scenes and alternate material
class Scene {
public:
  ATTR_PURE const Eigen::Affine3f &film_to_world() const {
    return film_to_world_;
  }

  using Triangle = intersect::Triangle;
  using TransformedObject = intersect::TransformedObject;
  using AABB = intersect::accel::AABB;

  ATTR_PURE_NDEBUG SpanSized<const unsigned> mesh_ends() const {
    return meshs_.get(tag_v<MeshT::End>);
  }

  ATTR_PURE_NDEBUG SpanSized<const AABB> mesh_aabbs() const {
    return meshs_.get(tag_v<MeshT::AABB>);
  }

  ATTR_PURE_NDEBUG SpanSized<const std::string> mesh_paths() const {
    return meshs_.get(tag_v<MeshT::Path>);
  }

  ATTR_PURE_NDEBUG SpanSized<const TransformedObject>
  transformed_mesh_objects() const {
    return transformed_objects_.get(tag_v<TransformedObjectT::Inst>);
  }

  ATTR_PURE_NDEBUG SpanSized<const unsigned> transformed_mesh_idxs() const {
    return transformed_objects_.get(tag_v<TransformedObjectT::MeshIdx>);
  }

  ATTR_PURE_NDEBUG SpanSized<const Triangle> triangles() const {
    return triangles_.get(tag_v<TriangleT::Inst>);
  }

  ATTR_PURE_NDEBUG SpanSized<const TriangleData> triangle_data() const {
    return triangles_.get(tag_v<TriangleT::Data>);
  }

  ATTR_PURE_NDEBUG SpanSized<const Material> materials() const {
    return materials_;
  }

  ATTR_PURE_NDEBUG SpanSized<const EmissiveCluster> emissive_clusters() const {
    return emissive_clusters_;
  }

  ATTR_PURE_NDEBUG SpanSized<const unsigned>
  emissive_cluster_ends_per_mesh() const {
    return meshs_.get(tag_v<MeshT::EmissiveClusterEnd>);
  }

  // Note: may not be very precise...
  ATTR_PURE const AABB &overall_aabb() const { return overall_aabb_; }

private:
  Scene() {}

  template <typename T> using Vec = std::vector<T>;

  enum class MeshT {
    End,
    AABB,
    Path,               // used as unique identifier
    EmissiveClusterEnd, // used as unique identifier
  };

  using MeshGroup =
      VectorGroup<Vec, MeshT, unsigned, AABB, std::string, unsigned>;

  enum class TransformedObjectT {
    Inst,
    MeshIdx,
  };

  using TransformedObjectGroup =
      VectorGroup<Vec, TransformedObjectT, TransformedObject, unsigned>;

  enum class TriangleT {
    Inst,
    Data,
  };

  using TriangleGroup = VectorGroup<Vec, TriangleT, Triangle, TriangleData>;

  enum class EmissiveClusterT {
    Inst,
    EndsPerMesh,
  };

  using EmissiveClusterGroup =
      VectorGroup<Vec, EmissiveClusterT, EmissiveCluster, unsigned>;

  MeshGroup meshs_;
  TransformedObjectGroup transformed_objects_;
  TriangleGroup triangles_;
  std::vector<Material> materials_;
  std::vector<EmissiveCluster> emissive_clusters_;

  intersect::accel::AABB overall_aabb_;

#if 0
  std::vector<CS123SceneLightData> lights_;
#endif

  Eigen::Affine3f film_to_world_;

  friend class scenefile_compat::ScenefileLoader;
};
} // namespace scene
