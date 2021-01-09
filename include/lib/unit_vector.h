#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <cmath>

class UnitVector {
public:
  HOST_DEVICE UnitVector() : v_({1.f, 0.f, 0.f}) {}

  ATTR_PURE_NDEBUG HOST_DEVICE static UnitVector
  new_normalize(const Eigen::Vector3f &v) {
    return UnitVector(v.normalized());
  }

  ATTR_PURE_NDEBUG HOST_DEVICE static UnitVector
  new_unchecked(const Eigen::Vector3f &v) {
    // TODO: epsilon?
    debug_assert(std::abs(v.norm() - 1.f) < 1e-6);
    return UnitVector(v);
  }

  ATTR_PURE_NDEBUG HOST_DEVICE const Eigen::Vector3f &operator*() const {
    return v_;
  }

  HOST_DEVICE const Eigen::Vector3f *operator->() const { return &v_; }

private:
  HOST_DEVICE explicit UnitVector(const Eigen::Vector3f &v) : v_(v) {}

  Eigen::Vector3f v_;
};