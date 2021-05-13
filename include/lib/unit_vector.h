#pragma once

#include "lib/assert.h"
#include "lib/attribute.h"
#include "lib/cuda/utils.h"

#include <Eigen/Core>

#include <cmath>

template <typename T> class UnitVectorGen {
public:
  HOST_DEVICE UnitVectorGen() : v_({1., 0., 0.}) {}

  ATTR_PURE_NDEBUG HOST_DEVICE static UnitVectorGen
  new_normalize(const Eigen::Vector3<T> &v) {
    return UnitVectorGen(v.normalized());
  }

  ATTR_PURE_NDEBUG HOST_DEVICE static UnitVectorGen
  new_unchecked(const Eigen::Vector3<T> &v) {
    // TODO: epsilon?
    debug_assert(std::abs(v.norm() - 1.) < 1e-6);
    return UnitVectorGen(v);
  }

  ATTR_PURE_NDEBUG HOST_DEVICE const Eigen::Vector3<T> &operator*() const {
    return v_;
  }

  HOST_DEVICE const Eigen::Vector3<T> *operator->() const { return &v_; }

private:
  HOST_DEVICE explicit UnitVectorGen(const Eigen::Vector3<T> &v) : v_(v) {}

  Eigen::Vector3<T> v_;
};

using UnitVector = UnitVectorGen<float>;
