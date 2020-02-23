#pragma once

#include "intersect/accel/accelerator_type.h"
#include "intersect/accel/accelerator_type_settings.h"
#include "intersect/accel/loop_all.h"
#include "execution_model/execution_model.h"

namespace intersect {
namespace accel {
template <typename Object, ExecutionModel execution_model, AccelType type>
class Generator;

template <typename Object, ExecutionModel execution_model>
class Generator<Object, execution_model, AccelType::LoopAll> {
private:
  using InstanceType = LoopAll<execution_model, Object>;
  InstanceType instance_;

public:
  using RefType = typename InstanceType::RefType;

  auto gen(Span<const Object> objects, unsigned start, unsigned end,
           const Eigen::Vector3f &, const Eigen::Vector3f &,
           AccelSettings<AccelType::LoopAll>) {
    return instance_.gen(objects, start, end);
  };
};

template <typename Object, ExecutionModel execution_model>
class Generator<Object, execution_model, AccelType::KDTree> {
private:
  using InstanceType = LoopAll<execution_model, Object>;
  InstanceType instance_;

public:
  using RefType = typename InstanceType::RefType;

  auto gen(Span<const Object> objects, unsigned start, unsigned end,
           const Eigen::Vector3f &, const Eigen::Vector3f &,
           AccelSettings<AccelType::KDTree>) {
    return instance_.gen(objects, start, end);
  };
};

template <typename Object, ExecutionModel execution_model>
class Generator<Object, execution_model, AccelType::DirTree> {
private:
  using InstanceType = LoopAll<execution_model, Object>;
  InstanceType instance_;

public:
  using RefType = typename InstanceType::RefType;

  auto gen(Span<const Object> objects, unsigned start, unsigned end,
           const Eigen::Vector3f &, const Eigen::Vector3f &,
           AccelSettings<AccelType::DirTree>) {
    return instance_.gen(objects, start, end);
  };
};
} // namespace accel
} // namespace intersect
