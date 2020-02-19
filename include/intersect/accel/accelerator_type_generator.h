#pragma once

#include "intersect/accel/accelerator_type.h"
#include "intersect/accel/accelerator_type_settings.h"
#include "intersect/accel/loop_all.h"
#include "lib/execution_model.h"

namespace intersect {
namespace accel {
template <typename Object, ExecutionModel execution_model, AcceleratorType type>
class Generator;

template <typename Object, ExecutionModel execution_model>
class Generator<Object, execution_model, AcceleratorType::LoopAll> {
private:
  using InstanceType = LoopAll<execution_model, Object>;
  InstanceType instance_;

public:
  using RefType = typename InstanceType::RefType;

  auto gen(Span<const Object> objects, unsigned start, unsigned end,
           const Eigen::Vector3f &, const Eigen::Vector3f &,
           Settings<AcceleratorType::LoopAll>) {
    return instance_.gen(objects, start, end);
  };
};

template <typename Object, ExecutionModel execution_model>
class Generator<Object, execution_model, AcceleratorType::KDTree> {
private:
  using InstanceType = LoopAll<execution_model, Object>;
  InstanceType instance_;

public:
  using RefType = typename InstanceType::RefType;

  auto gen(Span<const Object> objects, unsigned start, unsigned end,
           const Eigen::Vector3f &, const Eigen::Vector3f &,
           Settings<AcceleratorType::KDTree>) {
    return instance_.gen(objects, start, end);
  };
};

template <typename Object, ExecutionModel execution_model>
class Generator<Object, execution_model, AcceleratorType::DirTree> {
private:
  using InstanceType = LoopAll<execution_model, Object>;
  InstanceType instance_;

public:
  using RefType = typename InstanceType::RefType;

  auto gen(Span<const Object> objects, unsigned start, unsigned end,
           const Eigen::Vector3f &, const Eigen::Vector3f &,
           Settings<AcceleratorType::DirTree>) {
    return instance_.gen(objects, start, end);
  };
};
} // namespace accel
} // namespace intersect
