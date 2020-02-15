#pragma once

#include "lib/execution_model.h"
#include "ray/detail/accel/dir_tree/dir_tree.h"
#include "scene/light.h"
#include "scene/shape_data.h"

#include <memory>

namespace ray {
namespace detail {
namespace accel {
namespace dir_tree {
template <ExecutionModel execution_model> class DirTreeGeneratorImpl;

template <ExecutionModel execution_model> class DirTreeGenerator {
public:
  DirTreeGenerator();

  ~DirTreeGenerator();

  DirTreeLookup generate(SpanSized<const scene::ShapeData> shapes,
                         unsigned target_num_dir_trees,
                         const Eigen::Vector3f &min_bound,
                         const Eigen::Vector3f &max_bound);

private:
  DirTreeGeneratorImpl<execution_model> *ptr_;
};
} // namespace dir_tree
} // namespace accel
} // namespace detail
} // namespace ray
