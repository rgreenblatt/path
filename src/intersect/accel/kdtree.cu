#include "intersect/accel/kdtree.h"
#include "intersect/accel/kdtree/generator.h"
#include "intersect/impl/triangle_impl.h"
#include "intersect/transformed_object.h"

namespace intersect {
namespace accel {
template <ExecutionModel execution_model, Object O>
AccelImpl<AccelType::KDTree, execution_model, O>::AccelImpl() {
  gen_ = new kdtree::Generator<execution_model>();
}

template <ExecutionModel execution_model, Object O>
AccelImpl<AccelType::KDTree, execution_model, O>::~AccelImpl() {
  if (gen_ != nullptr) {
    delete gen_;
  }
}

template <ExecutionModel execution_model, Object O>
typename AccelImpl<AccelType::KDTree, execution_model, O>::Ref
AccelImpl<AccelType::KDTree, execution_model, O>::gen(
    const AccelSettings<AccelType::KDTree> &settings, Span<const O> objects,
    unsigned start, unsigned end, const AABB &aabb) {
  unsigned size = end - start;
  auto subsection = objects.slice(start, end);

  for (unsigned i = 0; i < size; ++i) {
    const auto &aabb = BoundedT<O>::bounds(objects[i]);
    bounds_[i] = {aabb, aabb.get_max_bound() - aabb.get_min_bound()};
  }

  auto [nodes, permuation] = gen_->gen(settings.generate, bounds_);

  local_idx_to_global_idx_.resize(size);
  ordered_objects_.resize(size);

  {
    auto start_it = thrust::make_counting_iterator(0u);
    thrust::copy(thrust_data.execution_policy(), start_it, start_it + size,
                 thrust::make_permutation_iterator(
                     local_idx_to_global_idx_.data(), permuation.data()));
  }
  {
    auto start_it = thrust::make_permutation_iterator(subsection.begin(),
                                                      permuation.begin());

    thrust::copy(thrust_data.execution_policy(), start_it, start_it + size,
                 ordered_objects_.data());
  }

  return Ref(nodes, ordered_objects_, start, permuation,
             local_idx_to_global_idx_, aabb);
}

template struct AccelImpl<AccelType::KDTree, ExecutionModel::CPU,
                          intersect::Triangle>;
template struct AccelImpl<AccelType::KDTree, ExecutionModel::CPU,
                          intersect::TransformedObject>;
template struct AccelImpl<AccelType::KDTree, ExecutionModel::GPU,
                          intersect::Triangle>;
template struct AccelImpl<AccelType::KDTree, ExecutionModel::GPU,
                          intersect::TransformedObject>;
} // namespace accel
} // namespace intersect
