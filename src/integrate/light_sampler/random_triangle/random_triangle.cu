#include "execution_model/execution_model_vector_type.h"
#include "integrate/light_sampler/random_triangle/random_triangle.h"

namespace integrate {
namespace light_sampler {
namespace random_triangle {
template <ExecutionModel exec> struct RandomTriangle<exec>::ExecStorage {
  template <typename T> using ExecVecT = ExecVector<exec, T>;
  TWGroup<ExecVecT> items_;
};

template <ExecutionModel exec> RandomTriangle<exec>::RandomTriangle() {
  exec_storage_ = std::make_unique<ExecStorage>();
}

template <ExecutionModel exec>
RandomTriangle<exec>::~RandomTriangle() = default;

template <ExecutionModel exec>
RandomTriangle<exec>::RandomTriangle(RandomTriangle &&) = default;

template <ExecutionModel exec>
RandomTriangle<exec> &
RandomTriangle<exec>::operator=(RandomTriangle &&) = default;

template <ExecutionModel exec>
detail::Ref
RandomTriangle<exec>::finish_gen_internal(const Settings &settings) {
  auto &items = exec_storage_->items_;
  host_items_.copy_to_other(items);

  return detail::Ref(settings, items.get(TAG(TWItem::Triangle)),
                     items.get(TAG(TWItem::Weight)));
}

template class RandomTriangle<ExecutionModel::CPU>;
template class RandomTriangle<ExecutionModel::GPU>;
} // namespace random_triangle
} // namespace light_sampler
} // namespace integrate
