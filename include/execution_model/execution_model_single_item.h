#pragma once

#include "data_structure/copyable_to_vec.h"
#include "execution_model/execution_model_vector_type.h"
#include "lib/span.h"

#include <concepts>

template <ExecutionModel exec, std::copyable T> class ExecSingleItem {
public:
  ExecSingleItem(T v) { set(v); }

  void set(T v) { copy_to_vec(std::array{v}, inner_); }

  Span<T, false> span() { return inner_; }
  Span<const T, false> span() const { return inner_; }

  T get() const {
    std::vector<T> out;
    copy_to_vec(inner_, out);
    return out[0];
  }

private:
  ExecVector<exec, T> inner_;
};
