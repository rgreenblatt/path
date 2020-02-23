#pragma once

#include "execution_model/execution_model.h"
#include "execution_model/vector_type.h"

// Ideally this would be a "real" trait (see lib/trait.h)
namespace execution_model {
namespace detail {
template <ExecutionModel execution_model, typename T> struct exec_vector;

template <typename T> struct exec_vector<ExecutionModel::CPU, T> {
  using type = HostVector<T>;
};

template <typename T> struct exec_vector<ExecutionModel::GPU, T> {
  using type = DeviceVector<T>;
};

template <ExecutionModel execution_model, typename T> struct shared_vector;

template <typename T> struct shared_vector<ExecutionModel::CPU, T> {
  using type = HostVector<T>;
};

template <typename T> struct shared_vector<ExecutionModel::GPU, T> {
  using type = HostDeviceVector<T>;
};
} // namespace detail
} // namespace execution_model

template <ExecutionModel execution_model, typename T>
using ExecVector =
    typename execution_model::detail::exec_vector<execution_model, T>::type;

template <ExecutionModel execution_model, typename T>
using SharedVector =
    typename execution_model::detail::shared_vector<execution_model, T>::type;
