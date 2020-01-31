#pragma once

#include "lib/cuda/managed_mem_vec.h"
#include "lib/execution_model.h"

#include <thrust/device_vector.h>
#include <vector>

template <typename T> using HostVectorType = std::vector<T>;
template <typename T> using DeviceVectorType = thrust::device_vector<T>;
template <typename T> using HostDeviceVectorType = ManangedMemVec<T>;

template <ExecutionModel execution_model, typename T>
struct get_exec_vector_type;

template <typename T> struct get_exec_vector_type<ExecutionModel::CPU, T> {
  using type = HostVectorType<T>;
};

template <typename T> struct get_exec_vector_type<ExecutionModel::GPU, T> {
  using type = DeviceVectorType<T>;
};

template <ExecutionModel execution_model, typename T>
using ExecVectorType = typename get_exec_vector_type<execution_model, T>::type;

template <ExecutionModel execution_model, typename T>
struct get_shared_vector_type;

template <typename T> struct get_shared_vector_type<ExecutionModel::CPU, T> {
  using type = HostVectorType<T>;
};

template <typename T> struct get_shared_vector_type<ExecutionModel::GPU, T> {
  using type = HostDeviceVectorType<T>;
};

template <ExecutionModel execution_model, typename T>
using SharedVectorType =
    typename get_shared_vector_type<execution_model, T>::type;
