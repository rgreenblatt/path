#pragma once

#include <lib/span.h>

#include <Eigen/Geometry>
#include <dbg.h>

#include <iostream>

#define DEBUG_PRINT

inline std::ostream &operator<<(std::ostream &s, const Eigen::Projective3f &v) {
  s << v.matrix();

  return s;
}

inline std::ostream &operator<<(std::ostream &s, const Eigen::Affine3f &v) {
  s << v.matrix();

  return s;
}

template <typename T, size_t size>
inline std::ostream &operator<<(std::ostream &out,
                                const std::array<T, size> &v) {
  if (!v.empty()) {
    out << "[\n";
    for (const auto &val : v) {
      out << val << "\n";
    }
    out << "]";
  }
  return out;
}

template <typename T, typename A>
inline std::ostream &operator<<(std::ostream &out, const std::vector<T, A> &v) {
  if (!v.empty()) {
    out << "[\n";
    for (const auto &val : v) {
      out << val << "\n";
    }
    out << "]";
  }
  return out;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const SpanSized<T> &v) {
  if (!v.empty()) {
    out << "[\n";
    for (const auto &val : v) {
      out << val << "\n";
    }
    out << "]";
  }
  return out;
}
