#pragma once

template <typename T> struct StartEnd {
  T start;
  T end;

  T size() const { return end - start; }
};
