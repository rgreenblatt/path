#pragma once

#include "lib/optional.h"

#include <chrono>
#include <iostream>
#include <optional>

class Timer {
private:
  using TimePoint = std::chrono::high_resolution_clock::time_point;

public:
  Timer() { start(); }
  Timer(std::optional<TimePoint> start) : start_(start) {}

  double elapsed() {
    auto op_time = optional_map(start_, [](TimePoint start) {
      return std::chrono::duration_cast<std::chrono::duration<double>>(now() -
                                                                       start)
          .count();
    });
    return total_ + optional_unwrap_or(op_time, 0.);
  }

  void stop() {
    total_ = elapsed();
    start_ = std::nullopt;
  }

  void start() { start_ = now(); }

  void report(const std::string &name) {
    std::cout << name << ": " << elapsed() << std::endl;
  }

private:
  static TimePoint now() { return std::chrono::high_resolution_clock::now(); }

  std::optional<TimePoint> start_ = std::nullopt;
  double total_ = 0.;
};
