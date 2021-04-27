#ifndef PROGRESSBAR_PROGRESSBAR_HPP
#define PROGRESSBAR_PROGRESSBAR_HPP

#include <chrono>
#include <iostream>

class ProgressBar {
private:
  unsigned ticks = 0;

  const unsigned total_ticks;
  const unsigned bar_width;
  const char complete_char = '=';
  const char incomplete_char = ' ';
  const std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();

public:
  ProgressBar(unsigned total, unsigned width, char complete, char incomplete)
      : total_ticks{total}, bar_width{width}, complete_char{complete},
        incomplete_char{incomplete} {}

  ProgressBar(unsigned total, unsigned width)
      : total_ticks{total}, bar_width{width} {}

  unsigned operator++() { return ++ticks; }

  unsigned operator+=(unsigned addr) { return ticks += addr; }

  void display() const {
    float progress = (float)ticks / total_ticks;
    unsigned pos = bar_width * progress;

    std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    auto time_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time)
            .count();

    std::cout << "[";

    for (unsigned i = 0; i < bar_width; ++i) {
      if (i < pos) {
        std::cout << complete_char;
      } else if (i == pos) {
        std::cout << ">";
      } else {
        std::cout << incomplete_char;
      }
    }
    std::cout << "] " << int(progress * 100.0) << "% "
              << float(time_elapsed) / 1000.0 << "s\r";
    std::cout.flush();
  }

  void done() const {
    display();
    std::cout << std::endl;
  }
};

#endif // PROGRESSBAR_PROGRESSBAR_HPP
