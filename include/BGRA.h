#pragma once

#include <cstdint>

// A structure for a color.  Each channel is 8 bits [0-255].
struct BGRA {
  BGRA() : b(0), g(0), r(0), a(0) {}
  BGRA(uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha = 255)
      : b(blue), g(green), r(red), a(alpha) {}

  union {
    struct {
      uint8_t b, g, r, a;
    };
    uint8_t channels[4];
  };

  BGRA operator+(const BGRA &that) const {
    return BGRA(static_cast<uint8_t>(this->b + that.b),
                static_cast<uint8_t>(this->g + that.g),
                static_cast<uint8_t>(this->r + that.r),
                static_cast<uint8_t>(this->a + that.a));
  }

  BGRA operator-(const BGRA &that) const {
    return BGRA(static_cast<uint8_t>(this->b - that.b),
                static_cast<uint8_t>(this->g - that.g),
                static_cast<uint8_t>(this->r - that.r),
                static_cast<uint8_t>(this->a - that.a));
  }

  BGRA operator*(const BGRA &that) const {
    return BGRA(static_cast<uint8_t>(this->b * that.b),
                static_cast<uint8_t>(this->g * that.g),
                static_cast<uint8_t>(this->r * that.r),
                static_cast<uint8_t>(this->a * that.a));
  }

  BGRA operator/(const BGRA &that) const {
    return BGRA(static_cast<uint8_t>(this->b / that.b),
                static_cast<uint8_t>(this->g / that.g),
                static_cast<uint8_t>(this->r / that.r),
                static_cast<uint8_t>(this->a / that.a));
  }

  friend bool operator==(const BGRA &c1, const BGRA &c2) {
    return (c1.r == c2.r) && (c1.g == c2.g) && (c1.b == c2.b) && (c1.a == c2.a);
  }

  friend bool operator!=(const BGRA &c1, const BGRA &c2) {
    return !operator==(c1, c2);
  }
};
