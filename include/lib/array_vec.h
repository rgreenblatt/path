#pragma once

#include "data_structure/get_ptr.h"
#include "data_structure/get_size.h"
#include "lib/assert.h"
#include "lib/attribute.h"
#include "meta/mock.h"

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>

// fixed capacity vector (backed by array)
// most vector functions aren't implemented right now, but they could be...
// also, this only supports semiregular types for now
// consider if it would be more efficient to only copy up to the
// size of the ArrayVec when copying/moving
template <std::semiregular T, unsigned max_size_in> class ArrayVec {
public:
  constexpr static unsigned max_size = max_size_in;

  constexpr ArrayVec() : size_(0) {}

  constexpr void push_back(const T &v) {
    debug_assert_assume(size_ < max_size);
    arr_[size_] = v;
    size_++;
  }

  constexpr void resize(unsigned new_size) {
    debug_assert_assume(new_size <= max_size);
    size_ = new_size;
  }

  constexpr void clear() { resize(0); }

  ATTR_PURE_NDEBUG constexpr const T &operator[](unsigned idx) const {
    debug_assert_assume(idx < size_);
    return arr_[idx];
  }

  ATTR_PURE_NDEBUG constexpr T &operator[](unsigned idx) {
    debug_assert_assume(idx < size_);
    return arr_[idx];
  }

  // this might not matter too much...
  using SizeType = std::conditional_t<
      (max_size > std::numeric_limits<uint16_t>::max()), unsigned,
      std::conditional_t<(max_size > std::numeric_limits<uint8_t>::max()),
                         uint16_t, uint8_t>>;

  ATTR_PURE constexpr SizeType size() const { return size_; }

  ATTR_PURE constexpr bool empty() const { return size_ == 0; }

  ATTR_PURE_NDEBUG constexpr T *data() { return arr_.data(); }
  ATTR_PURE_NDEBUG constexpr const T *data() const { return arr_.data(); }
  ATTR_PURE_NDEBUG constexpr auto begin() { return arr_.begin(); }
  ATTR_PURE_NDEBUG constexpr auto begin() const { return arr_.begin(); }
  ATTR_PURE_NDEBUG constexpr auto end() { return arr_.begin() + size(); }
  ATTR_PURE_NDEBUG constexpr auto end() const { return arr_.begin() + size(); }

private:
  std::array<T, max_size> arr_;

  SizeType size_;
};

template <typename> struct is_array_vec : std::false_type {};

template <typename T, unsigned max_size>
struct is_array_vec<ArrayVec<T, max_size>> : std::true_type {};

template <typename T>
concept ArrayVecSpecialization = is_array_vec<std::decay_t<T>>::value;

template <ArrayVecSpecialization T> struct GetPtrImpl<T> {
  ATTR_PURE_NDEBUG static constexpr auto get(T &&v) { return v.data(); }
};
template <ArrayVecSpecialization T> struct GetSizeImpl<T> {
  ATTR_PURE_NDEBUG constexpr static auto get(const T &v) { return v.size(); }
};

static_assert(GetPtrForElem<ArrayVec<MockSemiregular, 0>, MockSemiregular>);
static_assert(
    GetPtrForElem<const ArrayVec<MockSemiregular, 0>, const MockSemiregular>);
static_assert(GetSize<ArrayVec<MockSemiregular, 0>>);
