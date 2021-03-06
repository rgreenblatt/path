#pragma once

#include "data_structure/get_ptr.h"
#include "data_structure/get_size.h"
#include "lib/assert.h"
#include "lib/attribute.h"

#include <concepts>
#include <type_traits>

template <typename T, bool is_sized_in = false, bool is_debug = debug_build>
class Span;

namespace span_name {
namespace detail {
template <typename> struct IsSpanImpl : std::false_type {};

template <typename T, bool is_sized, bool is_debug>
struct IsSpanImpl<Span<T, is_sized, is_debug>> : std::true_type {};
} // namespace detail
} // namespace span_name

template <typename T>
concept SpanSpecialization =
    span_name::detail::IsSpanImpl<std::decay_t<T>>::value;

template <typename T, bool is_sized_in, bool is_debug_in> class Span {
public:
  constexpr static bool is_sized = is_sized_in;
  using ValueType = T;

private:
  constexpr static bool is_debug = is_debug_in;
  static constexpr bool use_size = is_sized_in || is_debug;

public:
  constexpr Span(T *ptr, std::size_t size) : ptr_(ptr) {
    if constexpr (use_size) {
      size_ = size;
    }
  }

  template <typename V>
  requires GetPtrForElem<V, T> &&
      (GetSize<V> ||
       (!is_sized && SpanSpecialization<V> &&
        std::decay_t<V>::is_debug == is_debug)) constexpr Span(V &&v)
      : ptr_(get_ptr(std::forward<V>(v))) {
    if constexpr (use_size) {
      if constexpr (GetSize<V>) {
        size_ = get_size(v);
      } else {
        size_ = v.size_;
        static_assert(SpanSpecialization<V>);
      }
    }
  }

  constexpr Span() = default;

  ATTR_PURE constexpr bool empty() const { return size() == 0; }

  ATTR_PURE constexpr std::size_t size() const requires is_sized {
    return size_;
  }

  ATTR_PURE_NDEBUG constexpr T &operator[](const std::size_t idx) const {
    debug_assert(check_index(idx));
    return ptr_[idx];
  }

  ATTR_PURE constexpr T *data() const { return ptr_; }

  ATTR_PURE constexpr T *begin() const { return ptr_; }

  ATTR_PURE constexpr T *end() const { return ptr_ + size(); }

  ATTR_PURE_NDEBUG constexpr Span<T, true, is_debug>
  slice(std::size_t start, std::size_t end) const {
    if constexpr (use_size) {
      debug_assert(end <= size_);
    }

    debug_assert(start <= end);

    return {ptr_ + start, end - start};
  }

  ATTR_PURE_NDEBUG constexpr Span<T, is_sized, is_debug>
  slice_from(std::size_t start) const {
    if constexpr (use_size) {
      debug_assert(start <= size_);

      return {ptr_ + start, size_ - start};
    } else {
      return {ptr_ + start};
    }
  }

  ATTR_PURE_NDEBUG constexpr Span<T, true, is_debug>
  slice_to(std::size_t end) const {
    if constexpr (use_size) {
      debug_assert(end <= size_);
    }

    return {ptr_, end};
  }

  constexpr void debug_assert_size_is(std::size_t size) const {
    if constexpr (is_debug) {
      return debug_assert(size == size_);
    }
  }

  ATTR_PURE_NDEBUG constexpr Span<T, false, is_debug> as_unsized() const {
    return *this;
  }

  ATTR_PURE_NDEBUG constexpr Span<const T, is_sized, is_debug>
  as_const() const {
    return *this;
  }

private:
  constexpr bool check_index(const std::size_t index) const {
    if constexpr (is_debug) {
      return index < size_;
    } else {
      return true;
    }
  }

  T *ptr_;

  struct NoSize {};

  [[no_unique_address]]
  typename std::conditional_t<use_size, std::size_t, NoSize>
      size_;

  template <typename TOther, bool is_sized_other, bool is_debug_other>
  friend class Span;
};

static_assert(sizeof(Span<int, false, false>) == sizeof(int *));
static_assert(sizeof(Span<int, true, false>) == 2 * sizeof(int *));
static_assert(sizeof(Span<int, false, true>) == 2 * sizeof(int *));

template <typename T> Span(T *ptr, std::size_t size) -> Span<T, true>;

template <typename V>
concept SpanConvertible = GetPtr<V> && GetSize<V>;

template <SpanConvertible V> Span(V &&) -> Span<PointerElemType<V>, true>;

template <typename T> using SpanSized = Span<T, true>;

template <typename T>
concept SizedSpanSpecialization = requires {
  requires SpanSpecialization<T>;
  requires std::decay_t<T>::is_sized;
};

template <SizedSpanSpecialization SpanT> struct GetSizeImpl<SpanT> {
  ATTR_PURE static constexpr std::size_t get(const SpanT &v) {
    return v.size();
  }
};

template <SpanSpecialization SpanT> struct GetPtrImpl<SpanT> {
  ATTR_PURE static constexpr auto get(SpanT &&v) { return v.data(); }
};

static_assert(!SizedSpanSpecialization<Span<bool>>);
static_assert(!GetSize<Span<bool>>);
static_assert(SizedSpanSpecialization<SpanSized<bool>>);
static_assert(GetSize<SpanSized<bool>>);
