#pragma once

#include "data_structure/get_ptr.h"
#include "data_structure/get_size.h"

#include <assert.h>
#include <concepts>
#include <type_traits>

template <typename Elem, bool is_sized = false> class Span;

template <typename Elem, bool is_sized, typename V>
concept SpanConvertable = requires {
  GetPtr<V, Elem>;
  !is_sized || GetSize<V>;
};

template <typename T, bool is_sized> class Span {
public:
  constexpr Span(T *ptr, std::size_t size) : ptr_(ptr) {
    if constexpr (use_size) {
      size_ = size;
    }
  }

  template <typename V>
  requires SpanConvertable<T, is_sized, V> constexpr Span(V &&v)
      : ptr_(GetPtrT<V, T>::get(v)) {
    if constexpr (use_size) {
      size_ = GetSizeT<V>::get(v);
    }
  }

  constexpr Span() = default;

  constexpr bool empty() const { return size() == 0; }

  constexpr std::size_t size() const {
    static_assert(is_sized, "size method can't be used if span isn't sized");
    return size_;
  }

  constexpr bool checkIndex(const std::size_t index) const {
    if constexpr (is_debug) {
      return index < size_;
    } else {
      return true;
    }
  }

  constexpr T &operator[](const std::size_t index) const {
    assert(checkIndex(index));
    return ptr_[index];
  }

  constexpr T *data() const { return ptr_; }

  constexpr T *begin() const { return ptr_; }

  constexpr T *end() const { return ptr_ + size(); }

  constexpr Span<T, true> slice(std::size_t start, std::size_t end) const {
    if constexpr (use_size) {
      assert(end <= size_);
    }

    return {ptr_ + start, end - start};
  }

private:
  T *ptr_;

  struct NoSize {};

  static constexpr bool is_debug =
#ifdef NDEBUG
      false
#else
      true
#endif
      ;
  static constexpr bool use_size = is_sized || is_debug;

  typename std::conditional_t<use_size, std::size_t, NoSize> size_;
};

template <typename Elem, typename Base>
struct GetSizeTraitImpl<Base, Span<Elem, true>> : Base {
  static constexpr unsigned get(const Span<Elem, true> &v) { return v.size(); }
};

template <typename Elem, bool is_sized, typename Base>
struct GetPtrTraitImpl<Base, Span<Elem, is_sized>> : Base {
  static constexpr unsigned get(const Span<Elem, is_sized> &v) {
    return v.data();
  }
};

template <typename T> using SpanSized = Span<T, true>;
