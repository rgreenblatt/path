#pragma once

#include "data_structure/get_ptr.h"
#include "data_structure/get_size.h"

#include <assert.h>
#include <concepts>
#include <type_traits>

/* template <typename Elem, bool is_sized, typename V> */
/* concept SpanConvertable = requires { */
/*   GetPtr<V, Elem>; */
/*   !is_sized || GetSize<V>; */
/* }; */

template <typename T, bool is_sized = false> class Span {
private:
  static constexpr bool is_debug =
#ifdef NDEBUG
      false
#else
      true
#endif
      ;
  static constexpr bool use_size = is_sized || is_debug;

public:
  constexpr Span(T *ptr, std::size_t size) : ptr_(ptr) {
    if constexpr (use_size) {
      size_ = size;
    }
  }

  template <typename V>
      requires GetPtr<V, T> &&
      (!Span::use_size || GetSize<V>)constexpr Span(V &&v)
      : ptr_(GetPtrT<V, T>::get(std::forward<V>(v))) {
    if constexpr (use_size) {
      size_ = GetSizeT<V>::get(std::forward<V>(v));
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

  template<typename Base, typename V>
  friend struct GetSizeTraitImpl;

  struct NoSize {};

  typename std::conditional_t<use_size, std::size_t, NoSize> size_;
};

template <typename> struct is_span : std::false_type {};

template <typename T, bool is_sized>
struct is_span<Span<T, is_sized>> : std::true_type {};

template <typename V>
concept SpanSpecialization = is_span<std::decay_t<V>>::value;

template <typename SpanT, typename Base>
requires SpanSpecialization<SpanT> 
struct GetSizeTraitImpl<Base, SpanT> : Base {
  static constexpr unsigned get(SpanT &&v) { return v.size_; }
};

template <typename SpanT, typename Base>
requires SpanSpecialization<SpanT> 
struct GetPtrTraitImpl<Base, SpanT> : Base {
  static constexpr auto get(SpanT &&v) { return v.data(); }
};

template <typename T> using SpanSized = Span<T, true>;
