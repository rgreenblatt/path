#pragma once

#include "meta/enum.h"

#include <magic_enum.hpp>

#include <variant>

#if 0
// gross patched_visit which is actually constexpr by avoiding exception...
template <typename Visitor, typename... Variants>
constexpr decltype(auto) patched_visit(Visitor &&visitor,
                                       Variants &&...variants) {
  assert((... && !variants.valueless_by_exception()));

  using Res =
      std::invoke_result_t<Visitor,
                           decltype(std::get<0>(std::declval<Variants>()))...>;

  using Tag = std::__detail::__variant::__deduce_visit_result<Res>;

  return std::__do_visit<Tag>(std::forward<Visitor>(visitor),
                              std::forward<Variants>(variants)...);
}
#endif

// uses variant under the hood
template <Enum E, std::movable... T>
requires(magic_enum::enum_count<E>() == sizeof...(T)) class TaggedUnion {
public:
  TaggedUnion() {}

  template <E type>
  static TaggedUnion
  create(__type_pack_element<magic_enum::enum_integer(type), T...> value) {
    TaggedUnion out;
    out.var_ = std::variant<T...>{
        std::in_place_index<magic_enum::enum_integer(type)>, std::move(value)};

    return out;
  }

  constexpr E type() const {
    return magic_enum::enum_value<E>(var_.index());
  }

  template<typename F>
  constexpr decltype(auto) visit(F&& f) const {
    return std::visit(std::forward<F>(f), var_);
  }

  template<typename F>
  constexpr decltype(auto) visit(F&& f) {
    return std::visit(std::forward<F>(f), var_);
  }

  template <typename F> constexpr decltype(auto) visit_indexed(F &&f) {
    return visit([&](auto &&v) { return f(type(), v); });
  }

  template <typename F> constexpr decltype(auto) visit_indexed(F &&f) const {
    return visit([&](auto &&v) { return f(type(), v); });
  }

  constexpr auto operator<=>(const TaggedUnion &other) requires( ...
      &&std::totally_ordered<T>) { var_ <=> other.var_; }

private:
  std::variant<T...> var_;
};
