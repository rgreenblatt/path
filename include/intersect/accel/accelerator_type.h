#pragma once

#include <magic_enum.hpp>
#include <petra/enum_map.hpp>

namespace intersect {
namespace accel {
enum class AcceleratorType {
  LoopAll,
  KDTree,
  DirTree,
};

template <template <AcceleratorType> class TypeOver,
          unsigned size = magic_enum::enum_count<AcceleratorType>()>
class OnePerAcceleratorType {
public:
  static constexpr AcceleratorType this_type =
      magic_enum::enum_value<AcceleratorType>(size - 1);

  using ItemType = TypeOver<this_type>;

  template <AcceleratorType type> auto &get_item() {
    if constexpr (type == this_type) {
      return item_;
    } else {
      static_assert(size != 1, "enum value not found");
      return next.template get_item<type>();
    }
  }

private:
  ItemType item_;

  struct NoneType {};

  std::conditional_t<size == 1, NoneType,
                     OnePerAcceleratorType<TypeOver, size - 1>>
      next;
};

template <typename F>
auto run_over_accelerator_types(const F &f, AcceleratorType type) {
  // TODO: equivalent with magic_enum ideally
  auto get_result =
      petra::make_enum_map<AcceleratorType, AcceleratorType::LoopAll,
                           AcceleratorType::KDTree, AcceleratorType::DirTree>(
          [&](auto &&i) {
            using T = decltype(i);
            if constexpr (petra::utilities::is_error_type<T>()) {
              std::cerr << "invalid enum value!" << std::endl;
              abort();
            } else {
              return f(i);
            }
          });

  return get_result(type);
}
} // namespace accel
} // namespace intersect
