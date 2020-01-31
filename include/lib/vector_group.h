#include "lib/span.h"

#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/hana/for_each.hpp>

template <template <typename> class VecT, typename... T> class VectorGroup {
public:
  static_assert(sizeof...(T) > 0);
  using FirstType = __type_pack_element<0, T...>;

  void resize_all(unsigned size) {
    boost::hana::for_each(data_, [=](auto &data) { data.resize(size); });
  }

  unsigned size() const { return std::get<0>(data_).size(); }

  template <unsigned i> Span<__type_pack_element<i, T...>> get() {
    return std::get<i>(data_);
  }

  Span<FirstType> operator[](unsigned i) { return data_[i]; }

private:
  using DataType =
      std::conditional_t<std::conjunction_v<std::is_same<FirstType, T>...>,
                         std::array<VecT<FirstType>, sizeof...(T)>,
                         std::tuple<VecT<T>...>>;
  DataType data_;
};
