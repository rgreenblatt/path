#include "data_structure/copyable.h"
#include "data_structure/vector.h"
#include "lib/span.h"
#include "meta/concepts.h"

#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/zip.hpp>

// useful for a struct of vecs which all have the same size
template <template <typename> class VecT, typename... T>
requires(... &&Vector<VecT<T>>) class VectorGroup  {
public:
  static_assert(sizeof...(T) > 0);
  using FirstType = __type_pack_element<0, T...>;

  void resize_all(unsigned size) {
    boost::hana::for_each(data_, [=](auto &data) { data.resize(size); });
  }

  void push_back_all(const T &...vs) {
    boost::hana::for_each(
        boost::hana::zip(as_ptr_tuple(), boost::hana::make_tuple(vs...)),
        [](const auto &item) {
          boost::hana::unpack(
              item, [](auto data, const auto &item) { data->push_back(item); });
        });
  }

  unsigned size() const { return std::get<0>(data_).size(); }

  template <unsigned i> SpanSized<__type_pack_element<i, T...>> get() {
    return std::get<i>(data_);
  }

  SpanSized<FirstType> operator[](unsigned i) { return data_[i]; }

  template <template <typename> class OtherVecT>
  requires(... &&Copyable<VecT<T>, OtherVecT<T>>) void copy_to_other(
      VectorGroup<OtherVecT, T...> &other) const {
    boost::hana::for_each(
        boost::hana::zip(data_, other.as_ptr_tuple()), [](const auto &item) {
          boost::hana::unpack(item, [](const auto &this_data, auto other_data) {
            copy_to(this_data, *other_data);
          });
        });
  }

private:
  auto as_ptr_tuple() {
    return boost::hana::unpack(data_, [&](auto &...data) {
      return boost::hana::make_tuple(&data...);
    });
  }

  using DataType =
      std::conditional_t<std::conjunction_v<std::is_same<FirstType, T>...>,
                         std::array<VecT<FirstType>, sizeof...(T)>,
                         std::tuple<VecT<T>...>>;

  template <template <typename> class OtherVecT, typename... OtherT>
  requires(... &&Vector<OtherVecT<OtherT>>) friend class VectorGroup;

  DataType data_;
};
