#pragma once

#include "data_structure/copyable_to_vec.h"
#include "data_structure/vector.h"
#include "lib/span.h"
#include "meta/all_values.h"
#include "meta/get_idx.h"

#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/zip.hpp>

// useful for a struct of vecs which all have the same size
template <template <typename> class VecT, AllValuesEnumerable E, typename... T>
requires((... && Vector<VecT<T>>)&&(AllValues<E>.size() == sizeof...(T) &&
                                    sizeof...(T) > 0)) class VectorGroup {
public:
  using FirstType = __type_pack_element<0, T...>;
  static constexpr bool all_types_same = (... && std::same_as<FirstType, T>);

  void resize_all(unsigned size) {
    for_each([=](auto &data) { data.resize(size); });
  }

  void clear_all() {
    for_each([=](auto &data) { data.clear(); });
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

  template <E value>
  SpanSized<__type_pack_element<get_idx(value), T...>> get() {
    return std::get<get_idx(value)>(data_);
  }

  template <E value>
  SpanSized<const __type_pack_element<get_idx(value), T...>> get() const {
    return std::get<get_idx(value)>(data_);
  }

  template <E value> using Tag = Tag<E, value>;

  template <E value>
  SpanSized<__type_pack_element<get_idx(value), T...>> get(Tag<value>) {
    return get<value>();
  }

  template <E value>
  SpanSized<const __type_pack_element<get_idx(value), T...>>
  get(Tag<value>) const {
    return get<value>();
  }

  SpanSized<FirstType> operator[](unsigned i) requires(all_types_same) {
    return data_[i];
  }

  template <template <typename> class OtherVecT>
  requires(... &&CopyableToVec<VecT<T>, OtherVecT<T>>) void copy_to_other(
      VectorGroup<OtherVecT, E, T...> &other) const {
    boost::hana::for_each(
        boost::hana::zip(data_, other.as_ptr_tuple()), [](const auto &item) {
          boost::hana::unpack(item, [](const auto &this_data, auto other_data) {
            copy_to_vec(this_data, *other_data);
          });
        });
  }

private:
  auto as_ptr_tuple() {
    return boost::hana::unpack(data_, [&](auto &...data) {
      return boost::hana::make_tuple(&data...);
    });
  }

  template <typename F> void for_each(F &&f) {
    boost::hana::for_each(data_, std::forward<F>(f));
  }

  using DataType = std::conditional_t<all_types_same,
                                      std::array<VecT<FirstType>, sizeof...(T)>,
                                      std::tuple<VecT<T>...>>;

  // can this be reduced somehow???
  template <template <typename> class OtherVecT, AllValuesEnumerable OtherE,
            typename... OtherT>
  requires((... && Vector<OtherVecT<OtherT>>)&&(
      AllValues<OtherE>.size() == sizeof...(OtherT) &&
      sizeof...(OtherT) > 0)) friend class VectorGroup;

  DataType data_;
};
