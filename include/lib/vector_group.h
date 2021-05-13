#pragma once

#include "data_structure/copyable_to_vec.h"
#include "data_structure/vector.h"
#include "lib/span.h"
#include "lib/tagged_tuple.h"
#include "meta/all_values/all_values.h"
#include "meta/all_values/tag.h"
#include "meta/pack_element.h"
#include "meta/tuple.h"

#include <boost/hana/zip.hpp>

// useful for a struct of vecs which all have the same size
template <template <typename> class VecT, AllValuesEnumerable E, typename... T>
requires((... && Vector<VecT<T>>)&&AllValues<E>.size() == sizeof...(T) &&
         sizeof...(T) > 0) class VectorGroup {
public:
  using FirstType = PackElement<0, T...>;
  static constexpr bool all_types_same = (... && std::same_as<FirstType, T>);

  void resize_all(unsigned size) {
    data_.for_each([=](auto &data) { data.resize(size); });
  }

  void clear_all() {
    data_.for_each([=](auto &data) { data.clear(); });
  }

  TaggedTuple<E, const T &...> get_all(unsigned idx) const {
    return boost::hana::unpack(data_.items, [&](auto &...values) {
      return TaggedTuple<E, const T &...>{{values[idx]...}};
    });
  }

  template <typename... V>
  void set_all_tup(unsigned idx, TaggedTuple<E, V...> values) {
    boost::hana::unpack(
        values.items, [&](const auto &...values) { set_all(idx, values...); });
  }

  void set_all(unsigned idx, const T &...vs) {
    boost::hana::for_each(
        boost::hana::zip(as_ptr_tuple(), boost::hana::make_tuple(vs...)),
        [&](const auto &item) {
          boost::hana::unpack(
              item, [&](auto data, const auto &item) { (*data)[idx] = item; });
        });
  }

  void push_back_all(const T &...vs) {
    boost::hana::for_each(
        boost::hana::zip(as_ptr_tuple(), boost::hana::make_tuple(vs...)),
        [](const auto &item) {
          boost::hana::unpack(
              item, [](auto data, const auto &item) { data->push_back(item); });
        });
  }

  unsigned size() const { return get(Tag<E, 0>{}).size(); }
  bool empty() const { return size() == 0; }

  template <unsigned idx>
  SpanSized<PackElement<idx, T...>> get(Tag<E, idx> tag) {
    return data_.get(tag);
  }

  template <unsigned idx>
  SpanSized<const PackElement<idx, T...>> get(Tag<E, idx> tag) const {
    return data_.get(tag);
  }

  template <template <typename> class OtherVecT>
  requires(... &&CopyableToVec<VecT<T>, OtherVecT<T>>) void copy_to_other(
      VectorGroup<OtherVecT, E, T...> &other) const {
    boost::hana::for_each(
        boost::hana::zip(data_.items, other.as_ptr_tuple()),
        [](const auto &item) {
          boost::hana::unpack(item, [](const auto &this_data, auto other_data) {
            copy_to_vec(this_data, *other_data);
          });
        });
  }

private:
  auto as_ptr_tuple() {
    return boost::hana::unpack(
        data_.items, [&](auto &...data) { return make_meta_tuple(&data...); });
  }

  // can this be reduced somehow???
  template <template <typename> class OtherVecT, AllValuesEnumerable OtherE,
            typename... OtherT>
  requires((... && Vector<OtherVecT<OtherT>>)&&AllValues<OtherE>.size() ==
               sizeof...(OtherT) &&
           sizeof...(OtherT) > 0) friend class VectorGroup;

  TaggedTuple<E, VecT<T>...> data_;
};
