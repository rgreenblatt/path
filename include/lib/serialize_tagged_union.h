#pragma once

#include "lib/serialize.h"
#include "lib/tagged_union.h"
#include "meta/dispatch_value.h"
#include "meta/specialization_of.h"

#include <iostream>
#include <string>

template <typename Archive, SpecializationOf<TaggedUnion> T>
void save(Archive &ar, const T &v) {
  v.visit_indexed(
      [&](const auto &type, const auto &value) { ar(NVP(type), NVP(value)); });
}

template <typename Archive, SpecializationOf<TaggedUnion> T>
void load(Archive &ar, T &v) {
  decltype(v.type()) type;
  ar(NVP(type));
  dispatch_value(
      [&](auto tag) {
        std::decay_t<decltype(v.get(tag))> value;
        ar(NVP(value));
        v = T(tag, value);
      },
      type);
}
