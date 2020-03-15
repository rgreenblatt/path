#pragma once

#include "execution_model/execution_model.h"
#include "intersect/accel/kdtree/settings.h"
#include "intersect/accel/s_a_heuristic_settings.h"
#include "intersect/object.h"
#include "lib/span.h"

namespace intersect {
namespace accel {
enum class AccelType {
  LoopAll,
  KDTree,
  DirTree,
};

template <AccelType type, ExecutionModel execution_model, Object O>
struct AccelImpl;

template <AccelType type> struct AccelSettings;

template <typename V> concept AccelRef = requires {
  Object<V>;

  typename V::InstO;
  V::inst_type;
  V::inst_execution_model;

  // indexing (for the reference)
  requires requires(const V &accel_ref, unsigned idx) {
    { accel_ref.get(idx) }
    ->std::convertible_to<typename V::InstO>;
  };
};

template <typename V, AccelType type>
concept AccelRefOfType = AccelRef<V> &&type == V::inst_type;

// why can't I use Object O? (concept cannot have associated constraints)
template <AccelType type, ExecutionModel execution_model, typename O>
concept Accel = requires {
  Object<O>;
  typename AccelImpl<type, execution_model, O>;
  std::semiregular<AccelImpl<type, execution_model, O>>;

  // Settings type is the same for each execution model and object
  typename AccelSettings<type>;
  Setting<AccelSettings<type>>;
  std::equality_comparable<AccelSettings<type>>;

  // generation
  requires requires(AccelImpl<type, execution_model, O> & accel,
                    const AccelSettings<type> &settings, Span<const O> objects,
                    unsigned start, unsigned end, const AABB &aabb) {
    { accel.gen(settings, objects, start, end, aabb) }
    ->AccelRefOfType<type>;
  };
};

template <AccelType accel_type, ExecutionModel execution_model, Object O>
requires Accel<accel_type, execution_model, O> struct AccelChecked {
  using type = AccelImpl<accel_type, execution_model, O>;
};

template <AccelType type, ExecutionModel execution_model, Object O>
using AccelT = typename AccelChecked<type, execution_model, O>::type;

template <> struct AccelSettings<AccelType::LoopAll> : EmptySettings {
  HOST_DEVICE inline bool
  operator==(const AccelSettings<AccelType::LoopAll> &) const = default;
};

template <> struct AccelSettings<AccelType::KDTree> {
  kdtree::Settings generation_settings;
  
  template <class Archive> void serialize(Archive &archive) {
    archive(CEREAL_NVP(generation_settings));
  }

  HOST_DEVICE inline bool operator==(const AccelSettings &) const = default;
};

template <> struct AccelSettings<AccelType::DirTree> : EmptySettings {
  // TODO: this will change when dir tree is implemented
  /* SAHeuristicSettings s_a_heuristic_settings; */
  /* unsigned num_dir_trees; */

  HOST_DEVICE inline bool operator==(const AccelSettings &) const = default;
};
} // namespace accel
} // namespace intersect
