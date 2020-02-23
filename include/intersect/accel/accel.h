#pragma once

#include "execution_model/execution_model.h"
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

template <AccelType type>
struct AccelSettings;

// why can't I use Object O? (concept cannot have associated constraints)
template <AccelType type, ExecutionModel execution_model, typename O>
concept Accel = requires {
  Object<O>;
  typename AccelImpl<type, execution_model, O>;
  std::default_initializable<AccelImpl<type, execution_model, O>>;

  // reference type which is used to actually find intersections
  typename AccelImpl<type, execution_model, O>::Ref;
  Object<typename AccelImpl<type, execution_model, O>::Ref>;

  // type aliases in the reference are used to map from reference types
  // to implementations
  typename AccelImpl<type, execution_model, O>::Ref::InstO;
  AccelImpl<type, execution_model, O>::Ref::inst_type == type;
  AccelImpl<type, execution_model, O>::Ref::inst_execution_model ==
      execution_model;
  std::same_as<typename AccelImpl<type, execution_model, O>::Ref::InstO, O>;

  // Settings type is the same for each execution model and object
  typename AccelSettings<type>;

  // generation
  requires requires(AccelImpl<type, execution_model, O> & accel,
                    const AccelSettings<type> &settings, Span<const O> objects,
                    unsigned start, unsigned end, const AABB &aabb) {
    { accel.gen(settings, objects, start, end, aabb) }
    ->std::convertible_to<typename AccelImpl<type, execution_model, O>::Ref>;
  };

  // indexing (for the reference)
  requires requires(
      const typename AccelImpl<type, execution_model, O>::Ref &accel_ref,
      unsigned idx) {
    { accel_ref.get(idx) }
    ->std::convertible_to<O>;
  };
};

template <AccelType accel_type, ExecutionModel execution_model, Object O>
requires Accel<accel_type, execution_model, O> struct AccelChecked {
  using type = AccelImpl<accel_type, execution_model, O>;
};

template <AccelType type, ExecutionModel execution_model, Object O>
using AccelT = typename AccelChecked<type, execution_model, O>::type;

template <AccelType type, typename V> concept RefSpecialization = requires {
  typename V::InstO;
  { V::inst_execution_model }
  ->std::common_with<ExecutionModel>;
  { V::inst_type }
  ->std::common_with<AccelType>;
  std::same_as<
      V, typename accel::AccelT<type, V::inst_execution_model, typename V::InstO>::Ref>;
}
&&V::inst_type == type;

template<typename V> concept AccelRef = requires {
  { V::inst_type }
  ->std::common_with<AccelType>;
  RefSpecialization<V::inst_type, V>;
};
} // namespace accel
} // namespace intersect
